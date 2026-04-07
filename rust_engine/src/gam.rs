//! GAM Classifier — High-level interface

use nalgebra::{DVector, DMatrix};
use std::collections::HashMap;

use crate::bspline::{create_basis_matrix, create_penalty_matrix, FeatureType};
use crate::pirls::{pirls_logistic, find_best_lambda, gcv_score, logistic};

/// Kết quả training GAM
#[derive(Debug, Clone)]
pub struct GAMTrainingResult {
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub f1: f64,
    pub recall: f64,
    pub precision: f64,
    pub brier_score: f64,
    pub n_iterations: usize,
    pub converged: bool,
    pub best_lambda: f64,
    pub gcv_score: f64,
    pub feature_importance: Vec<FeatureImportance>,
}

/// Feature importance từ GAM
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    pub feature: String,
    pub variance_importance: f64,
    pub term_index: usize,
}

/// GAM Classifier
pub struct GAMClassifier {
    pub feature_types: Vec<FeatureType>,
    pub feature_names: Vec<String>,
    pub n_splines: usize,
    pub spline_degree: usize,
    pub coefficients: Option<DVector<f64>>,
    pub intercept: f64,
    pub lambdas: Vec<f64>,
    pub x_ranges: Vec<(f64, f64)>,
    pub nominal_mappings: HashMap<String, Vec<f64>>,
    pub penalties: Vec<DMatrix<f64>>,
    pub feature_col_ranges: Vec<(usize, usize)>,
    pub optimize_lambda: bool,
    pub lambda_grid: Vec<f64>,
}

impl GAMClassifier {
    pub fn new(
        feature_types: Vec<FeatureType>,
        feature_names: Vec<String>,
        n_splines: usize,
    ) -> Self {
        // Reduced lambda grid for speed
        let lambda_grid: Vec<f64> = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];

        GAMClassifier {
            feature_types,
            feature_names,
            n_splines,
            spline_degree: 3,
            coefficients: None,
            intercept: 0.0,
            lambdas: Vec::new(),
            x_ranges: Vec::new(),
            nominal_mappings: HashMap::new(),
            penalties: Vec::new(),
            feature_col_ranges: Vec::new(),
            optimize_lambda: true,
            lambda_grid,
        }
    }

    pub fn fit(
        &mut self,
        x_data: &[f64],
        y_data: &[f64],
        n_samples: usize,
        n_features: usize,
        feature_types: &[String],
        feature_names: &[String],
    ) -> GAMTrainingResult {
        let types: Vec<FeatureType> = feature_types.iter().map(|s| match s.as_str() {
            "numeric" => FeatureType::Numeric,
            "ordinal" => FeatureType::Ordinal,
            "nominal" => FeatureType::Nominal,
            _ => FeatureType::Numeric,
        }).collect();

        let (design_matrix, col_ranges, x_ranges_vec, penalties_vec) = 
            Self::build_design_matrix_static(x_data, n_samples, n_features, &types, self.n_splines, self.spline_degree);

        self.feature_col_ranges = col_ranges;
        self.x_ranges = x_ranges_vec;
        self.penalties = penalties_vec;

        let y = DVector::from_vec(y_data.to_vec());
        let n = design_matrix.nrows();

        // Step 1: Find best lambda per feature
        let mut best_lambdas = Vec::new();

        for feat_idx in 0..n_features {
            let (col_start, col_end) = self.feature_col_ranges[feat_idx];
            let feat_cols = design_matrix.columns(col_start, col_end - col_start).clone_owned();
            let penalty = &self.penalties[feat_idx];

            let (best_lambda, _) = if self.optimize_lambda && !penalty.is_empty() && penalty.trace().abs() > 1e-12 {
                find_best_lambda(&feat_cols, &y, penalty, &self.lambda_grid, 200, 1e-7)
            } else {
                (1.0, 0.0)
            };

            best_lambdas.push(best_lambda);
        }

        self.lambdas = best_lambdas;

        // Step 2: Combined penalty with per-feature lambdas
        let combined_penalty = self.build_combined_penalty(design_matrix.ncols());

        let result = pirls_logistic(
            &design_matrix,
            &y,
            &combined_penalty,
            1.0,
            200,
            1e-7,
        );

        self.coefficients = Some(result.coefficients.clone());

        let proba = self.predict_proba_raw(x_data, n_samples, n_features, feature_types, feature_names);

        let roc_auc = compute_roc_auc(y_data, &proba);
        let pr_auc = compute_pr_auc(y_data, &proba);
        let (f1, recall, precision) = compute_f1_recall_precision(y_data, &proba);
        let brier = compute_brier_score(y_data, &proba);

        let importance = self.compute_feature_importance(&design_matrix, &result.coefficients);

        GAMTrainingResult {
            roc_auc,
            pr_auc,
            f1,
            recall,
            precision,
            brier_score: brier,
            n_iterations: result.n_iterations,
            converged: result.converged,
            best_lambda: if self.lambdas.is_empty() { 0.0 } else {
                self.lambdas.iter().sum::<f64>() / self.lambdas.len() as f64
            },
            gcv_score: gcv_score(result.log_likelihood, result.edf, n),
            feature_importance: importance,
        }
    }

    /// Build design matrix — static method (không cần &mut self)
    fn build_design_matrix_static(
        x_data: &[f64],
        n_samples: usize,
        n_features: usize,
        feature_types: &[FeatureType],
        n_splines: usize,
        spline_degree: usize,
    ) -> (DMatrix<f64>, Vec<(usize, usize)>, Vec<(f64, f64)>, Vec<DMatrix<f64>>) {
        let mut col_ranges = Vec::new();
        let mut x_ranges_vec = Vec::new();
        let mut penalties_vec = Vec::new();
        let mut total_cols = 0;

        // First pass: count total columns
        let mut col_counts: Vec<usize> = Vec::with_capacity(n_features);
        for feat_idx in 0..n_features {
            let mut feature_vals = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                feature_vals.push(x_data[i * n_features + feat_idx]);
            }

            match feature_types[feat_idx] {
                FeatureType::Numeric | FeatureType::Ordinal => {
                    let basis = create_basis_matrix(&feature_vals, n_splines, spline_degree);
                    let n_basis = basis.ncols();
                    col_counts.push(n_basis);
                    total_cols += n_basis;
                }
                FeatureType::Nominal => {
                    let unique_vals: Vec<f64> = {
                        let mut sorted = feature_vals.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted.dedup();
                        sorted
                    };
                    col_counts.push(unique_vals.len());
                    total_cols += unique_vals.len();
                }
            }
        }

        // Second pass: build matrix
        let mut matrix = DMatrix::zeros(n_samples, total_cols);
        let mut col_offset = 0;

        for feat_idx in 0..n_features {
            let mut feature_vals = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                feature_vals.push(x_data[i * n_features + feat_idx]);
            }

            match feature_types[feat_idx] {
                FeatureType::Numeric | FeatureType::Ordinal => {
                    let basis = create_basis_matrix(&feature_vals, n_splines, spline_degree);
                    let n_basis = basis.ncols();
                    let x_min = feature_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let x_max = feature_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let penalty = create_penalty_matrix(n_splines, x_min, x_max, spline_degree);

                    col_ranges.push((col_offset, col_offset + n_basis));
                    x_ranges_vec.push((x_min, x_max));
                    penalties_vec.push(penalty);

                    // Copy basis vào matrix
                    for i in 0..n_samples {
                        for j in 0..n_basis {
                            matrix[(i, col_offset + j)] = basis[(i, j)];
                        }
                    }
                    col_offset += n_basis;
                }
                FeatureType::Nominal => {
                    let unique_vals: Vec<f64> = {
                        let mut sorted = feature_vals.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted.dedup();
                        sorted
                    };

                    for level_idx in 0..unique_vals.len() {
                        for i in 0..n_samples {
                            matrix[(i, col_offset)] = if (feature_vals[i] - unique_vals[level_idx]).abs() < 1e-10 {
                                1.0
                            } else {
                                0.0
                            };
                        }
                        col_ranges.push((col_offset, col_offset + 1));
                        x_ranges_vec.push((0.0, 1.0));
                        penalties_vec.push(DMatrix::zeros(1, 1));
                        col_offset += 1;
                    }
                }
            }
        }

        (matrix, col_ranges, x_ranges_vec, penalties_vec)
    }

    fn build_combined_penalty(&self, total_cols: usize) -> DMatrix<f64> {
        let mut penalty = DMatrix::zeros(total_cols, total_cols);

        for feat_idx in 0..self.feature_col_ranges.len() {
            let (start, end) = self.feature_col_ranges[feat_idx];
            let n_cols = end - start;
            let p = &self.penalties[feat_idx];
            let lambda = self.lambdas[feat_idx];

            if p.nrows() == n_cols && p.ncols() == n_cols {
                let block = p.scale(lambda);
                for i in 0..n_cols {
                    for j in 0..n_cols {
                        penalty[(start + i, start + j)] += block[(i, j)];
                    }
                }
            }
        }

        penalty
    }

    pub fn predict_proba_raw(
        &self,
        x_data: &[f64],
        n_samples: usize,
        n_features: usize,
        feature_types: &[String],
        _feature_names: &[String],
    ) -> Vec<f64> {
        let types: Vec<FeatureType> = feature_types.iter().map(|s| match s.as_str() {
            "numeric" => FeatureType::Numeric,
            "ordinal" => FeatureType::Ordinal,
            "nominal" => FeatureType::Nominal,
            _ => FeatureType::Numeric,
        }).collect();

        let (design_matrix, _, _, _) = 
            Self::build_design_matrix_static(x_data, n_samples, n_features, &types, self.n_splines, self.spline_degree);

        let beta = match &self.coefficients {
            Some(b) => b,
            None => return vec![0.5; n_samples],
        };

        let eta = design_matrix * beta;
        eta.iter().map(|&e| logistic(e + self.intercept)).collect()
    }

    pub fn predict(&self, proba: &[f64], threshold: f64) -> Vec<i32> {
        proba.iter().map(|&p| if p >= threshold { 1 } else { 0 }).collect()
    }

    fn compute_feature_importance(
        &self,
        design_matrix: &DMatrix<f64>,
        coefficients: &DVector<f64>,
    ) -> Vec<FeatureImportance> {
        let mut importance = Vec::new();

        for feat_idx in 0..self.feature_col_ranges.len() {
            let (col_start, col_end) = self.feature_col_ranges[feat_idx];
            let n_cols = col_end - col_start;
            let feat_name = if feat_idx < self.feature_names.len() {
                self.feature_names[feat_idx].clone()
            } else {
                format!("feature_{}", feat_idx)
            };

            let n = design_matrix.nrows();
            let mut contribution = DVector::zeros(n);
            for i in 0..n {
                let mut val = 0.0;
                for j in 0..n_cols {
                    val += design_matrix[(i, col_start + j)] * coefficients[col_start + j];
                }
                contribution[i] = val;
            }

            let variance = variance_of_vector(&contribution);

            importance.push(FeatureImportance {
                feature: feat_name,
                variance_importance: variance,
                term_index: feat_idx,
            });
        }

        importance.sort_by(|a, b| b.variance_importance.partial_cmp(&a.variance_importance).unwrap());
        importance
    }
}

// ========== Metric Functions ==========

fn compute_roc_auc(y_true: &[f64], y_score: &[f64]) -> f64 {
    let n = y_true.len();
    if n < 2 { return 0.5; }
    
    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter()).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let mut n_pos = 0f64;
    for &(_, l) in &pairs { if l >= 0.5 { n_pos += 1.0; } }
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    
    let mut tp = 0f64;
    let mut fp = 0f64;
    let mut auc = 0.0;
    let mut prev_fpr = 0f64;
    
    for &(_score, label) in &pairs {
        if label >= 0.5 { tp += 1.0; }
        else { fp += 1.0; }
        let tpr = tp / n_pos;
        let fpr = fp / n_neg;
        auc += (fpr - prev_fpr) * (tpr + tp / n_pos) / 2.0;
        prev_fpr = fpr;
    }
    
    auc
}

fn compute_pr_auc(y_true: &[f64], y_score: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0 { return 0.0; }

    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter())
        .map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos: f64 = y_true.iter().filter(|&&l| l >= 0.5).count() as f64;
    if n_pos == 0.0 { return 0.0; }

    let mut tp = 0f64;
    let mut fp = 0f64;
    let mut ap = 0.0;
    let mut prev_recall = 0.0;

    for &(_score, label) in &pairs {
        if label >= 0.5 { tp += 1.0; }
        else { fp += 1.0; }
        let recall = tp / n_pos;
        let precision = tp / (tp + fp);
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }

    ap
}

fn compute_f1_recall_precision(y_true: &[f64], y_proba: &[f64]) -> (f64, f64, f64) {
    let threshold = 0.5;
    let mut tp = 0f64;
    let mut fp = 0f64;
    let mut fn_ = 0f64;

    for i in 0..y_true.len() {
        let pred = if y_proba[i] >= threshold { 1 } else { 0 };
        let actual = if y_true[i] >= 0.5 { 1 } else { 0 };

        if pred == 1 && actual == 1 { tp += 1.0; }
        else if pred == 1 && actual == 0 { fp += 1.0; }
        else if pred == 0 && actual == 1 { fn_ += 1.0; }
    }

    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let f1 = if recall + precision > 0.0 {
        2.0 * recall * precision / (recall + precision)
    } else { 0.0 };

    (f1, recall, precision)
}

fn compute_brier_score(y_true: &[f64], y_proba: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let sum_sq: f64 = y_true.iter().zip(y_proba.iter())
        .map(|(y, p)| (y - p).powi(2))
        .sum();
    sum_sq / n
}

fn variance_of_vector(v: &DVector<f64>) -> f64 {
    let n = v.len() as f64;
    if n <= 1.0 { return 0.0; }
    let mean = v.sum() / n;
    let sum_sq: f64 = v.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq / (n - 1.0)
}
