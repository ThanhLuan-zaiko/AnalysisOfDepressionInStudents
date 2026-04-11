//! GAM Classifier — High-level interface

use nalgebra::{DVector, DMatrix};
use std::collections::HashMap;

use crate::bspline::{create_basis_matrix, create_penalty_matrix, FeatureType};
use crate::pirls::{pirls_logistic_block, gcv_score, logistic, FeatureBlock};

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
        // Reasonable lambda grid — includes small values to allow learning
        let lambda_grid: Vec<f64> = vec![
            0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0,
        ];

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

        let (design_matrix, col_ranges, x_ranges_vec, penalties_vec, expanded_names) =
            Self::build_design_matrix_static(x_data, n_samples, n_features, &types, feature_names, self.n_splines, self.spline_degree);

        self.feature_col_ranges = col_ranges;
        self.x_ranges = x_ranges_vec;
        self.penalties = penalties_vec;
        self.feature_names = expanded_names;

        let n = design_matrix.nrows();
        let p = design_matrix.ncols();

        // Lambda grid: reasonable range for logistic GAM
        let lambda_candidates = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0];

        // Convert design matrix to row-major for block solver
        let mut design_data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                design_data.push(design_matrix[(i, j)]);
            }
        }

        // Build FeatureBlocks
        let build_blocks = |lambda: f64| -> Vec<FeatureBlock> {
            self.feature_col_ranges.iter().enumerate().map(|(idx, &(cs, ce))| {
                FeatureBlock {
                    col_start: cs,
                    n_cols: ce - cs,
                    penalty: self.penalties[idx].clone(),
                    lambda,
                }
            }).collect()
        };

        // FAST grid search: subsample to 5000 samples
        let grid_n = n.min(5000);
        let step = n / grid_n;
        let mut grid_x_data = Vec::with_capacity(grid_n * p);
        let mut grid_y = Vec::with_capacity(grid_n);
        for i in (0..n).step_by(step) {
            let ro = i * p;
            for j in 0..p {
                grid_x_data.push(design_data[ro + j]);
            }
            grid_y.push(y_data[i]);
            if grid_y.len() >= grid_n { break; }
        }
        let grid_search_n = grid_y.len();

        // Grid search — best GCV, don't require convergence (block solver may not converge formally)
        let mut best_lambda = 10.0;
        let mut best_gcv = f64::MAX;

        for &lambda in &lambda_candidates {
            let blocks = build_blocks(lambda);
            let result = pirls_logistic_block(&grid_x_data, &grid_y, grid_search_n, p, &blocks, 30, 1e-4);
            let gcv = gcv_score(result.log_likelihood, result.edf, grid_search_n);

            if gcv < best_gcv {
                best_gcv = gcv;
                best_lambda = lambda;
            }
        }

        // Store lambdas for reporting
        self.lambdas = vec![best_lambda; n_features];

        // Final fit with best lambda on FULL data
        let final_blocks = build_blocks(best_lambda);
        let result = pirls_logistic_block(&design_data, y_data, n, p, &final_blocks, 60, 1e-4);

        self.coefficients = Some(result.coefficients.clone());

        // Predict using original x_data for compatibility
        let proba = self.predict_proba_raw(x_data, n_samples, n_features, feature_types, feature_names);

        let roc_auc = compute_roc_auc(y_data, &proba);
        let pr_auc = compute_pr_auc(y_data, &proba);
        let (f1, recall, precision) = compute_f1_recall_precision(y_data, &proba);
        let brier = compute_brier_score(y_data, &proba);

        // Feature importance computation
        let dm = DMatrix::from_row_slice(n, p, &design_data);
        let importance = self.compute_feature_importance(&dm, &result.coefficients);

        // Debug logging
        let coef_norm = result.coefficients.norm();
        let coef_min = result.coefficients.min();
        let coef_max = result.coefficients.max();
        let proba_min = proba.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let proba_max = proba.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        eprintln!("[GAM DEBUG] best_lambda={}, coef_norm={:.2}, coef_range=[{:.3}, {:.3}]", 
                  best_lambda, coef_norm, coef_min, coef_max);
        eprintln!("[GAM DEBUG] proba range=[{:.4}, {:.4}], converged={}, iterations={}",
                  proba_min, proba_max, result.converged, result.n_iterations);

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
        feature_names: &[String],
        n_splines: usize,
        spline_degree: usize,
    ) -> (DMatrix<f64>, Vec<(usize, usize)>, Vec<(f64, f64)>, Vec<DMatrix<f64>>, Vec<String>) {
        let mut col_ranges = Vec::new();
        let mut x_ranges_vec = Vec::new();
        let mut penalties_vec = Vec::new();
        let mut expanded_names = Vec::new();
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

                    // Store expanded name for this feature
                    let orig_name = if feat_idx < feature_names.len() {
                        feature_names[feat_idx].clone()
                    } else {
                        format!("feature_{}", feat_idx)
                    };
                    expanded_names.push(orig_name.clone());

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

                    let orig_name = if feat_idx < feature_names.len() {
                        feature_names[feat_idx].clone()
                    } else {
                        format!("feature_{}", feat_idx)
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
                        // Store expanded name for each level
                        expanded_names.push(format!("{} [level {}]", orig_name, level_idx));
                        col_offset += 1;
                    }
                }
            }
        }

        (matrix, col_ranges, x_ranges_vec, penalties_vec, expanded_names)
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

        let (design_matrix, _, _, _, _) =
            Self::build_design_matrix_static(x_data, n_samples, n_features, &types, _feature_names, self.n_splines, self.spline_degree);

        let beta = match &self.coefficients {
            Some(b) => b,
            None => return vec![0.5; n_samples],
        };

        // Dimension safety: handle mismatch between training and prediction features
        let n_p = design_matrix.ncols();
        let p = n_p.min(beta.len());
        let eta = if p == n_p {
            design_matrix * beta
        } else {
            let dt = design_matrix.columns(0, p);
            let bt = beta.rows(0, p);
            &dt * &bt
        };
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
        let n = design_matrix.nrows();
        let n_ranges = self.feature_col_ranges.len();
        let n_names = self.feature_names.len();

        let mut importance = Vec::new();

        for range_idx in 0..n_ranges {
            let (col_start, col_end) = self.feature_col_ranges[range_idx];
            let n_cols = col_end - col_start;

            // Map range index to feature name
            let feat_name = if n_ranges == n_names {
                self.feature_names[range_idx].clone()
            } else if range_idx < n_names {
                self.feature_names[range_idx].clone()
            } else {
                let mut orig_idx = n_names - 1;
                for oi in (0..n_names).rev() {
                    if oi <= range_idx {
                        orig_idx = oi;
                        break;
                    }
                }
                format!("{} (level)", self.feature_names[orig_idx.min(n_names - 1)])
            };

            let mut contribution = DVector::zeros(n);
            for i in 0..n {
                let mut val = 0.0;
                for j in 0..n_cols {
                    if col_start + j < coefficients.len() {
                        val += design_matrix[(i, col_start + j)] * coefficients[col_start + j];
                    }
                }
                contribution[i] = val;
            }

            let variance = variance_of_vector(&contribution);
            importance.push(FeatureImportance {
                feature: feat_name,
                variance_importance: variance,
                term_index: range_idx,
            });
        }

        importance.sort_by(|a, b| b.variance_importance.partial_cmp(&a.variance_importance)
            .unwrap_or(std::cmp::Ordering::Equal));
        importance
    }
}

// ========== Metric Functions ==========

fn compute_roc_auc(y_true: &[f64], y_score: &[f64]) -> f64 {
    let n = y_true.len();
    if n < 2 { return 0.5; }
    
    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter())
        .map(|(&s, &l)| (s, l))
        .filter(|&(s, _)| !s.is_nan())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut n_pos = 0f64;
    for &(_, l) in &pairs { if l >= 0.5 { n_pos += 1.0; } }
    let n_neg = pairs.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    
    // FIX: Correct trapezoidal rule for AUC
    let mut tp = 0f64;
    let mut fp = 0f64;
    let mut auc = 0.0;
    let mut prev_tpr = 0f64;
    let mut prev_fpr = 0f64;
    
    for &(_score, label) in &pairs {
        if label >= 0.5 { tp += 1.0; }
        else { fp += 1.0; }
        let tpr = tp / n_pos;
        let fpr = fp / n_neg;
        // Correct trapezoidal: average of prev_tpr and current tpr
        auc += (fpr - prev_fpr) * (prev_tpr + tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    
    auc
}

fn compute_pr_auc(y_true: &[f64], y_score: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0 { return 0.0; }

    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter())
        .map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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
