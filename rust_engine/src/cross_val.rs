//! Optimized Cross-validation engine với Rayon parallelization
//!
//! Tối ưu cho large datasets:
//! - Design matrix built once, reused across folds (zero redundant computation)
//! - Parallel folds với rayon
//! - Subsampled folds khi n > threshold
//! - Improved lambda grid for better regularization

use rayon::prelude::*;
use nalgebra::DMatrix;

use crate::pirls::{pirls_logistic, logistic, gcv_score};
use crate::bspline::{create_basis_matrix, create_penalty_matrix, FeatureType};

/// Kết quả của một fold
#[derive(Debug, Clone)]
pub struct FoldResult {
    pub fold_idx: usize,
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub f1: f64,
    pub recall: f64,
    pub precision: f64,
    pub brier_score: f64,
    pub converged: bool,
    pub n_iterations: usize,
}

/// Kết quả của toàn bộ CV
#[derive(Debug, Clone)]
pub struct CVResult {
    pub folds: Vec<FoldResult>,
    pub mean_roc_auc: f64,
    pub std_roc_auc: f64,
    pub mean_pr_auc: f64,
    pub std_pr_auc: f64,
    pub mean_f1: f64,
    pub std_f1: f64,
    pub mean_recall: f64,
    pub std_recall: f64,
    pub mean_precision: f64,
    pub std_precision: f64,
    pub mean_brier: f64,
    pub std_brier: f64,
    pub converged_folds: usize,
}

/// Stratified K-Fold split
fn stratified_kfold_split(
    y: &[f64],
    n_splits: usize,
    random_seed: u64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let (pos, neg): (Vec<_>, Vec<_>) = y.iter().enumerate()
        .partition(|(_, &label)| label >= 0.5);
    let mut pos: Vec<_> = pos.into_iter().map(|(i, _)| i).collect();
    let mut neg: Vec<_> = neg.into_iter().map(|(i, _)| i).collect();

    let mut rng = random_seed.max(1);
    shuffle_usize(&mut pos, &mut rng);
    shuffle_usize(&mut neg, &mut rng);

    let pos_folds = split_indices(&pos, n_splits);
    let neg_folds = split_indices(&neg, n_splits);

    (0..n_splits).map(|i| {
        let val: Vec<_> = pos_folds.get(i).cloned().unwrap_or_default()
            .into_iter()
            .chain(neg_folds.get(i).cloned().unwrap_or_default())
            .collect();
        let train: Vec<_> = (0..n_splits)
            .filter(|&j| j != i)
            .flat_map(|j| {
                let mut idx = pos_folds.get(j).cloned().unwrap_or_default();
                idx.extend(neg_folds.get(j).cloned().unwrap_or_default());
                idx
            })
            .collect();
        (train, val)
    }).collect()
}

fn split_indices(indices: &[usize], n_splits: usize) -> Vec<Vec<usize>> {
    let n = indices.len();
    let base = n / n_splits;
    let rem = n % n_splits;
    let mut folds = Vec::with_capacity(n_splits);
    let mut start = 0;
    for i in 0..n_splits {
        let size = base + if i < rem { 1 } else { 0 };
        folds.push(indices[start..start + size].to_vec());
        start += size;
    }
    folds
}

fn shuffle_usize(v: &mut [usize], rng: &mut u64) {
    let n = v.len();
    for i in (1..n).rev() {
        *rng = lcg_next(*rng);
        let j = (*rng % i as u64) as usize;
        v.swap(i, j);
    }
}

fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

/// Pre-built design matrix info — cached once, reused across folds
#[derive(Clone)]
pub struct CachedDesignMatrix {
    /// Full design matrix (all samples, all features)
    pub matrix: DMatrix<f64>,
    /// Column ranges per feature: (start_col, end_col)
    pub col_ranges: Vec<(usize, usize)>,
    /// Penalty matrices per feature
    pub penalties: Vec<DMatrix<f64>>,
    /// Feature types
    pub feature_types: Vec<FeatureType>,
}

/// Build design matrix once from raw data — single-pass optimized version
/// Processes each feature completely in one outer loop, avoiding redundant data extraction.
pub fn build_cached_design_matrix(
    x_data: &[f64],
    n_samples: usize,
    n_features: usize,
    feature_types: &[String],
    n_splines: usize,
    spline_degree: usize,
) -> CachedDesignMatrix {
    let types: Vec<FeatureType> = feature_types.iter().map(|s| match s.as_str() {
        "numeric" => FeatureType::Numeric,
        "ordinal" => FeatureType::Ordinal,
        "nominal" => FeatureType::Nominal,
        _ => FeatureType::Numeric,
    }).collect();

    // First pass: count total columns — compute basis matrix to get exact column count
    let mut total_cols = 0;
    let mut col_counts: Vec<usize> = Vec::with_capacity(n_features);
    for feat_idx in 0..n_features {
        match types[feat_idx] {
            FeatureType::Numeric | FeatureType::Ordinal => {
                let mut feature_vals = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    feature_vals.push(x_data[i * n_features + feat_idx]);
                }
                let basis = create_basis_matrix(&feature_vals, n_splines, spline_degree);
                col_counts.push(basis.ncols());
                total_cols += basis.ncols();
            }
            FeatureType::Nominal => {
                let mut feature_vals = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    feature_vals.push(x_data[i * n_features + feat_idx]);
                }
                feature_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                feature_vals.dedup();
                col_counts.push(feature_vals.len());
                total_cols += feature_vals.len();
            }
        }
    }

    // Second pass: build matrix — one feature at a time, fully processed
    let mut matrix = DMatrix::zeros(n_samples, total_cols);
    let mut col_ranges = Vec::with_capacity(n_features);
    let mut penalties_vec = Vec::with_capacity(n_features);
    let mut col_offset = 0;

    for feat_idx in 0..n_features {
        let mut feature_vals = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature_vals.push(x_data[i * n_features + feat_idx]);
        }

        match types[feat_idx] {
            FeatureType::Numeric | FeatureType::Ordinal => {
                let basis = create_basis_matrix(&feature_vals, n_splines, spline_degree);
                let n_basis = basis.ncols();
                let x_min = feature_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let x_max = feature_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let penalty = create_penalty_matrix(n_splines, x_min, x_max, spline_degree);

                col_ranges.push((col_offset, col_offset + n_basis));
                penalties_vec.push(penalty);

                for i in 0..n_samples {
                    for j in 0..n_basis {
                        matrix[(i, col_offset + j)] = basis[(i, j)];
                    }
                }
                col_offset += n_basis;
            }
            FeatureType::Nominal => {
                let mut sorted = feature_vals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted.dedup();

                for _level_idx in 0..sorted.len() {
                    for i in 0..n_samples {
                        matrix[(i, col_offset)] = if (feature_vals[i] - sorted[_level_idx]).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        };
                    }
                    col_ranges.push((col_offset, col_offset + 1));
                    penalties_vec.push(DMatrix::zeros(1, 1));
                    col_offset += 1;
                }
            }
        }
    }

    CachedDesignMatrix {
        matrix,
        col_ranges,
        penalties: penalties_vec,
        feature_types: types,
    }
}

/// Submatrix extractor — returns rows by index
/// Uses contiguous buffer allocation for better cache performance.
fn extract_rows(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    let n = indices.len();
    let p = matrix.ncols();
    if n == 0 || p == 0 {
        return DMatrix::zeros(0, 0);
    }
    let mut out = vec![0.0f64; n * p];
    for (row_pos, &i) in indices.iter().enumerate() {
        let ro_out = row_pos * p;
        for j in 0..p {
            out[ro_out + j] = matrix[(i, j)];
        }
    }
    DMatrix::from_vec(n, p, out)
}

/// Optimized: chạy stratified K-fold CV với cached design matrix
pub fn run_cross_validation_cached(
    cached: &CachedDesignMatrix,
    y_data: &[f64],
    n_splits: usize,
    random_seed: u64,
) -> CVResult {
    let splits = stratified_kfold_split(y_data, n_splits, random_seed);
    let subsample_threshold = 50_000;

    let fold_results: Vec<FoldResult> = splits
        .into_par_iter()
        .enumerate()
        .map(|(fold_idx, (train_idx, val_idx))| {
            run_single_fold_cached(
                fold_idx,
                cached,
                y_data,
                &train_idx,
                &val_idx,
                subsample_threshold,
                random_seed + fold_idx as u64,
            )
        })
        .collect();

    compute_cv_summary(&fold_results)
}

/// Chạy một fold với cached design matrix
fn run_single_fold_cached(
    fold_idx: usize,
    cached: &CachedDesignMatrix,
    y_data: &[f64],
    train_idx: &[usize],
    val_idx: &[usize],
    subsample_threshold: usize,
    seed: u64,
) -> FoldResult {
    let mut train_indices = train_idx.to_vec();

    // Subsample nếu train set quá lớn
    if train_indices.len() > subsample_threshold {
        let mut rng = seed.max(1);
        shuffle_usize(&mut train_indices, &mut rng);
        train_indices.truncate(subsample_threshold);
        train_indices.sort_unstable();
    }

    let val_indices = val_idx;

    // Extract sub-matrices from cached design matrix (no rebuilding!)
    let train_x = extract_rows(&cached.matrix, &train_indices);
    let train_y: Vec<f64> = train_indices.iter().map(|&i| y_data[i]).collect();

    // Fit GAM using per-feature lambda optimization
    let result = fit_gam_cached(&train_x, &train_y, cached);

    // Predict on validation set
    let val_x = extract_rows(&cached.matrix, val_indices);
    let val_y: Vec<f64> = val_indices.iter().map(|&i| y_data[i]).collect();
    let val_proba = predict_gam_matrix(&val_x, &result.coefficients, &cached.col_ranges);

    let roc_auc = compute_roc_auc(&val_y, &val_proba);
    let pr_auc = compute_pr_auc(&val_y, &val_proba);
    let (f1, recall, precision) = compute_f1_recall_precision(&val_y, &val_proba);
    let brier = compute_brier_score(&val_y, &val_proba);

    FoldResult {
        fold_idx,
        roc_auc,
        pr_auc,
        f1,
        recall,
        precision,
        brier_score: brier,
        converged: result.converged,
        n_iterations: result.n_iterations,
    }
}

struct GAMFitResult {
    coefficients: Vec<f64>,
    converged: bool,
    n_iterations: usize,
}

fn fit_gam_cached(
    x: &DMatrix<f64>,
    y: &[f64],
    cached: &CachedDesignMatrix,
) -> GAMFitResult {
    let p = x.ncols();
    let n_features = cached.col_ranges.len();
    let y_vec = nalgebra::DVector::from_vec(y.to_vec());

    // Use the full-matrix P-IRLS solver which is numerically stable
    // Build combined penalty from all features
    let mut penalty = DMatrix::zeros(p, p);
    let lambda = 1.0; // Moderate regularization
    for feat_idx in 0..n_features {
        let (col_start, col_end) = cached.col_ranges[feat_idx];
        let nc = col_end - col_start;
        let block_penalty = &cached.penalties[feat_idx];
        if !block_penalty.is_empty() && block_penalty.trace().abs() > 1e-12 {
            for i in 0..nc {
                for j in 0..nc {
                    penalty[(col_start + i, col_start + j)] += block_penalty[(i, j)];
                }
            }
        }
    }

    // Run full P-IRLS with combined penalty
    let result = pirls_logistic(x, &y_vec, &penalty, lambda, 200, 1e-7);

    GAMFitResult {
        coefficients: result.coefficients.iter().copied().collect(),
        converged: result.converged,
        n_iterations: result.n_iterations,
    }
}

/// Predict using design matrix directly
fn predict_gam_matrix(
    x: &DMatrix<f64>,
    coefficients: &[f64],
    _col_ranges: &[(usize, usize)],
) -> Vec<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let effective_p = p.min(coefficients.len());

    let mut proba = Vec::with_capacity(n);
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..effective_p {
            eta += x[(i, j)] * coefficients[j];
        }
        proba.push(logistic(eta));
    }
    proba
}

// ========== Legacy: non-cached CV (for backward compatibility) ==========

pub fn run_cross_validation(
    x_data: &[f64],
    y_data: &[f64],
    _n_samples: usize,
    n_features: usize,
    feature_types: &[String],
    feature_names: &[String],
    n_splits: usize,
    n_splines: usize,
    random_seed: u64,
) -> CVResult {
    let splits = stratified_kfold_split(y_data, n_splits, random_seed);
    let subsample_threshold = 50_000;

    let fold_results: Vec<FoldResult> = splits
        .into_par_iter()
        .enumerate()
        .map(|(fold_idx, (train_idx, val_idx))| {
            run_single_fold_optimized(
                fold_idx,
                x_data, y_data,
                &train_idx, &val_idx,
                n_features,
                feature_types,
                feature_names,
                n_splines,
                subsample_threshold,
                random_seed + fold_idx as u64,
            )
        })
        .collect();

    compute_cv_summary(&fold_results)
}

fn run_single_fold_optimized(
    fold_idx: usize,
    x_data: &[f64],
    y_data: &[f64],
    train_idx: &[usize],
    val_idx: &[usize],
    n_features: usize,
    feature_types: &[String],
    _feature_names: &[String],
    n_splines: usize,
    subsample_threshold: usize,
    seed: u64,
) -> FoldResult {
    let mut train_indices = train_idx.to_vec();
    if train_indices.len() > subsample_threshold {
        let mut rng = seed.max(1);
        shuffle_usize(&mut train_indices, &mut rng);
        train_indices.truncate(subsample_threshold);
        train_indices.sort_unstable();
    }

    let val_indices = val_idx;
    let train_x = extract_features(x_data, &train_indices, n_features);
    let train_y: Vec<f64> = train_indices.iter().map(|&i| y_data[i]).collect();

    let rust_ft: Vec<String> = feature_types.iter()
        .map(|s| match s.as_str() {
            "numeric" => "numeric".to_string(),
            "ordinal" => "ordinal".to_string(),
            _ => "numeric".to_string(),
        }).collect();

    let result = fit_gam_rust(&train_x, &train_y, n_features, n_splines, &rust_ft);

    let val_x = extract_features(x_data, val_indices, n_features);
    let val_y: Vec<f64> = val_indices.iter().map(|&i| y_data[i]).collect();
    let val_proba = predict_gam_rust(&val_x, &result.coefficients, n_features, n_splines, &rust_ft);

    let roc_auc = compute_roc_auc(&val_y, &val_proba);
    let pr_auc = compute_pr_auc(&val_y, &val_proba);
    let (f1, recall, precision) = compute_f1_recall_precision(&val_y, &val_proba);
    let brier = compute_brier_score(&val_y, &val_proba);

    FoldResult {
        fold_idx, roc_auc, pr_auc, f1, recall, precision,
        brier_score: brier, converged: result.converged, n_iterations: result.n_iterations,
    }
}

fn fit_gam_rust(
    x: &[f64], y: &[f64], n_features: usize, n_splines: usize, feature_types: &[String],
) -> GAMFitResult {
    let n = x.len() / n_features;
    let (design, col_ranges) = build_design_matrix(x, n, n_features, feature_types, n_splines, 3);
    let d_n = design.nrows();

    let penalty = build_combined_penalty(&col_ranges, n_splines, 3, n_features);
    let y_vec = nalgebra::DVector::from_vec(y.to_vec());

    let lambda_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
    let mut best_lambda = 1.0;
    let mut best_gcv = f64::MAX;

    for &lambda in &lambda_values {
        let result = pirls_logistic(&design, &y_vec, &penalty, lambda, 30, 1e-5);
        let gcv = gcv_score(result.log_likelihood, result.edf, d_n);
        if gcv < best_gcv { best_gcv = gcv; best_lambda = lambda; }
    }

    let result = pirls_logistic(&design, &y_vec, &penalty, best_lambda, 50, 1e-6);
    GAMFitResult {
        coefficients: result.coefficients.iter().copied().collect(),
        converged: result.converged, n_iterations: result.n_iterations,
    }
}

fn predict_gam_rust(
    x: &[f64], coefficients: &[f64], n_features: usize, n_splines: usize, feature_types: &[String],
) -> Vec<f64> {
    let n = x.len() / n_features;
    let (design, _) = build_design_matrix(x, n, n_features, feature_types, n_splines, 3);
    let n_p = design.ncols();
    let beta = nalgebra::DVector::from_vec(coefficients.to_vec());
    let p = n_p.min(beta.len());
    let eta = if p == n_p {
        design * &beta
    } else {
        let dt = design.columns(0, p);
        let bt = beta.rows(0, p);
        dt * &bt
    };
    eta.iter().map(|&e| logistic(e)).collect()
}

fn build_design_matrix(
    x: &[f64], n: usize, n_features: usize, feature_types: &[String],
    n_splines: usize, degree: usize,
) -> (nalgebra::DMatrix<f64>, Vec<(usize, usize)>) {
    let mut col_ranges = Vec::new();
    let mut total_cols = 0;
    let mut col_counts = Vec::with_capacity(n_features);

    for feat_idx in 0..n_features {
        let mut fvals = Vec::with_capacity(n);
        for i in 0..n { fvals.push(x[i * n_features + feat_idx]); }
        if feature_types[feat_idx] == "nominal" {
            let mut sorted = fvals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();
            col_counts.push(sorted.len());
            total_cols += sorted.len();
        } else {
            let basis = create_basis_matrix(&fvals, n_splines, degree);
            col_counts.push(basis.ncols());
            total_cols += basis.ncols();
        }
    }

    let mut matrix = nalgebra::DMatrix::zeros(n, total_cols);
    let mut col_offset = 0;
    for feat_idx in 0..n_features {
        let mut fvals = Vec::with_capacity(n);
        for i in 0..n { fvals.push(x[i * n_features + feat_idx]); }
        if feature_types[feat_idx] == "nominal" {
            let mut sorted = fvals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();
            for lv in sorted {
                for i in 0..n {
                    matrix[(i, col_offset)] = if (fvals[i] - lv).abs() < 1e-10 { 1.0 } else { 0.0 };
                }
                col_ranges.push((col_offset, col_offset + 1));
                col_offset += 1;
            }
        } else {
            let basis = create_basis_matrix(&fvals, n_splines, degree);
            let nb = basis.ncols();
            for i in 0..n {
                for j in 0..nb { matrix[(i, col_offset + j)] = basis[(i, j)]; }
            }
            col_ranges.push((col_offset, col_offset + nb));
            col_offset += nb;
        }
    }
    (matrix, col_ranges)
}

fn build_combined_penalty(
    col_ranges: &[(usize, usize)], _n_splines: usize, _degree: usize, _n_features: usize,
) -> nalgebra::DMatrix<f64> {
    let total = col_ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let mut penalty = nalgebra::DMatrix::zeros(total, total);
    for (_, (s, e)) in col_ranges.iter().enumerate() {
        let nc = e - s;
        let mut p: nalgebra::DMatrix<f64> = nalgebra::DMatrix::zeros(nc, nc);
        for i in 0..nc.saturating_sub(2) {
            p[(i, i)] += 1.0; p[(i, i+1)] -= 2.0; p[(i, i+2)] += 1.0;
            p[(i+1, i)] -= 2.0; p[(i+1, i+1)] += 4.0; p[(i+1, i+2)] -= 2.0;
            p[(i+2, i)] += 1.0; p[(i+2, i+1)] -= 2.0; p[(i+2, i+2)] += 1.0;
        }
        for i in 0..nc { for j in 0..nc { penalty[(s + i, s + j)] += p[(i, j)]; } }
    }
    penalty
}

fn extract_features(x: &[f64], indices: &[usize], n_features: usize) -> Vec<f64> {
    let n = indices.len();
    let mut out = Vec::with_capacity(n * n_features);
    for &i in indices {
        let ro = i * n_features;
        for j in 0..n_features { out.push(x[ro + j]); }
    }
    out
}

fn compute_cv_summary(folds: &[FoldResult]) -> CVResult {
    let n = folds.len() as f64;
    if n == 0.0 {
        return CVResult {
            folds: Vec::new(),
            mean_roc_auc: 0.0, std_roc_auc: 0.0,
            mean_pr_auc: 0.0, std_pr_auc: 0.0,
            mean_f1: 0.0, std_f1: 0.0,
            mean_recall: 0.0, std_recall: 0.0,
            mean_precision: 0.0, std_precision: 0.0,
            mean_brier: 0.0, std_brier: 0.0,
            converged_folds: 0,
        };
    }

    let mean = |vals: &[f64]| vals.iter().sum::<f64>() / n;
    let std = |vals: &[f64], m: f64| {
        (vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
    };

    let roc_aucs: Vec<f64> = folds.iter().map(|f| f.roc_auc).collect();
    let pr_aucs: Vec<f64> = folds.iter().map(|f| f.pr_auc).collect();
    let f1s: Vec<f64> = folds.iter().map(|f| f.f1).collect();
    let recalls: Vec<f64> = folds.iter().map(|f| f.recall).collect();
    let precisions: Vec<f64> = folds.iter().map(|f| f.precision).collect();
    let briers: Vec<f64> = folds.iter().map(|f| f.brier_score).collect();

    let m_roc = mean(&roc_aucs);
    let m_pr = mean(&pr_aucs);
    let m_f1 = mean(&f1s);
    let m_rec = mean(&recalls);
    let m_prec = mean(&precisions);
    let m_brier = mean(&briers);

    CVResult {
        folds: folds.to_vec(),
        mean_roc_auc: m_roc, std_roc_auc: std(&roc_aucs, m_roc),
        mean_pr_auc: m_pr, std_pr_auc: std(&pr_aucs, m_pr),
        mean_f1: m_f1, std_f1: std(&f1s, m_f1),
        mean_recall: m_rec, std_recall: std(&recalls, m_rec),
        mean_precision: m_prec, std_precision: std(&precisions, m_prec),
        mean_brier: m_brier, std_brier: std(&briers, m_brier),
        converged_folds: folds.len(),
    }
}

// ========== Metric functions ==========

fn compute_roc_auc(y_true: &[f64], y_score: &[f64]) -> f64 {
    let n = y_true.len();
    if n < 2 { return 0.5; }
    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter()).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut n_pos = 0f64;
    for &(_, l) in &pairs { if l >= 0.5 { n_pos += 1.0; } }
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    let (mut tp, mut fp, mut auc, mut prev_fpr) = (0f64, 0f64, 0.0, 0f64);
    for &(_s, label) in &pairs {
        if label >= 0.5 { tp += 1.0; } else { fp += 1.0; }
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
    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter()).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos: f64 = y_true.iter().filter(|&&l| l >= 0.5).count() as f64;
    if n_pos == 0.0 { return 0.0; }
    let (mut tp, mut fp, mut ap, mut prev_recall) = (0f64, 0f64, 0.0, 0.0);
    for &(_s, label) in &pairs {
        if label >= 0.5 { tp += 1.0; } else { fp += 1.0; }
        let rec = tp / n_pos;
        let prec = tp / (tp + fp);
        ap += (rec - prev_recall) * prec;
        prev_recall = rec;
    }
    ap
}

fn compute_f1_recall_precision(y_true: &[f64], y_proba: &[f64]) -> (f64, f64, f64) {
    let (mut tp, mut fp, mut fn_) = (0f64, 0f64, 0f64);
    for i in 0..y_true.len() {
        let pred = if y_proba[i] >= 0.5 { 1 } else { 0 };
        let actual = if y_true[i] >= 0.5 { 1 } else { 0 };
        match (pred, actual) {
            (1, 1) => tp += 1.0,
            (1, 0) => fp += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let f1 = if recall + precision > 0.0 { 2.0 * recall * precision / (recall + precision) } else { 0.0 };
    (f1, recall, precision)
}

fn compute_brier_score(y_true: &[f64], y_proba: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    y_true.iter().zip(y_proba.iter()).map(|(y, p)| (y - p).powi(2)).sum::<f64>() / n
}
