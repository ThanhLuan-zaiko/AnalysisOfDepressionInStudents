//! Cross-validation engine với Rayon parallelization

use rayon::prelude::*;

use crate::gam::GAMClassifier;

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
    let mut pos_indices: Vec<usize> = y.iter()
        .enumerate()
        .filter(|(_, &label)| label >= 0.5)
        .map(|(i, _)| i)
        .collect();

    let mut neg_indices: Vec<usize> = y.iter()
        .enumerate()
        .filter(|(_, &label)| label < 0.5)
        .map(|(i, _)| i)
        .collect();

    let mut rng = SimpleRng::new(random_seed);
    shuffle(&mut pos_indices, &mut rng);
    shuffle(&mut neg_indices, &mut rng);

    let pos_splits = split_into_folds(&pos_indices, n_splits);
    let neg_splits = split_into_folds(&neg_indices, n_splits);

    let mut folds = Vec::with_capacity(n_splits);
    for i in 0..n_splits {
        let val_indices: Vec<usize> = pos_splits.get(i)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .chain(neg_splits.get(i).cloned().unwrap_or_default())
            .collect();

        let train_indices: Vec<usize> = (0..n_splits)
            .filter(|&j| j != i)
            .flat_map(|j| {
                let mut idx = pos_splits.get(j).cloned().unwrap_or_default();
                idx.extend(neg_splits.get(j).cloned().unwrap_or_default());
                idx
            })
            .collect();

        folds.push((train_indices, val_indices));
    }

    folds
}

fn split_into_folds(indices: &[usize], n_splits: usize) -> Vec<Vec<usize>> {
    let n = indices.len();
    let fold_size = n / n_splits;
    let remainder = n % n_splits;
    let mut folds = Vec::with_capacity(n_splits);
    let mut start = 0;

    for i in 0..n_splits {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        folds.push(indices[start..end.min(n)].to_vec());
        start = end;
    }

    folds
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut SimpleRng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = (rng.next() % i as u64) as usize;
        v.swap(i, j);
    }
}

/// Chạy stratified K-fold CV — PARALLEL với Rayon
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

    let fold_results: Vec<FoldResult> = splits
        .par_iter()
        .enumerate()
        .map(|(fold_idx, (train_idx, val_idx))| {
            run_single_fold(
                fold_idx,
                x_data, y_data,
                train_idx, val_idx,
                n_features,
                feature_types,
                feature_names,
                n_splines,
            )
        })
        .collect();

    compute_cv_summary(&fold_results)
}

fn run_single_fold(
    fold_idx: usize,
    x_data: &[f64],
    y_data: &[f64],
    train_idx: &[usize],
    val_idx: &[usize],
    n_features: usize,
    feature_types: &[String],
    feature_names: &[String],
    n_splines: usize,
) -> FoldResult {
    let train_x: Vec<f64> = train_idx.iter()
        .flat_map(|&i| (0..n_features).map(move |j| x_data[i * n_features + j]))
        .collect();
    let train_y: Vec<f64> = train_idx.iter().map(|&i| y_data[i]).collect();

    let val_x: Vec<f64> = val_idx.iter()
        .flat_map(|&i| (0..n_features).map(move |j| x_data[i * n_features + j]))
        .collect();
    let val_y: Vec<f64> = val_idx.iter().map(|&i| y_data[i]).collect();

    let mut gam = GAMClassifier::new(
        vec![],
        feature_names.to_vec(),
        n_splines,
    );

    let _result = gam.fit(
        &train_x,
        &train_y,
        train_idx.len(),
        n_features,
        feature_types,
        feature_names,
    );

    let val_proba = gam.predict_proba_raw(
        &val_x,
        val_idx.len(),
        n_features,
        feature_types,
        feature_names,
    );

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
    }
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

    let mean = |vals: &[f64]| -> f64 { vals.iter().sum::<f64>() / n };
    let std = |vals: &[f64], m: f64| -> f64 {
        let variance: f64 = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
        variance.sqrt()
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
        mean_roc_auc: m_roc,
        std_roc_auc: std(&roc_aucs, m_roc),
        mean_pr_auc: m_pr,
        std_pr_auc: std(&pr_aucs, m_pr),
        mean_f1: m_f1,
        std_f1: std(&f1s, m_f1),
        mean_recall: m_rec,
        std_recall: std(&recalls, m_rec),
        mean_precision: m_prec,
        std_precision: std(&precisions, m_prec),
        mean_brier: m_brier,
        std_brier: std(&briers, m_brier),
        converged_folds: folds.len(),
    }
}

// Metric functions — optimized O(n log n)
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
    let mut pairs: Vec<(f64, f64)> = y_score.iter().zip(y_true.iter()).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos: f64 = y_true.iter().filter(|&&l| l >= 0.5).count() as f64;
    if n_pos == 0.0 { return 0.0; }
    let mut tp = 0f64;
    let mut fp = 0f64;
    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for &(_score, label) in &pairs {
        if label >= 0.5 { tp += 1.0; } else { fp += 1.0; }
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
    let f1 = if recall + precision > 0.0 { 2.0 * recall * precision / (recall + precision) } else { 0.0 };
    (f1, recall, precision)
}

fn compute_brier_score(y_true: &[f64], y_proba: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let sum_sq: f64 = y_true.iter().zip(y_proba.iter()).map(|(y, p)| (y - p).powi(2)).sum();
    sum_sq / n
}
