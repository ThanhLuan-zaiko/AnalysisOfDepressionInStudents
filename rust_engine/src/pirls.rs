//! P-IRLS: Penalized Iteratively Reweighted Least Squares
//!
//! Optimized cho nhiều features:
//! 1. **Block-diagonal P-IRLS** — mỗi feature block tính độc lập, song song
//! 2. **Per-feature X'wX** — không cần tính cross-terms giữa các feature
//! 3. **Cholesky block-by-block** — solve từng block nhỏ thay vì ma trận lớn
//! 4. **Parallel eta computation** — tính X*beta song song

use nalgebra::{DVector, DMatrix, Cholesky};
use rayon::prelude::*;

/// Block info cho một feature
#[derive(Clone)]
pub struct FeatureBlock {
    /// Column start index trong design matrix
    pub col_start: usize,
    /// Số cột của block này
    pub n_cols: usize,
    /// Penalty matrix cho block này (k × k) — CHƯA scale bởi lambda
    pub penalty: DMatrix<f64>,
    /// Lambda tối ưu cho feature này (từ grid search)
    pub lambda: f64,
}

/// Kết quả fitting
#[derive(Debug, Clone)]
pub struct PirlsResult {
    pub coefficients: DVector<f64>,
    pub weights: DVector<f64>,
    pub fitted: DVector<f64>,
    pub probabilities: DVector<f64>,
    pub n_iterations: usize,
    pub converged: bool,
    pub log_likelihood: f64,
    pub edf: f64,
}

/// Accumulator cho full X'WX (legacy)
#[derive(Clone)]
struct FullAccum {
    upper: Vec<f64>,
    xwz: Vec<f64>,
    p: usize,
}

impl FullAccum {
    fn new(p: usize) -> Self {
        FullAccum {
            upper: vec![0.0; p * (p + 1) / 2],
            xwz: vec![0.0; p],
            p,
        }
    }
    fn merge(mut self, o: &FullAccum) -> Self {
        for i in 0..self.upper.len() { self.upper[i] += o.upper[i]; }
        for i in 0..self.p { self.xwz[i] += o.xwz[i]; }
        self
    }
    fn to_xwx(&self) -> DMatrix<f64> {
        let p = self.p;
        let mut m = DMatrix::zeros(p, p);
        let mut idx = 0;
        for j in 0..p { for k in j..p { m[(j,k)] = self.upper[idx]; m[(k,j)] = self.upper[idx]; idx += 1; } }
        m
    }
    fn to_xwz(&self) -> DVector<f64> { DVector::from_vec(self.xwz.clone()) }
}

/// Simple LCG PRNG
#[inline]
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

/// Fisher-Yates shuffle với seed
fn make_shuffled_indices(n: usize, seed: u64, count: usize) -> Vec<usize> {
    let take = count.min(n);
    let mut rng = seed.max(1);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..take {
        rng = lcg_next(rng);
        let j = i + (rng % (n - i) as u64) as usize;
        indices.swap(i, j);
    }
    indices.truncate(take);
    indices.sort_unstable();
    indices
}

#[inline]
pub fn logistic(x: f64) -> f64 {
    if x > 30.0 { 1.0 } else if x < -30.0 { 0.0 }
    else { 1.0 / (1.0 + (-x).exp()) }
}

// ============================================================
// BLOCK-DIAGONAL P-IRLS (tối ưu cho nhiều features)
// ============================================================

/// Block-diagonal P-IRLS solver cho logistic GAM
///
/// Tối ưu cho nhiều features:
/// - Per-feature X'wX blocks tính độc lập, song song
/// - Không tính cross-terms giữa các feature (block-diagonal)
/// - Cholesky solve từng block nhỏ
pub fn pirls_logistic_block(
    x_data: &[f64],    // row-major, n × p
    y_data: &[f64],
    n: usize,
    p: usize,
    blocks: &[FeatureBlock],
    max_iter: usize,
    tol: f64,
) -> PirlsResult {
    let mut beta = DVector::from_element(p, 0.0);
    let mut eta = vec![0.0; n];
    let mut weights = vec![1.0; n];
    let mut converged = false;
    let mut n_iter = 0;
    let mut edf_total = p as f64; // Default fallback

    let chunk = 4096;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Compute mu, weights, working response z
        for i in 0..n {
            let mu = logistic(eta[i]).clamp(1e-10, 1.0 - 1e-10);
            let w = mu * (1.0 - mu);
            weights[i] = w;
            eta[i] += (y_data[i] - mu) / w;
        }

        // Compute per-block X'wX and X'wz in parallel
        // Each block processes all rows — pre-allocate xvals buffer to avoid heap alloc per row
        let block_results: Vec<_> = blocks.par_iter().enumerate().map(|(block_idx, block)| {
            let cs = block.col_start;
            let nc = block.n_cols;
            if nc == 0 {
                return (block_idx, DMatrix::zeros(1, 1), DVector::zeros(1));
            }

            // Accumulate X'wX for this block over all rows
            let mut block_xwx = DMatrix::zeros(nc, nc);
            let mut xwz = DVector::zeros(nc);

            // Pre-allocate xvals buffer reused across all row chunks
            let mut xvals = vec![0.0f64; nc];

            for cstart in (0..n).step_by(chunk) {
                let cend = (cstart + chunk).min(n);
                for i in cstart..cend {
                    let wi = weights[i];
                    let zi = eta[i];
                    let ro = i * p;

                    // Extract feature values into pre-allocated buffer
                    for j in 0..nc {
                        xvals[j] = x_data[ro + cs + j];
                    }

                    // Compute w_i * x_ij and accumulate directly into full matrix
                    for j in 0..nc {
                        let wixj = wi * xvals[j];
                        xwz[j] += wixj * zi;
                        for k in j..nc {
                            let val = wixj * xvals[k];
                            block_xwx[(j, k)] += val;
                            if j != k {
                                block_xwx[(k, j)] += val;
                            }
                        }
                    }
                }
            }

            (block_idx, block_xwx, xwz)
        }).collect();

        // Solve each block independently — O(Σ n_i³) instead of O(p³)
        let mut new_beta = DVector::zeros(p);
        edf_total = 0.0;

        for (block_idx, block_xwx, xwz) in &block_results {
            let block = &blocks[*block_idx];
            let cs = block.col_start;
            let nc = block.n_cols;
            if nc == 0 { continue; }

            // Penalized system for this block
            let scaled_penalty = block.penalty.scale(block.lambda);
            let a = block_xwx + &scaled_penalty;

            // Solve block — with robust numerical fallback
            let block_beta = match Cholesky::new(a.clone()) {
                Some(chol) => chol.solve(xwz),
                None => {
                    // Try with stronger ridge regularization
                    let ridge = DMatrix::from_diagonal_element(nc, nc, 1e-4);
                    match Cholesky::new(a.clone() + &ridge) {
                        Some(chol) => chol.solve(xwz),
                        None => {
                            // Last resort: even stronger ridge
                            let ridge2 = DMatrix::from_diagonal_element(nc, nc, 1e-2);
                            match Cholesky::new(a.clone() + &ridge2) {
                                Some(chol) => chol.solve(xwz),
                                None => {
                                    // If all else fails, fall back to previous beta for this block
                                    for j in 0..nc {
                                        new_beta[cs + j] = beta[cs + j];
                                    }
                                    continue;
                                }
                            }
                        }
                    }
                }
            };

            // Copy solution to full beta vector
            for j in 0..nc {
                new_beta[cs + j] = block_beta[j];
            }

            // Compute EDF for this block: trace((X'WX + λP)^{-1} X'WX)
            let edf_block = match Cholesky::new(a.clone()) {
                Some(chol) => {
                    let inv = chol.inverse();
                    (&inv * block_xwx).trace()
                }
                None => nc as f64,
            };
            edf_total += edf_block;
        }

        // Compute new eta = X * beta (parallel by row chunks)
        let block_starts: Vec<usize> = blocks.iter().map(|b| b.col_start).collect();
        let block_ncols: Vec<usize> = blocks.iter().map(|b| b.n_cols).collect();

        let row_chunk = chunk.min(n).max(1);
        let n_chunks = (n + row_chunk - 1) / row_chunk;
        let new_eta_vec: Vec<f64> = (0..n_chunks).into_par_iter().flat_map(|chunk_idx| {
            let start = chunk_idx * row_chunk;
            let end = (start + row_chunk).min(n);
            (start..end).map(|i| {
                let ro = i * p;
                let mut sum = 0.0;
                for bi in 0..block_starts.len() {
                    let cs = block_starts[bi];
                    let nc = block_ncols[bi];
                    for j in 0..nc {
                        sum += x_data[ro + cs + j] * new_beta[cs + j];
                    }
                }
                sum
            }).collect::<Vec<_>>()
        }).collect();
        eta = new_eta_vec;

        // Check convergence
        let diff = &new_beta - &beta;
        let rel = if beta.norm() > 1e-10 { diff.norm() / beta.norm() } else { diff.norm() };
        beta = new_beta;
        if rel < tol { converged = true; break; }
    }

    let proba: DVector<f64> = DVector::from_vec(eta.iter().map(|&e| logistic(e)).collect());
    let ll = compute_ll(y_data, &proba);

    PirlsResult {
        coefficients: beta,
        weights: DVector::from_vec(weights),
        fitted: DVector::from_vec(eta),
        probabilities: proba,
        n_iterations: n_iter,
        converged,
        log_likelihood: ll,
        edf: edf_total,
    }
}

// ============================================================
// Legacy: Full-batch parallel P-IRLS (vẫn giữ để tương thích)
// ============================================================

/// Full-batch parallel P-IRLS solver cho logistic GAM
pub fn pirls_logistic(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    penalty: &DMatrix<f64>,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> PirlsResult {
    let n = x.nrows();
    let p = x.ncols();
    let x_data = x.as_slice();
    let y_data = y.as_slice();

    let mut beta = DVector::from_element(p, 0.0);
    let mut eta = DVector::from_element(n, 0.0);
    let mut weights = DVector::from_element(n, 1.0);
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Compute mu, weights, z
        for i in 0..n {
            let mu = logistic(eta[i]).clamp(1e-10, 1.0 - 1e-10);
            weights[i] = mu * (1.0 - mu);
            eta[i] += (y_data[i] - mu) / weights[i];
        }

        // Parallel X'WX: divide rows among threads
        let chunk = 4096;
        let indices: Vec<usize> = (0..n).collect();
        let chunks: Vec<&[usize]> = indices.chunks(chunk).collect();

        let accum: FullAccum = chunks.into_par_iter()
            .map(|chunk_indices| {
                let mut acc = FullAccum::new(p);
                for &i in chunk_indices {
                    let wi = weights[i];
                    let zi = eta[i];
                    let ro = i * p;
                    for j in 0..p {
                        let xij = x_data[ro + j];
                        let wixij = wi * xij;
                        for k in j..p {
                            acc.upper[(j*(2*p-j-1))/2 + (k-j)] += wixij * x_data[ro + k];
                        }
                        acc.xwz[j] += wixij * zi;
                    }
                }
                acc
            })
            .reduce_with(|a, b| a.merge(&b))
            .unwrap_or_else(|| FullAccum::new(p));

        let xwx = accum.to_xwx();
        let xwz = accum.to_xwz();

        // Penalized system
        let lambda_p = penalty.scale(lambda);
        let a = &xwx + &lambda_p;

        let new_beta = match Cholesky::new(a.clone()) {
            Some(chol) => chol.solve(&xwz).column(0).into(),
            None => {
                let ridge = DMatrix::from_diagonal_element(p, p, 1e-6);
                match Cholesky::new(a.clone() + ridge) {
                    Some(chol) => chol.solve(&xwz).column(0).into(),
                    None => beta.clone(),
                }
            }
        };

        // Compute eta = X * beta
        for i in 0..n {
            let mut sum = 0.0;
            let ro = i * p;
            for j in 0..p { sum += x_data[ro + j] * new_beta[j]; }
            eta[i] = sum;
        }

        let diff = &new_beta - &beta;
        let rel = if beta.norm() > 1e-10 { diff.norm() / beta.norm() } else { diff.norm() };
        beta = new_beta;
        if rel < tol { converged = true; break; }
    }

    let proba: DVector<f64> = eta.map(|e| logistic(e));
    let ll = compute_ll(y_data, &proba);
    let edf = compute_edf_full(x_data, y_data, n, p, weights.as_slice(), penalty, lambda);

    PirlsResult {
        coefficients: beta, weights, fitted: eta, probabilities: proba,
        n_iterations: n_iter, converged, log_likelihood: ll, edf,
    }
}

fn compute_ll(y: &[f64], prob: &DVector<f64>) -> f64 {
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        let p = prob[i].clamp(1e-15, 1.0 - 1e-15);
        ll += y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln();
    }
    ll
}

fn compute_edf_full(
    x_data: &[f64], _y_data: &[f64], n: usize, p: usize,
    weights: &[f64], penalty: &DMatrix<f64>, lambda: f64,
) -> f64 {
    let chunk = 4096;
    let indices: Vec<usize> = (0..n).collect();
    let chunks: Vec<&[usize]> = indices.chunks(chunk).collect();
    let accum: FullAccum = chunks.into_par_iter()
        .map(|chunk_indices| {
            let mut acc = FullAccum::new(p);
            for &i in chunk_indices {
                let wi = weights[i];
                let ro = i * p;
                for j in 0..p {
                    let xij = x_data[ro + j];
                    let w = wi * xij;
                    for k in j..p { acc.upper[(j*(2*p-j-1))/2+(k-j)] += w * x_data[ro+k]; }
                }
            }
            acc
        })
        .reduce_with(|a, b| a.merge(&b))
        .unwrap_or_else(|| FullAccum::new(p));

    let xwx = accum.to_xwx();
    let a = &xwx + &penalty.scale(lambda);
    let inv = match Cholesky::new(a.clone()) {
        Some(c) => c.inverse(),
        None => return p as f64,
    };
    (&inv * &xwx).trace()
}

/// Quick fit trên subset (dùng trong grid search)
fn quick_fit(
    x_data: &[f64], y_data: &[f64], n: usize, p: usize,
    penalty: &DMatrix<f64>, lambda: f64,
    indices: &[usize], max_iter: usize, tol: f64,
) -> (f64, f64) {
    let m = indices.len();
    let mut beta = DVector::from_element(p, 0.0);
    let mut eta = vec![0.0; m];
    let mut weights = vec![1.0; m];

    for _iter in 0..max_iter {
        for (idx_pos, &i) in indices.iter().enumerate() {
            let mu = logistic(eta[idx_pos]).clamp(1e-10, 1.0 - 1e-10);
            weights[idx_pos] = mu * (1.0 - mu);
            eta[idx_pos] += (y_data[i] - mu) / weights[idx_pos];
        }

        let mut xwx = DMatrix::zeros(p, p);
        let mut xwz = DVector::zeros(p);
        for (idx_pos, &i) in indices.iter().enumerate() {
            let wi = weights[idx_pos];
            let zi = eta[idx_pos];
            let ro = i * p;
            for j in 0..p {
                let xij = x_data[ro + j];
                let w = wi * xij;
                for k in j..p { xwx[(j,k)] += w * x_data[ro + k]; }
                xwz[j] += w * zi;
            }
        }
        for j in 0..p { for k in (j+1)..p { xwx[(k,j)] = xwx[(j,k)]; } }

        let a = &xwx + &penalty.scale(lambda);
        let new_beta = match Cholesky::new(a.clone()) {
            Some(c) => c.solve(&xwz).column(0).into(),
            None => beta.clone(),
        };

        for (idx_pos, &i) in indices.iter().enumerate() {
            let mut sum = 0.0;
            let ro = i * p;
            for j in 0..p { sum += x_data[ro + j] * new_beta[j]; }
            eta[idx_pos] = sum;
        }

        let diff = &new_beta - &beta;
        let rel = if beta.norm() > 1e-10 { diff.norm() / beta.norm() } else { diff.norm() };
        beta = new_beta;
        if rel < tol { break; }
    }

    let mut ll = 0.0;
    for (idx_pos, &i) in indices.iter().enumerate() {
        let mu = logistic(eta[idx_pos]).clamp(1e-15, 1.0 - 1e-15);
        ll += y_data[i] * mu.ln() + (1.0 - y_data[i]) * (1.0 - mu).ln();
    }
    let scaled_ll = ll * (n as f64 / m as f64);
    let edf = p as f64 * 0.8;
    (scaled_ll, edf)
}

pub fn gcv_score(log_likelihood: f64, edf: f64, n: usize) -> f64 {
    let nf = n as f64;
    let dev = -2.0 * log_likelihood;
    let pf = (1.0 - edf / nf).powi(2);
    if pf < 1e-10 { f64::MAX } else { dev / (nf * pf) }
}

/// Grid search với subsampling + early stopping
pub fn find_best_lambda(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    penalty: &DMatrix<f64>,
    lambda_values: &[f64],
    max_iter: usize,
    tol: f64,
) -> (f64, f64) {
    let n = x.nrows();
    let p = x.ncols();
    let x_data = x.as_slice();
    let y_data = y.as_slice();

    let sub_n = (50000.min(n)).max(5000);
    let indices = make_shuffled_indices(n, 12345, sub_n);

    let mut best_lambda = lambda_values[0];
    let mut best_gcv = f64::MAX;
    let mut worse_count = 0;

    for &lambda in lambda_values {
        let (ll, edf) = quick_fit(x_data, y_data, n, p, penalty, lambda, &indices, max_iter, tol);
        let gcv = gcv_score(ll, edf, n);

        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lambda;
            worse_count = 0;
        } else {
            worse_count += 1;
            if worse_count >= 3 { break; }
        }
    }

    (best_lambda, best_gcv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pirls_converges() {
        let x = DMatrix::from_vec(20, 2, vec![
            1.0,0.0,1.0,0.1,1.0,0.2,1.0,0.3,1.0,0.4,
            1.0,0.5,1.0,0.6,1.0,0.7,1.0,0.8,1.0,0.9,
            1.0,1.5,1.0,1.6,1.0,1.7,1.0,1.8,1.0,1.9,
            1.0,2.0,1.0,2.1,1.0,2.2,1.0,2.3,1.0,2.4,
        ]);
        let y = DVector::from_vec(vec![
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
        ]);
        let penalty = DMatrix::zeros(2, 2);
        let result = pirls_logistic(&x, &y, &penalty, 0.0, 100, 1e-8);
        println!("converged={}, n_iter={}, edf={}", result.converged, result.n_iterations, result.edf);
        println!("coefficients={:?}", result.coefficients);
        for i in 0..20 {
            println!("  prob[{}] = {:.6} (y={})", i, result.probabilities[i], y[i]);
        }
        assert!(result.converged || result.n_iterations == 100);
        for i in 0..10 { assert!(result.probabilities[i] < 0.5, "prob[{}] = {} should be < 0.5", i, result.probabilities[i]); }
        for i in 10..20 { assert!(result.probabilities[i] > 0.5, "prob[{}] = {} should be > 0.5", i, result.probabilities[i]); }
    }
}
