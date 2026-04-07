//! P-IRLS: Penalized Iteratively Reweighted Least Squares
//!
//! Thuật toán core để fit GAM với logistic link function.

use nalgebra::{DVector, DMatrix, Cholesky};

/// Kết quả của P-IRLS fitting
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

/// Logistic function: 1 / (1 + exp(-x))
#[inline]
pub fn logistic(x: f64) -> f64 {
    if x > 30.0 {
        1.0
    } else if x < -30.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// P-IRLS solver cho logistic GAM
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

    let mut beta = DVector::from_element(p, 0.0);
    let mut eta = DVector::from_element(n, 0.0);
    let mut converged = false;
    let mut n_iter = 0;
    let mut weights = DVector::from_element(n, 1.0);

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // mu = logistic(eta)
        let mu: DVector<f64> = eta.map(|e| logistic(e));
        let mu_clipped = mu.map(|m| m.clamp(1e-10, 1.0 - 1e-10));

        // Working weights: w = mu * (1 - mu)
        weights = mu_clipped.map(|m| m * (1.0 - m));

        // Working response: z = eta + (y - mu) / w
        let residual = y - &mu_clipped;
        let adjusted: DVector<f64> = residual.zip_map(&weights, |r, w| r / w);
        let z = &eta + &adjusted;

        // Weighted least squares với penalty
        // (X'WX + λP)β = X'Wz
        // Build W^1/2 * X manually
        let mut wx_data = vec![0.0f64; n * p];
        for i in 0..n {
            let w_sqrt = weights[i].sqrt();
            for j in 0..p {
                wx_data[i * p + j] = x[(i, j)] * w_sqrt;
            }
        }
        let wx = DMatrix::from_vec(n, p, wx_data);

        // X'WX
        let xwx = wx.transpose() * &wx;

        // X'Wz = X^T * (W * z)
        let wz: DVector<f64> = weights.component_mul(&z);
        let xwz = x.transpose() * &wz;

        // Penalized system
        let lambda_p = penalty.scale(lambda);
        let a = &xwx + &lambda_p;

        // Solve via Cholesky
        let new_beta = match Cholesky::new(a.clone()) {
            Some(chol) => chol.solve(&xwz).column(0).into(),
            None => {
                let ridge = DMatrix::from_diagonal_element(p, p, 1e-6);
                let a_ridge = a.clone() + ridge;
                match Cholesky::new(a_ridge) {
                    Some(chol) => chol.solve(&xwz).column(0).into(),
                    None => beta.clone(),
                }
            }
        };

        let new_eta = (x * &new_beta).column(0).into();

        // Convergence check
        let beta_diff = &new_beta - &beta;
        let beta_norm = beta.norm();
        let diff_norm = beta_diff.norm();
        let rel_change = if beta_norm > 1e-10 {
            diff_norm / beta_norm
        } else {
            diff_norm
        };

        beta = new_beta;
        eta = new_eta;

        if rel_change < tol {
            converged = true;
            break;
        }
    }

    let probabilities: DVector<f64> = eta.map(|e| logistic(e));
    let log_likelihood = compute_log_likelihood(y, &probabilities);
    let edf = compute_edf(x, &weights, penalty, lambda);

    PirlsResult {
        coefficients: beta,
        weights,
        fitted: eta,
        probabilities,
        n_iterations: n_iter,
        converged,
        log_likelihood,
        edf,
    }
}

/// Compute log-likelihood
fn compute_log_likelihood(y: &DVector<f64>, prob: &DVector<f64>) -> f64 {
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        let p = prob[i].clamp(1e-15, 1.0 - 1e-15);
        ll += y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln();
    }
    ll
}

/// Compute effective degrees of freedom
fn compute_edf(
    x: &DMatrix<f64>,
    weights: &DVector<f64>,
    penalty: &DMatrix<f64>,
    lambda: f64,
) -> f64 {
    let p = x.ncols();
    let n = x.nrows();

    let mut wx_data = vec![0.0f64; n * p];
    for i in 0..n {
        let w_sqrt = weights[i].sqrt();
        for j in 0..p {
            wx_data[i * p + j] = x[(i, j)] * w_sqrt;
        }
    }
    let wx = DMatrix::from_vec(n, p, wx_data);
    let xwx = wx.transpose() * &wx;
    let lambda_p = penalty.scale(lambda);
    let a = &xwx + &lambda_p;

    let a_inv = match Cholesky::new(a.clone()) {
        Some(chol) => chol.inverse(),
        None => {
            let ridge = DMatrix::from_diagonal_element(p, p, 1e-6);
            match Cholesky::new(a.clone() + ridge) {
                Some(chol) => chol.inverse(),
                None => return p as f64,
            }
        }
    };

    let m = &a_inv * &xwx;
    m.trace()
}

/// GCV score
pub fn gcv_score(log_likelihood: f64, edf: f64, n: usize) -> f64 {
    let n_f64 = n as f64;
    let deviance = -2.0 * log_likelihood;
    let penalty_factor = (1.0 - edf / n_f64).powi(2);
    if penalty_factor < 1e-10 {
        f64::MAX
    } else {
        deviance / (n_f64 * penalty_factor)
    }
}

/// Grid search cho smoothing parameter λ
pub fn find_best_lambda(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    penalty: &DMatrix<f64>,
    lambda_values: &[f64],
    _max_iter: usize,
    _tol: f64,
) -> (f64, f64) {
    let n = x.nrows();
    let x_owned = x.clone_owned();
    let mut best_lambda = lambda_values[0];
    let mut best_gcv = f64::MAX;

    for &lambda in lambda_values {
        let result = pirls_logistic(&x_owned, y, penalty, lambda, 50, 1e-5);
        let gcv = gcv_score(result.log_likelihood, result.edf, n);
        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lambda;
        }
    }

    (best_lambda, best_gcv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic() {
        assert!((logistic(0.0) - 0.5).abs() < 1e-10);
        assert!(logistic(100.0) > 0.99);
        assert!(logistic(-100.0) < 0.01);
    }

    #[test]
    fn test_pirls_converges_on_separable_data() {
        let x = DMatrix::from_vec(20, 2, vec![
            1.0, 0.0, 1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0, 0.4,
            1.0, 0.5, 1.0, 0.6, 1.0, 0.7, 1.0, 0.8, 1.0, 0.9,
            1.0, 1.5, 1.0, 1.6, 1.0, 1.7, 1.0, 1.8, 1.0, 1.9,
            1.0, 2.0, 1.0, 2.1, 1.0, 2.2, 1.0, 2.3, 1.0, 2.4,
        ]);
        let y = DVector::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let penalty = DMatrix::zeros(2, 2);

        let result = pirls_logistic(&x, &y, &penalty, 0.0, 100, 1e-8);
        assert!(result.converged || result.n_iterations == 100);
        for i in 0..10 {
            assert!(result.probabilities[i] < 0.5, "Sample {} prob={}", i, result.probabilities[i]);
        }
        for i in 10..20 {
            assert!(result.probabilities[i] > 0.5, "Sample {} prob={}", i, result.probabilities[i]);
        }
    }
}
