//! B-spline basis functions implementation
//!
//! Cubic B-splines (degree=3) với automatic knot placement.
//! Dùng de Boor's algorithm để tính basis values.

use nalgebra::{DVector, DMatrix};

/// Knot sequence cho B-spline
#[derive(Debug, Clone)]
pub struct KnotSequence {
    pub knots: DVector<f64>,
    pub degree: usize,
}

impl KnotSequence {
    /// Tạo knot sequence đều cho data range
    pub fn uniform(n_knots: usize, x_min: f64, x_max: f64, degree: usize) -> Self {
        let n_internal = n_knots.saturating_sub(2);
        let total_len = n_knots + 2 * degree;
        let mut knots = DVector::zeros(total_len);

        let knot_len = knots.len();
        // Boundary knots (repeat degree+1 times)
        for i in 0..=degree {
            knots[i] = x_min;
            knots[knot_len - 1 - i] = x_max;
        }

        // Internal knots
        if n_internal > 0 {
            for i in 0..n_internal {
                let t = (i + 1) as f64 / (n_internal + 1) as f64;
                knots[degree + 1 + i] = x_min + t * (x_max - x_min);
            }
        }

        KnotSequence { knots, degree }
    }

    /// Số basis functions = n_knots - degree - 1
    pub fn n_basis(&self) -> usize {
        self.knots.len().saturating_sub(self.degree + 1)
    }

    /// Tính B-spline basis value tại điểm x cho basis index i
    /// Dùng Cox-de Boor recursion formula
    pub fn basis_value(&self, i: usize, x: f64) -> f64 {
        self.cox_de_boor(i, self.degree, x)
    }

    fn cox_de_boor(&self, i: usize, p: usize, x: f64) -> f64 {
        if p == 0 {
            if (self.knots[i] <= x && x < self.knots[i + 1]) || 
               (i == self.knots.len() - self.degree - 2 && x == self.knots[i + 1]) {
                1.0
            } else {
                0.0
            }
        } else {
            let mut val = 0.0;
            let denom1 = self.knots[i + p] - self.knots[i];
            if denom1.abs() > 1e-12 {
                val += (x - self.knots[i]) / denom1 * self.cox_de_boor(i, p - 1, x);
            }
            let denom2 = self.knots[i + p + 1] - self.knots[i + 1];
            if denom2.abs() > 1e-12 {
                val += (self.knots[i + p + 1] - x) / denom2 * self.cox_de_boor(i + 1, p - 1, x);
            }
            val
        }
    }
}

/// B-spline basis matrix cho một feature vector
/// Shape: (n_samples, n_basis_functions)
pub fn create_basis_matrix(x: &[f64], n_splines: usize, degree: usize) -> DMatrix<f64> {
    if x.is_empty() {
        return DMatrix::zeros(0, 0);
    }

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Handle edge case: all same value
    let x_range = if (x_max - x_min).abs() < 1e-12 {
        1.0
    } else {
        x_max - x_min
    };

    let n_knots = n_splines + degree + 1;
    let knots = KnotSequence::uniform(n_knots, x_min - 0.01 * x_range, x_max + 0.01 * x_range, degree);
    let n_basis = knots.n_basis();
    let n = x.len();

    let mut matrix = DMatrix::zeros(n, n_basis);
    for (i, &xi) in x.iter().enumerate() {
        for j in 0..n_basis {
            matrix[(i, j)] = knots.basis_value(j, xi);
        }
    }

    matrix
}

/// Penalty matrix cho B-splines (integral of squared second derivative)
/// P[i,j] = integral of B_i''(x) * B_j''(x) dx
/// Simplified: dùng finite difference approximation thay vì numerical integration
pub fn create_penalty_matrix(n_splines: usize, _x_min: f64, _x_max: f64, degree: usize) -> DMatrix<f64> {
    let n_knots = n_splines + degree + 1;
    let knots = KnotSequence::uniform(n_knots, 0.0, 1.0, degree);
    let n_basis = knots.n_basis();

    // Simplified penalty: second-order finite difference matrix
    // P = D'D where D is second difference operator
    let mut penalty = DMatrix::zeros(n_basis, n_basis);
    
    for i in 0..n_basis.saturating_sub(2) {
        penalty[(i, i)] += 1.0;
        penalty[(i, i+1)] -= 2.0;
        penalty[(i, i+2)] += 1.0;
        
        penalty[(i+1, i)] -= 2.0;
        penalty[(i+1, i+1)] += 4.0;
        penalty[(i+1, i+2)] -= 2.0;
        
        penalty[(i+2, i)] += 1.0;
        penalty[(i+2, i+1)] -= 2.0;
        penalty[(i+2, i+2)] += 1.0;
    }

    penalty
}

/// Feature term types
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    Numeric,    // Smooth spline
    Ordinal,    // Smooth spline (treat as continuous)
    Nominal,    // Factor/one-hot encoding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_matrix_shape() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let matrix = create_basis_matrix(&x, 10, 3);
        assert_eq!(matrix.nrows(), 100);
        assert!(matrix.ncols() > 0);
    }

    #[test]
    fn test_basis_partition_of_unity() {
        let x = vec![0.3f64];
        let matrix = create_basis_matrix(&x, 10, 3);
        let row_sum: f64 = (0..matrix.ncols()).map(|j| matrix[(0, j)]).sum();
        assert!((row_sum - 1.0).abs() < 0.01);
    }
}
