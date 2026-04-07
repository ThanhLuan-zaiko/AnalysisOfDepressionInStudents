//! # Rust Engine — GAM (Generalized Additive Model) Implementation
//!
//! Tự implement GAM spline fitting trong Rust, thay thế pyGAM.
//! 
//! Kiến trúc:
//! - B-spline basis functions (bậc 3, cubic)
//! - P-IRLS solver (Penalized Iteratively Reweighted Least Squares)
//! - GCV (Generalized Cross Validation) cho automatic smoothing
//! - Logistic link function cho binary classification
//! - Parallel cross-validation với Rayon

pub mod bspline;
pub mod pirls;
pub mod gam;
pub mod cross_val;
pub mod python_bindings;

pub use gam::GAMClassifier;
pub use cross_val::run_cross_validation;
