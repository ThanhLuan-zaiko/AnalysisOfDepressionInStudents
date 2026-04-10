//! # Rust Engine — GAM (Generalized Additive Model) Implementation
//!
//! Tối ưu cho nhiều features:
//! - Block-diagonal P-IRLS: tính per-feature blocks song song
//! - Cached design matrix: build 1 lần, reuse qua CV folds
//! - Parallel folds với Rayon

pub mod bspline;
pub mod pirls;
pub mod gam;
pub mod cross_val;
pub mod python_bindings;

pub use gam::GAMClassifier;
pub use cross_val::run_cross_validation_cached;
pub use pirls::FeatureBlock;
