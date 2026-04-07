//! PyO3 Python bindings

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, IntoPyArray};

use crate::gam::GAMClassifier;
use crate::cross_val::run_cross_validation;

/// Python wrapper cho GAMClassifier
#[pyclass]
pub struct PyGAMClassifier {
    classifier: GAMClassifier,
}

#[pymethods]
impl PyGAMClassifier {
    #[new]
    #[pyo3(signature = (n_splines=10, optimize_lambda=true, _random_seed=42))]
    fn new(n_splines: usize, optimize_lambda: bool, _random_seed: u64) -> Self {
        let mut gam = GAMClassifier::new(
            Vec::new(),
            Vec::new(),
            n_splines,
        );
        gam.optimize_lambda = optimize_lambda;
        PyGAMClassifier { classifier: gam }
    }

    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        feature_types: Vec<String>,
        feature_names: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x_shape = x.shape();
        let n_samples = x_shape[0];
        let n_features = x_shape[1];

        let x_slice = x.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot read X array: {}", e))
        })?;
        let y_slice = y.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot read y array: {}", e))
        })?;

        let result = self.classifier.fit(
            x_slice,
            y_slice,
            n_samples,
            n_features,
            &feature_types,
            &feature_names,
        );

        let dict = PyDict::new(py);
        dict.set_item("roc_auc", result.roc_auc)?;
        dict.set_item("pr_auc", result.pr_auc)?;
        dict.set_item("f1", result.f1)?;
        dict.set_item("recall", result.recall)?;
        dict.set_item("precision", result.precision)?;
        dict.set_item("brier_score", result.brier_score)?;
        dict.set_item("n_iterations", result.n_iterations as i64)?;
        dict.set_item("converged", result.converged)?;
        dict.set_item("best_lambda", result.best_lambda)?;
        dict.set_item("gcv_score", result.gcv_score)?;

        let importance_list = PyList::empty(py);
        for imp in &result.feature_importance {
            let imp_dict = PyDict::new(py);
            imp_dict.set_item("feature", &imp.feature)?;
            imp_dict.set_item("variance_importance", imp.variance_importance)?;
            imp_dict.set_item("term_index", imp.term_index as i64)?;
            importance_list.append(imp_dict)?;
        }
        dict.set_item("feature_importance", importance_list)?;

        Ok(dict)
    }

    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        feature_types: Vec<String>,
        feature_names: Vec<String>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let x_shape = x.shape();
        let n_samples = x_shape[0];
        let n_features = x_shape[1];

        let x_slice = x.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot read X array: {}", e))
        })?;

        let proba = self.classifier.predict_proba_raw(
            x_slice,
            n_samples,
            n_features,
            &feature_types,
            &feature_names,
        );

        Ok(proba.into_pyarray(py))
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        proba: PyReadonlyArray1<'py, f64>,
        threshold: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<i32>>> {
        let proba_slice = proba.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot read proba: {}", e))
        })?;

        let labels = self.classifier.predict(proba_slice, threshold);
        Ok(labels.into_pyarray(py))
    }

    pub fn get_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let coef_vec = match &self.classifier.coefficients {
            Some(c) => c.iter().copied().collect(),
            None => vec![],
        };
        coef_vec.into_pyarray(py)
    }

    pub fn get_lambdas(&self) -> Vec<f64> {
        self.classifier.lambdas.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, feature_types, feature_names, n_splits=5, n_splines=10, random_seed=42))]
pub fn cross_validate_gam<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    feature_types: Vec<String>,
    feature_names: Vec<String>,
    n_splits: usize,
    n_splines: usize,
    random_seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let x_shape = x.shape();
    let n_samples = x_shape[0];
    let n_features = x_shape[1];

    let x_slice = x.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot read X array: {}", e))
    })?;
    let y_slice = y.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot read y array: {}", e))
    })?;

    let cv_result = run_cross_validation(
        x_slice,
        y_slice,
        n_samples,
        n_features,
        &feature_types,
        &feature_names,
        n_splits,
        n_splines,
        random_seed,
    );

    let dict = PyDict::new(py);
    dict.set_item("mean_roc_auc", cv_result.mean_roc_auc)?;
    dict.set_item("std_roc_auc", cv_result.std_roc_auc)?;
    dict.set_item("mean_pr_auc", cv_result.mean_pr_auc)?;
    dict.set_item("std_pr_auc", cv_result.std_pr_auc)?;
    dict.set_item("mean_f1", cv_result.mean_f1)?;
    dict.set_item("std_f1", cv_result.std_f1)?;
    dict.set_item("mean_recall", cv_result.mean_recall)?;
    dict.set_item("std_recall", cv_result.std_recall)?;
    dict.set_item("mean_precision", cv_result.mean_precision)?;
    dict.set_item("std_precision", cv_result.std_precision)?;
    dict.set_item("mean_brier_score", cv_result.mean_brier)?;
    dict.set_item("std_brier_score", cv_result.std_brier)?;
    dict.set_item("converged_folds", cv_result.converged_folds as i64)?;
    dict.set_item("n_splits", n_splits as i64)?;

    let folds_list = PyList::empty(py);
    for fold in &cv_result.folds {
        let fold_dict = PyDict::new(py);
        fold_dict.set_item("fold", fold.fold_idx as i64)?;
        fold_dict.set_item("roc_auc", fold.roc_auc)?;
        fold_dict.set_item("pr_auc", fold.pr_auc)?;
        fold_dict.set_item("f1", fold.f1)?;
        fold_dict.set_item("recall", fold.recall)?;
        fold_dict.set_item("precision", fold.precision)?;
        fold_dict.set_item("brier_score", fold.brier_score)?;
        folds_list.append(fold_dict)?;
    }
    dict.set_item("folds", folds_list)?;

    Ok(dict)
}

#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGAMClassifier>()?;
    m.add_function(wrap_pyfunction!(cross_validate_gam, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
