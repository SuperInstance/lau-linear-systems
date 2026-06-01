//! Controllability and observability matrices and rank tests.

use nalgebra::DMatrix;
use crate::ss::{StateSpaceContinuous, StateSpaceDiscrete};

/// Build the controllability matrix: [B, AB, A²B, ..., A^(n-1)B]
pub fn controllability_matrix(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let m = b.ncols();
    let mut cols = Vec::new();
    let mut akb = b.clone();
    for _ in 0..n {
        cols.push(akb.clone());
        akb = a * &akb;
    }
    let mut result = DMatrix::zeros(n, n * m);
    for (k, col) in cols.iter().enumerate() {
        for i in 0..n {
            for j in 0..m {
                result[(i, k * m + j)] = col[(i, j)];
            }
        }
    }
    result
}

/// Build the observability matrix: [C; CA; CA²; ...; CA^(n-1)]
pub fn observability_matrix(a: &DMatrix<f64>, c: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let p = c.nrows();
    let mut rows = Vec::new();
    let mut cak = c.clone();
    for _ in 0..n {
        rows.push(cak.clone());
        cak = &cak * a;
    }
    let mut result = DMatrix::zeros(n * p, n);
    for (k, row) in rows.iter().enumerate() {
        for i in 0..p {
            for j in 0..n {
                result[(k * p + i, j)] = row[(i, j)];
            }
        }
    }
    result
}

/// Numerical rank using SVD-like threshold.
fn numerical_rank(mat: &DMatrix<f64>) -> usize {
    let svd = mat.clone().svd(true, true);
    let sigma = svd.singular_values;
    let tol = sigma[0] * mat.nrows().max(mat.ncols()) as f64 * 1e-10;
    sigma.iter().filter(|&&s| s > tol).count()
}

/// Check if (A, B) is controllable.
pub fn is_controllable(a: &DMatrix<f64>, b: &DMatrix<f64>) -> bool {
    let ctrb = controllability_matrix(a, b);
    numerical_rank(&ctrb) == a.nrows()
}

/// Check if (A, C) is observable.
pub fn is_observable(a: &DMatrix<f64>, c: &DMatrix<f64>) -> bool {
    let obs = observability_matrix(a, c);
    numerical_rank(&obs) == a.nrows()
}

impl StateSpaceContinuous {
    pub fn controllability(&self) -> DMatrix<f64> {
        controllability_matrix(&self.a, &self.b)
    }
    pub fn observability(&self) -> DMatrix<f64> {
        observability_matrix(&self.a, &self.c)
    }
    pub fn is_controllable(&self) -> bool {
        is_controllable(&self.a, &self.b)
    }
    pub fn is_observable(&self) -> bool {
        is_observable(&self.a, &self.c)
    }
}

impl StateSpaceDiscrete {
    pub fn controllability(&self) -> DMatrix<f64> {
        controllability_matrix(&self.a, &self.b)
    }
    pub fn observability(&self) -> DMatrix<f64> {
        observability_matrix(&self.a, &self.c)
    }
    pub fn is_controllable(&self) -> bool {
        is_controllable(&self.a, &self.b)
    }
    pub fn is_observable(&self) -> bool {
        is_observable(&self.a, &self.c)
    }
}
