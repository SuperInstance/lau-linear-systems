//! State-space model definitions.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Continuous-time state-space model: dx/dt = A x + B u, y = C x + D u
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceContinuous {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
}

/// Discrete-time state-space model: x[k+1] = A x[k] + B u[k], y[k] = C x[k] + D u[k]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceDiscrete {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub dt: f64,
}

/// Generic state-space model (continuous or discrete).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateSpace {
    Continuous(StateSpaceContinuous),
    Discrete(StateSpaceDiscrete),
}

impl StateSpaceContinuous {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        assert_eq!(a.nrows(), a.ncols(), "A must be square");
        assert_eq!(a.nrows(), b.nrows(), "A rows must match B rows");
        assert_eq!(a.ncols(), c.ncols(), "A cols must match C cols");
        assert_eq!(b.ncols(), d.ncols(), "B cols must match D cols");
        assert_eq!(c.nrows(), d.nrows(), "C rows must match D rows");
        Self { a, b, c, d }
    }

    pub fn states(&self) -> usize { self.a.nrows() }
    pub fn inputs(&self) -> usize { self.b.ncols() }
    pub fn outputs(&self) -> usize { self.c.nrows() }

    /// Simulate one step of the continuous system using Euler integration.
    pub fn step(&self, x: &DVector<f64>, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let dx = &self.a * x + &self.b * u;
        x + dx * dt
    }

    /// Compute output y = C x + D u
    pub fn output(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        &self.c * x + &self.d * u
    }

    /// Compute eigenvalues of A.
    pub fn poles(&self) -> Vec<num_complex::Complex64> {
    let _n = self.a.nrows();
        let _mat = self.a.clone();
        // Use power iteration fallback or direct eigenvalue computation
        // nalgebra doesn't have a built-in eigen solver for general real matrices,
        // so we implement a simple characteristic polynomial approach for small matrices
        // For now, compute via the companion-matrix approach
        compute_eigenvalues(&self.a)
    }

    /// Check if the system is stable (all eigenvalues have negative real parts).
    pub fn is_stable(&self) -> bool {
        self.poles().iter().all(|e| e.re < 0.0)
    }
}

impl StateSpaceDiscrete {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>, dt: f64) -> Self {
        assert_eq!(a.nrows(), a.ncols(), "A must be square");
        assert_eq!(a.nrows(), b.nrows(), "A rows must match B rows");
        assert_eq!(a.ncols(), c.ncols(), "A cols must match C cols");
        assert_eq!(b.ncols(), d.ncols(), "B cols must match D cols");
        assert_eq!(c.nrows(), d.nrows(), "C rows must match D rows");
        Self { a, b, c, d, dt }
    }

    pub fn states(&self) -> usize { self.a.nrows() }
    pub fn inputs(&self) -> usize { self.b.ncols() }
    pub fn outputs(&self) -> usize { self.c.nrows() }

    /// Advance state: x[k+1] = A x[k] + B u[k]
    pub fn step(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        &self.a * x + &self.b * u
    }

    /// Compute output y = C x + D u
    pub fn output(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        &self.c * x + &self.d * u
    }

    /// Compute eigenvalues of A.
    pub fn poles(&self) -> Vec<num_complex::Complex64> {
        compute_eigenvalues(&self.a)
    }

    /// Check if the discrete system is stable (all eigenvalues inside unit circle).
    pub fn is_stable(&self) -> bool {
        self.poles().iter().all(|e| e.norm() < 1.0)
    }
}

/// Compute eigenvalues of a real matrix using the QR algorithm.
pub fn compute_eigenvalues(a: &DMatrix<f64>) -> Vec<num_complex::Complex64> {
    let n = a.nrows();
    if n == 0 { return vec![]; }
    if n == 1 { return vec![num_complex::Complex64::new(a[(0, 0)], 0.0)]; }

    // QR algorithm with shifts to find eigenvalues
    let mut h = a.clone();
    // First, reduce to upper Hessenberg form
    hessenberg_reduce(&mut h);
    // Then apply QR iterations
    let tol = 1e-12;
    let max_iter = 1000;
    for _ in 0..max_iter {
        // Check for convergence of subdiagonal elements
        let mut converged = true;
        for i in 1..n {
            if h[(i, i - 1)].abs() > tol * (h[(i, i)].abs() + h[(i - 1, i - 1)].abs() + 1e-30) {
                converged = false;
                break;
            }
        }
        if converged { break; }

        // Wilkinson shift
        let shift = wilkinson_shift(&h);
        for i in 0..n {
            h[(i, i)] -= shift;
        }
        let q = h.clone().qr().q();
        let r = h.clone().qr().r();
        h = &r * &q;
        for i in 0..n {
            h[(i, i)] += shift;
        }
    }

    // Extract eigenvalues from quasi-upper triangular form
    let mut eigenvalues = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n && h[(i + 1, i)].abs() > 1e-10 {
            // 2x2 block
            let a11 = h[(i, i)];
            let a12 = h[(i, i + 1)];
            let a21 = h[(i + 1, i)];
            let a22 = h[(i + 1, i + 1)];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                eigenvalues.push(num_complex::Complex64::new((tr + disc.sqrt()) / 2.0, 0.0));
                eigenvalues.push(num_complex::Complex64::new((tr - disc.sqrt()) / 2.0, 0.0));
            } else {
                eigenvalues.push(num_complex::Complex64::new(tr / 2.0, disc.abs().sqrt() / 2.0));
                eigenvalues.push(num_complex::Complex64::new(tr / 2.0, -disc.abs().sqrt() / 2.0));
            }
            i += 2;
        } else {
            eigenvalues.push(num_complex::Complex64::new(h[(i, i)], 0.0));
            i += 1;
        }
    }
    eigenvalues
}

/// Reduce matrix to upper Hessenberg form in place.
fn hessenberg_reduce(h: &mut DMatrix<f64>) {
    let n = h.nrows();
    for k in 0..n.saturating_sub(2) {
        // Build Householder reflector for column k, rows k+1..n
        let m = n - k - 1;
        if m < 1 { continue; }
        let mut v = Vec::with_capacity(m);
        for i in (k + 1)..n {
            v.push(h[(i, k)]);
        }
        let norm_x = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_x < 1e-15 { continue; }
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_x;
        let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_v < 1e-15 { continue; }
        for x in v.iter_mut() { *x /= norm_v; }

        // Apply reflector: H = I - 2 v v^T
        // Left multiply: H[k+1:n, :] -= 2 v (v^T H[k+1:n, :])
        for j in 0..n {
            let dot: f64 = (0..m).map(|i| v[i] * h[(k + 1 + i, j)]).sum();
            for i in 0..m {
                h[(k + 1 + i, j)] -= 2.0 * v[i] * dot;
            }
        }
        // Right multiply: H[:, k+1:n] -= 2 (H[:, k+1:n] v) v^T
        for i in 0..n {
            let dot: f64 = (0..m).map(|jj| v[jj] * h[(i, k + 1 + jj)]).sum();
            for jj in 0..m {
                h[(i, k + 1 + jj)] -= 2.0 * dot * v[jj];
            }
        }
    }
}

fn wilkinson_shift(h: &DMatrix<f64>) -> f64 {
    let n = h.nrows();
    if n < 2 { return 0.0; }
    let a = h[(n - 2, n - 2)];
    let b = h[(n - 2, n - 1)];
    let c = h[(n - 1, n - 2)];
    let d = h[(n - 1, n - 1)];
    let tr = a + d;
    let det = a * d - b * c;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let e1 = tr / 2.0 + disc;
    let e2 = tr / 2.0 - disc;
    // Pick eigenvalue of 2x2 corner closest to h[n-1,n-1]
    if (e1 - d).abs() < (e2 - d).abs() { e1 } else { e2 }
}
