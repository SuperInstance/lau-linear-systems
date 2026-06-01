//! State estimation: Luenberger observer and Kalman filter.

use nalgebra::{DMatrix, DVector};
use crate::ss::StateSpaceDiscrete;

/// Luenberger observer for a discrete-time system.
///
/// Estimates state via: x̂[k+1] = A x̂[k] + B u[k] + L (y[k] - C x̂[k])
#[derive(Debug, Clone)]
pub struct LuenbergerObserver {
    pub sys: StateSpaceDiscrete,
    pub l: DMatrix<f64>, // Observer gain (outputs x states)
    pub x_hat: DVector<f64>,
}

impl LuenbergerObserver {
    /// Create observer with given gain L and initial state estimate.
    pub fn new(sys: StateSpaceDiscrete, l: DMatrix<f64>, x0: DVector<f64>) -> Self {
        assert_eq!(l.ncols(), sys.outputs(), "L cols must match output count");
        assert_eq!(l.nrows(), sys.states(), "L rows must match state count");
        Self { sys, l, x_hat: x0 }
    }

    /// Design observer by pole placement: place (A-LC) eigenvalues at desired_poles.
    pub fn design_by_poles(
        sys: StateSpaceDiscrete,
        desired_poles: &[f64],
        x0: DVector<f64>,
    ) -> Self {
        // For observer design, we place poles of (A - LC).
        // This is equivalent to placing poles of (A^T - C^T L^T).
        // Use pole placement on (A^T, C^T) to get K, then L = K^T.
        let at = sys.a.transpose();
        let ct = sys.c.transpose();
        let k = crate::feedback::place_poles(&at, &ct, desired_poles);
        let l = k.transpose();
        Self::new(sys, l, x0)
    }

    /// Update estimate with measurement and input.
    pub fn update(&mut self, y: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        let y_hat = &self.sys.c * &self.x_hat + &self.sys.d * u;
        let innovation = y - &y_hat;
        self.x_hat = &self.sys.a * &self.x_hat + &self.sys.b * u + &self.l * &innovation;
        self.x_hat.clone()
    }

    /// Get current state estimate.
    pub fn estimate(&self) -> &DVector<f64> {
        &self.x_hat
    }
}

/// Discrete-time Kalman filter.
///
/// State model:  x[k+1] = A x[k] + B u[k] + w,  w ~ N(0, Q)
/// Measurement: y[k]   = C x[k] + D u[k] + v,   v ~ N(0, R)
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub q: DMatrix<f64>, // Process noise covariance (states x states)
    pub r: DMatrix<f64>, // Measurement noise covariance (outputs x outputs)
    pub x_hat: DVector<f64>,
    pub p: DMatrix<f64>, // Estimate covariance
}

impl KalmanFilter {
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        d: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        x0: DVector<f64>,
        p0: DMatrix<f64>,
    ) -> Self {
        let n = a.nrows();
        assert_eq!(q.nrows(), n);
        assert_eq!(q.ncols(), n);
        assert_eq!(r.nrows(), c.nrows());
        assert_eq!(r.ncols(), c.nrows());
        Self { a, b, c, d, q, r, x_hat: x0, p: p0 }
    }

    /// Predict step: propagate state and covariance forward.
    pub fn predict(&mut self, u: &DVector<f64>) {
        self.x_hat = &self.a * &self.x_hat + &self.b * u;
        self.p = &self.a * &self.p * &self.a.transpose() + &self.q;
    }

    /// Update step: incorporate measurement.
    pub fn update(&mut self, y: &DVector<f64>, u: &DVector<f64>) {
        let y_hat = &self.c * &self.x_hat + &self.d * u;
        let innovation = y - &y_hat;
        let s = &self.c * &self.p * &self.c.transpose() + &self.r;
        let s_inv = s.clone().try_inverse().expect("Innovation covariance is singular");
        let k = &self.p * &self.c.transpose() * &s_inv;
        self.x_hat = &self.x_hat + &k * &innovation;
        let n = self.p.nrows();
        let i = DMatrix::identity(n, n);
        let kh = &k * &self.c;
        self.p = (&i - &kh) * &self.p;
    }

    /// Full predict + update step.
    pub fn step(&mut self, y: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        self.predict(u);
        self.update(y, u);
        self.x_hat.clone()
    }

    /// Get current state estimate.
    pub fn estimate(&self) -> &DVector<f64> {
        &self.x_hat
    }

    /// Get current estimate covariance.
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.p
    }
}
