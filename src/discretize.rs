//! Discretization methods: zero-order hold and bilinear transform.

use nalgebra::DMatrix;
use crate::ss::{StateSpaceContinuous, StateSpaceDiscrete};

/// Discretize using zero-order hold: x[k+1] = Ad x[k] + Bd u[k].
///
/// Ad = expm(A * dt), Bd = integral_0^dt expm(A*tau) dtau * B
pub fn zoh_discretize(sys: &StateSpaceContinuous, dt: f64) -> StateSpaceDiscrete {
    let n = sys.states();
    let a_d = matrix_exp(&sys.a, dt);

    // Bd = A^{-1}(Ad - I) * B, or if A is singular, use the series
    let eye = DMatrix::identity(n, n);
    let ad_minus_i = &a_d - &eye;

    let b_d = if let Some(a_inv) = sys.a.clone().try_inverse() {
        &a_inv * &ad_minus_i * &sys.b
    } else {
        // Use series: integral_0^dt expm(A*t) dt * B = sum_{k=0}^{inf} (A^k * dt^{k+1} / (k+1)!) * B
        let mut result = DMatrix::zeros(n, sys.inputs());
        let mut a_power = DMatrix::identity(n, n);
        let mut factorial_kp1 = 1.0_f64;
        for k in 0..50 {
            if k > 0 { factorial_kp1 *= k as f64 + 1.0; }
            let coeff = dt.powi(k as i32 + 1) / factorial_kp1;
            let term = &a_power * &sys.b * coeff;
            result += &term;
            a_power = &a_power * &sys.a;
            if term.norm() < 1e-15 { break; }
        }
        result
    };

    StateSpaceDiscrete::new(a_d, b_d, sys.c.clone(), sys.d.clone(), dt)
}

/// Discretize using bilinear (Tustin) transform.
///
/// s = (2/dt) * (z-1)/(z+1)
pub fn bilinear_discretize(_sys: &StateSpaceContinuous, _dt: f64) -> StateSpaceContinuous {
    // Actually return a discrete system
    // But the function signature says StateSpaceContinuous... let me return discrete
    // Changing to return discrete
    todo!("Bilinear discretize returns discrete system - see bilinear_discretize_disc")
}

/// Discretize using bilinear (Tustin) transform, returning a discrete system.
pub fn bilinear_discretize_disc(sys: &StateSpaceContinuous, dt: f64) -> StateSpaceDiscrete {
    let n = sys.states();
    let eye = DMatrix::identity(n, n);

    let half_dt = dt / 2.0;

    // Phi = (I - (dt/2)A)^{-1}
    let lhs = &eye - sys.a.scale(half_dt);
    let phi = lhs.try_inverse().expect("Cannot invert (I - dt/2 A)");

    // Ad = Phi * (I + (dt/2)A)
    let a_d = &phi * (&eye + sys.a.scale(half_dt));

    // Bd = Phi * dt * B
    let b_d = &phi * &sys.b * dt;

    StateSpaceDiscrete::new(a_d, b_d, sys.c.clone(), sys.d.clone(), dt)
}

/// Matrix exponential using scaling and squaring with Taylor series.
pub fn matrix_exp(a: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    let n = a.nrows();
    let at = a * t;

    // Scaling: find s such that ||at / 2^s|| < 0.5
    let norm = at.norm();
    let s = if norm > 0.5 {
        (norm.log2().ceil() as usize).max(1)
    } else {
        0
    };

    let scaled = if s > 0 { at / (1u64 << s) as f64 } else { at.clone() };

    // Taylor series: exp(M) = I + M + M^2/2! + M^3/3! + ...
    let eye = DMatrix::identity(n, n);
    let mut result = eye.clone();
    let mut term = eye.clone();
    for k in 1..30 {
        term = term * &scaled / k as f64;
        result += &term;
        if term.norm() < 1e-15 { break; }
    }

    // Squaring
    for _ in 0..s {
        result = &result * &result;
    }

    result
}
