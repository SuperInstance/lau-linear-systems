//! State feedback: pole placement and LQR.

use nalgebra::{DMatrix, DVector};

/// Pole placement for single-input systems using Ackermann's formula.
///
/// Places poles of (A - BK) at desired locations.
/// K has shape (1 x n) for single-input systems.
/// desired_poles: eigenvalues of (A - BK) — must be real or come in conjugate pairs.
pub fn place_poles(a: &DMatrix<f64>, b: &DMatrix<f64>, desired_poles: &[f64]) -> DMatrix<f64> {
    pole_placement_ackermann(a, b, desired_poles)
}

/// Pole placement for single-input systems using Ackermann's formula.
fn pole_placement_ackermann(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    desired_poles: &[f64],
) -> DMatrix<f64> {
    let n = a.nrows();
    assert_eq!(b.ncols(), 1, "Ackermann's formula requires single-input system");
    assert_eq!(desired_poles.len(), n, "Must specify exactly n poles");

    let ctrb = crate::ctrb_obs::controllability_matrix(a, b);
    let ctrb_inv = ctrb.clone().try_inverse().expect("System not controllable");

    let mut coeffs = vec![0.0_f64; n + 1];
    coeffs[0] = 1.0;
    for &pole in desired_poles {
        for j in (1..=n).rev() {
            coeffs[j] -= pole * coeffs[j - 1];
        }
    }

    let mut powers = Vec::with_capacity(n + 1);
    powers.push(DMatrix::identity(n, n));
    for i in 1..=n {
        powers.push(&powers[i - 1] * a);
    }

    let mut phi_d = DMatrix::zeros(n, n);
    for k in 0..=n {
        let power_idx = n - k;
        phi_d += coeffs[k] * &powers[power_idx];
    }

    let mut e_n = DMatrix::zeros(1, n);
    e_n[(0, n - 1)] = 1.0;

    &e_n * &ctrb_inv * &phi_d
}

/// Linear Quadratic Regulator (LQR) for continuous-time systems.
///
/// Minimizes: J = integral (x'Qx + u'Ru) dt
/// Returns gain K such that u = -Kx.
///
/// Uses Kleinman's iteration (Newton's method on the CARE).
pub fn lqr(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> DMatrix<f64> {
    let n = a.nrows();
    let m = b.ncols();

    assert_eq!(q.nrows(), n);
    assert_eq!(q.ncols(), n);
    assert_eq!(r.nrows(), m);
    assert_eq!(r.ncols(), m);

    let r_inv = r.clone().try_inverse().expect("R must be invertible");

    // Start with a stabilizing gain via pole placement (single-input case)
    let stable_poles: Vec<f64> = (0..n).map(|i| -1.0 - i as f64).collect();
    let k0 = if m == 1 {
        place_poles(a, b, &stable_poles)
    } else {
        // For multi-input, use a simple initial gain
        DMatrix::zeros(m, n)
    };

    // Kleinman iteration: solve Lyapunov equations
    let mut k = k0;
    for _ in 0..100 {
        let a_cl = a - b * &k;
        // Lyapunov: A_cl' P + P A_cl = -(Q + K'R K)
        let rhs = -(q + &k.transpose() * r * &k);

        match solve_lyapunov(&a_cl.transpose(), &rhs) {
            Some(p) => {
                let k_new = &r_inv * b.transpose() * &p;
                let diff = (&k_new - &k).norm() / (k.norm().max(1e-15));
                k = k_new;
                if diff < 1e-10 {
                    break;
                }
            }
            None => break,
        }
    }

    k
}

/// Solve continuous Lyapunov equation: A'P + PA = Q via vectorization.
fn solve_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let n2 = n * n;
    let mut lhs = DMatrix::zeros(n2, n2);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let row = i * n + k;
                    let col = j * n + l;
                    if i == j {
                        lhs[(row, col)] += a[(k, l)];
                    }
                    if k == l {
                        lhs[(row, col)] += a[(i, j)];
                    }
                }
            }
        }
    }

    let mut rhs = DVector::zeros(n2);
    for i in 0..n {
        for j in 0..n {
            rhs[i * n + j] = q[(i, j)];
        }
    }

    let sol = lhs.lu().solve(&rhs)?;
    let mut result = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = sol[i * n + j];
        }
    }
    Some(result)
}
