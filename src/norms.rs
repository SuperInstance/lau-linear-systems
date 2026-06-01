//! System norms: H2 and H-infinity approximations.

use nalgebra::DMatrix;
use crate::ss::StateSpaceContinuous;

/// Compute H2 norm of a stable continuous-time system.
///
/// ||G||_2 = sqrt(trace(C * P * C')) where A*P + P*A' = -B*B'
pub fn h2_norm(sys: &StateSpaceContinuous) -> f64 {
    let _n = sys.states();
    
    // Solve the Lyapunov equation: A*P + P*A' = -B*B'
    let bb_t = &sys.b * sys.b.transpose();
    let neg_bb_t = -&bb_t;

    // Use the same vectorization approach as in feedback.rs
    let p = solve_controllability_lyapunov(&sys.a, &neg_bb_t);

    let cpct = &sys.c * &p * sys.c.transpose();
    cpct.trace().max(0.0).sqrt()
}

/// Solve continuous Lyapunov: A*P + P*A' = Q
fn solve_controllability_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> DMatrix<f64> {
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

    let mut rhs = nalgebra::DVector::zeros(n2);
    for i in 0..n {
        for j in 0..n {
            rhs[i * n + j] = q[(i, j)];
        }
    }

    let sol = lhs.lu().solve(&rhs).expect("Lyapunov equation solve failed");
    let mut result = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = sol[i * n + j];
        }
    }
    result
}

/// Approximate H-infinity norm using frequency sweep.
///
/// ||G||_∞ ≈ max_ω |G(jω)| for SISO systems.
pub fn hinf_norm_approx(sys: &StateSpaceContinuous, freq_range: (f64, f64), n_points: usize) -> f64 {
    let mut max_sv = 0.0_f64;
    let log_min = freq_range.0.ln();
    let log_max = freq_range.1.ln();

    let n = sys.states();

    for i in 0..n_points {
        let omega = ((log_min + (log_max - log_min) * i as f64 / (n_points - 1).max(1) as f64)).exp();
        
        // G(jω) = C(jωI - A)^{-1}B + D
        // For SISO, evaluate the transfer function at s = jω
        // We compute (jωI - A)^{-1} using real arithmetic:
        // Separate into real and imaginary parts
        
        // For SISO: G(jω) = C * inv(jωI - A) * B + D
        // (jωI - A) * X = B => solve for X, then G = C*X + D
        // The matrix (jωI - A) is complex. We handle it by doubling the system:
        // Real(A) -Im(I)*ω | Real(B)   = | Real(X)
        // Im(I)*ω  Real(A)  | Im(B)      | Im(X)
        
        let mut mat = DMatrix::zeros(2 * n, 2 * n);
        // Top-left: Real(jωI - A) = -A
        for r in 0..n {
            for c in 0..n {
                mat[(r, c)] = -sys.a[(r, c)];
            }
        }
        // Top-right: Im(jωI - A) = -ωI... wait
        // (jωI - A) = -A + jωI
        // Real part = -A, Imaginary part = ωI
        // No: Real part = -A, but diagonal has +0, and imaginary part on diagonal = ω
        // Actually: jωI - A has real part = -A (real part of A), imaginary diagonal = ω
        
        // Hmm, A is real, so jωI - A = -A + jωI
        // Real part of (jωI - A) = -A
        // Imaginary part of (jωI - A) = ωI (diagonal only)
        
        // So the doubled system for solving (jωI - A)x = b where x = xr + j*xi, b = br + j*bi:
        // -A*xr - ω*xi = br
        // ω*xr - A*xi = bi
        // Matrix: [-A, -ωI; ωI, -A]

        // Top-left: -A
        for r in 0..n {
            for c in 0..n {
                mat[(r, c)] = -sys.a[(r, c)];
            }
        }
        // Top-right: -ωI
        for r in 0..n {
            mat[(r, n + r)] = -omega;
        }
        // Bottom-left: ωI
        for r in 0..n {
            mat[(n + r, r)] = omega;
        }
        // Bottom-right: -A
        for r in 0..n {
            for c in 0..n {
                mat[(n + r, n + c)] = -sys.a[(r, c)];
            }
        }

        // RHS: B is real, so br = B, bi = 0
        let m = sys.inputs();
        let mut rhs = nalgebra::DVector::zeros(2 * n);
        for r in 0..n {
            for c in 0..m {
                if c == 0 { rhs[r] = sys.b[(r, c)]; }
            }
        }

        if let Some(sol) = mat.lu().solve(&rhs) {
            // Extract real and imaginary parts of x
            let xr: nalgebra::DVector<f64> = sol.rows(0, n).into();
            let xi: nalgebra::DVector<f64> = sol.rows(n, n).into();

            // G(jω) = C*x + D = C*(xr + j*xi) + D = (C*xr + D) + j*(C*xi)
            let gr = &sys.c * &xr + &sys.d.column(0);
            let gi = &sys.c * &xi;

            // |G(jω)| for SISO
            let mag = (gr[0] * gr[0] + gi[0] * gi[0]).sqrt();
            max_sv = max_sv.max(mag);
        }
    }

    max_sv
}
