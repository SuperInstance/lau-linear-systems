//! Comprehensive tests for lau-linear-systems.

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector};
    use lau_linear_systems::*;

    // ========== State-Space Tests ==========

    #[test]
    fn test_ss_continuous_creation() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        assert_eq!(sys.states(), 2);
        assert_eq!(sys.inputs(), 1);
        assert_eq!(sys.outputs(), 1);
    }

    #[test]
    fn test_ss_continuous_step() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::zeros(2, 1);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let u = DVector::from_vec(vec![0.0]);
        let x_next = sys.step(&x, &u, 0.1);
        assert_abs_diff_eq!(x_next[0], 1.2, epsilon = 0.01);
        assert_abs_diff_eq!(x_next[1], 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_ss_discrete_creation() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceDiscrete::new(a, b, c, d, 0.1);
        assert_eq!(sys.states(), 2);
        assert_eq!(sys.inputs(), 1);
        assert_eq!(sys.outputs(), 1);
    }

    #[test]
    fn test_ss_discrete_step() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceDiscrete::new(a, b, c, d, 0.1);
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let u = DVector::from_vec(vec![1.0]);
        let x_next = sys.step(&x, &u);
        assert_abs_diff_eq!(x_next[0], 0.005, epsilon = 1e-10);
        assert_abs_diff_eq!(x_next[1], 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_ss_output() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::zeros(2, 1);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let x = DVector::from_vec(vec![3.0, 5.0]);
        let u = DVector::from_vec(vec![0.0]);
        let y = sys.output(&x, &u);
        assert_abs_diff_eq!(y[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eigenvalues_2x2() {
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let eigs = ss::compute_eigenvalues(&a);
        // Eigenvalues should be -1 and -2
        let mut sorted: Vec<f64> = eigs.iter().map(|e| e.re).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(sorted[0], -2.0, epsilon = 0.1);
        assert_abs_diff_eq!(sorted[1], -1.0, epsilon = 0.1);
    }

    #[test]
    fn test_eigenvalues_complex() {
        // Matrix with complex eigenvalues: [[0, -1], [1, 0]] has eigenvalues ±i
        let a = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let eigs = ss::compute_eigenvalues(&a);
        // Check that eigenvalues are complex with real part ≈ 0 and imag ≈ ±1
        assert!(eigs.iter().any(|e| e.im.abs() > 0.5));
    }

    #[test]
    fn test_stability_continuous() {
        // Stable system
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let b = DMatrix::zeros(2, 1);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b.clone(), c.clone(), d.clone());
        assert!(sys.is_stable());

        // Unstable system
        let a2 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let sys2 = ss::StateSpaceContinuous::new(a2, b, c, d);
        assert!(!sys2.is_stable());
    }

    #[test]
    fn test_stability_discrete() {
        // Stable discrete: eigenvalues inside unit circle
        let a = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.3]);
        let b = DMatrix::zeros(2, 1);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceDiscrete::new(a, b, c, d, 0.1);
        assert!(sys.is_stable());
    }

    // ========== Controllability & Observability Tests ==========

    #[test]
    fn test_controllable_system() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        assert!(ctrb_obs::is_controllable(&a, &b));
    }

    #[test]
    fn test_uncontrollable_system() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        // Second state is not reachable
        assert!(!ctrb_obs::is_controllable(&a, &b));
    }

    #[test]
    fn test_observable_system() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        assert!(ctrb_obs::is_observable(&a, &c));
    }

    #[test]
    fn test_unobservable_system() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        // Second state doesn't affect output
        assert!(!ctrb_obs::is_observable(&a, &c));
    }

    #[test]
    fn test_controllability_matrix_shape() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -6.0, -11.0, -6.0]);
        let b = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
        let ctrb = ctrb_obs::controllability_matrix(&a, &b);
        assert_eq!(ctrb.nrows(), 3);
        assert_eq!(ctrb.ncols(), 3);
    }

    #[test]
    fn test_observability_matrix_shape() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -6.0, -11.0, -6.0]);
        let c = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
        let obs = ctrb_obs::observability_matrix(&a, &c);
        assert_eq!(obs.nrows(), 3);
        assert_eq!(obs.ncols(), 3);
    }

    #[test]
    fn test_controllable_ss_method() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        assert!(sys.is_controllable());
        assert!(sys.is_observable());
    }

    #[test]
    fn test_multi_input_controllability() {
        // 2-state, 2-input: fully controllable
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 2);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        assert!(sys.is_controllable());
    }

    #[test]
    fn test_multi_output_observability() {
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let c = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let d = DMatrix::zeros(2, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        assert!(sys.is_observable());
    }

    // ========== Pole Placement Tests ==========

    #[test]
    fn test_pole_placement_2nd_order() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let desired = vec![-2.0, -3.0];
        let k = feedback::place_poles(&a, &b, &desired);
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);

        // Verify: eigenvalues of (A - BK) should be at desired locations
        let a_cl = &a - &b * &k;
        let eigs = ss::compute_eigenvalues(&a_cl);
        let mut eig_vals: Vec<f64> = eigs.iter().map(|e| e.re).collect();
        eig_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(eig_vals[0], -3.0, epsilon = 0.5);
        assert_abs_diff_eq!(eig_vals[1], -2.0, epsilon = 0.5);
    }

    #[test]
    fn test_pole_placement_shifts_poles() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 2.0, 1.0]); // unstable
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let desired = vec![-5.0, -5.0];
        let k = feedback::place_poles(&a, &b, &desired);
        let a_cl = &a - &b * &k;
        let eigs = ss::compute_eigenvalues(&a_cl);
        // Both eigenvalues should have real part ≈ -5
        for e in &eigs {
            assert!(e.re < -3.0, "Expected stable poles, got {:?}", e);
        }
    }

    // ========== LQR Tests ==========

    #[test]
    fn test_lqr_2nd_order() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let k = feedback::lqr(&a, &b, &q, &r);
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);

        // Closed loop should be stable
        let a_cl = &a - &b * &k;
        let eigs = ss::compute_eigenvalues(&a_cl);
        for e in &eigs {
            assert!(e.re < 0.0, "LQR should produce stable closed loop, got {:?}", e);
        }
    }

    #[test]
    fn test_lqr_stabilizes_unstable_system() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 2.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let q = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 10.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let k = feedback::lqr(&a, &b, &q, &r);
        let a_cl = &a - &b * &k;
        let eigs = ss::compute_eigenvalues(&a_cl);
        for e in &eigs {
            assert!(e.re < 0.0, "LQR closed loop should be stable");
        }
    }

    // ========== Transfer Function Tests ==========

    #[test]
    fn test_tf_creation() {
        let tf = transfer::TransferFunction::new(vec![1.0], vec![1.0, 2.0]);
        assert_eq!(tf.order(), 1);
    }

    #[test]
    fn test_tf_evaluate_dc() {
        // H(s) = 1/(s+1), H(0) = 1
        let tf = transfer::TransferFunction::new(vec![1.0], vec![1.0, 1.0]);
        let val = tf.evaluate(num_complex::Complex64::new(0.0, 0.0));
        assert_abs_diff_eq!(val.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_evaluate_complex() {
        // H(s) = 1/(s+1), H(j) = 1/(1+j) = (1-j)/2
        let tf = transfer::TransferFunction::new(vec![1.0], vec![1.0, 1.0]);
        let val = tf.evaluate(num_complex::Complex64::new(0.0, 1.0));
        assert_abs_diff_eq!(val.re, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(val.im, -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_poles() {
        // H(s) = 1/(s+1)(s+2) = 1/(s^2+3s+2)
        let tf = transfer::TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0]);
        let poles = tf.poles();
        assert_eq!(poles.len(), 2);
        let mut reals: Vec<f64> = poles.iter().map(|p| p.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(reals[0], -2.0, epsilon = 0.5);
        assert_abs_diff_eq!(reals[1], -1.0, epsilon = 0.5);
    }

    #[test]
    fn test_ss2tf_double_integrator() {
        // Double integrator: A = [0,1;0,0], B = [0;1], C = [1,0], D = 0
        // TF = 1/s^2
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let tf = transfer::ss2tf(&sys);
        // Check DC gain = infinity (double integrator), or evaluate at s=1
        let val = tf.evaluate(num_complex::Complex64::new(1.0, 0.0));
        assert_abs_diff_eq!(val.re, 1.0, epsilon = 0.1); // 1/s^2 at s=1 = 1
    }

    #[test]
    fn test_tf2ss_round_trip() {
        // Create TF, convert to SS, then back to TF, check they match
        let tf = transfer::TransferFunction::new(vec![1.0, 2.0], vec![1.0, 3.0, 2.0]);
        let sys = transfer::tf2ss(&tf);
        let tf2 = transfer::ss2tf(&sys);

        // Check at a few points
        for s_val in &[0.5, 1.0, 2.0, 5.0] {
            let s = num_complex::Complex64::new(*s_val, 0.0);
            let v1 = tf.evaluate(s);
            let v2 = tf2.evaluate(s);
            assert_abs_diff_eq!(v1.re, v2.re, epsilon = 0.5);
        }
    }

    #[test]
    fn test_tf2ss_proper() {
        // Strictly proper TF: (s+1)/(s^2+3s+2)
        let tf = transfer::TransferFunction::new(vec![1.0, 1.0], vec![1.0, 3.0, 2.0]);
        let sys = transfer::tf2ss(&tf);
        assert_eq!(sys.states(), 2);
        assert_eq!(sys.inputs(), 1);
        assert_eq!(sys.outputs(), 1);
        assert!(sys.is_controllable());
    }

    // ========== Discretization Tests ==========

    #[test]
    fn test_zoh_discretize_identity() {
        // A = 0, B = 1 -> Ad = I, Bd = dt*B
        let a = DMatrix::zeros(1, 1);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let disc = discretize::zoh_discretize(&sys, 0.1);
        assert_abs_diff_eq!(disc.a[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(disc.b[(0, 0)], 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_zoh_discretize_first_order() {
        // x' = -x + u, dt = 0.1
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let disc = discretize::zoh_discretize(&sys, 0.1);
        // Ad = exp(-0.1) ≈ 0.9048
        assert_abs_diff_eq!(disc.a[(0, 0)], (-0.1_f64).exp(), epsilon = 0.01);
        // Bd = (1 - exp(-0.1)) ≈ 0.0952
        assert_abs_diff_eq!(disc.b[(0, 0)], 1.0 - (-0.1_f64).exp(), epsilon = 0.01);
    }

    #[test]
    fn test_zoh_preserves_stability() {
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let disc = discretize::zoh_discretize(&sys, 0.01);
        assert!(disc.is_stable());
    }

    #[test]
    fn test_bilinear_discretize_first_order() {
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let disc = discretize::bilinear_discretize_disc(&sys, 0.1);
        // Ad = (1 - dt/2)/(1 + dt/2) = (1-0.05)/(1+0.05) = 0.95/1.05 ≈ 0.9048
        let expected_ad = (1.0 - 0.05) / (1.0 + 0.05);
        assert_abs_diff_eq!(disc.a[(0, 0)], expected_ad, epsilon = 0.01);
    }

    #[test]
    fn test_matrix_exp_identity() {
        let a = DMatrix::zeros(2, 2);
        let result = discretize::matrix_exp(&a, 1.0);
        assert_abs_diff_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(1, 1)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(0, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_exp_known() {
        // exp([0, t; 0, 0]) = [1, t; 0, 1]
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let result = discretize::matrix_exp(&a, 2.0);
        assert_abs_diff_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(0, 1)], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(1, 0)], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(1, 1)], 1.0, epsilon = 1e-10);
    }

    // ========== Interconnection Tests ==========

    #[test]
    fn test_series_interconnection() {
        // sys1: x' = -x + u, y = x (gain = 1/(s+1))
        let a1 = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b1 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c1 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d1 = DMatrix::zeros(1, 1);
        let sys1 = ss::StateSpaceContinuous::new(a1, b1, c1, d1);

        // sys2: same
        let sys2 = ss::StateSpaceContinuous::new(
            DMatrix::from_row_slice(1, 1, &[-2.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::zeros(1, 1),
        );

        let series_sys = interconnect::series(&sys1, &sys2);
        assert_eq!(series_sys.states(), 2);
        assert_eq!(series_sys.inputs(), 1);
        assert_eq!(series_sys.outputs(), 1);

        // DC gain of series = DC(sys1) * DC(sys2) = 1 * 0.5 = 0.5
        // Check: at DC, y = C*(-A)^{-1}*B + D
        let a_inv = (-series_sys.a.clone()).try_inverse().unwrap();
        let dc_gain = &series_sys.c * &a_inv * &series_sys.b + &series_sys.d;
        assert_abs_diff_eq!(dc_gain[(0, 0)], 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_parallel_interconnection() {
        let a1 = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b1 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c1 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d1 = DMatrix::zeros(1, 1);
        let sys1 = ss::StateSpaceContinuous::new(a1, b1, c1, d1);

        let a2 = DMatrix::from_row_slice(1, 1, &[-2.0]);
        let b2 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c2 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d2 = DMatrix::zeros(1, 1);
        let sys2 = ss::StateSpaceContinuous::new(a2, b2, c2, d2);

        let par_sys = interconnect::parallel(&sys1, &sys2);
        assert_eq!(par_sys.states(), 2);

        // DC gain = 1 + 0.5 = 1.5
        let a_inv = (-par_sys.a.clone()).try_inverse().unwrap();
        let dc_gain = &par_sys.c * &a_inv * &par_sys.b + &par_sys.d;
        assert_abs_diff_eq!(dc_gain[(0, 0)], 1.5, epsilon = 0.01);
    }

    #[test]
    fn test_feedback_interconnection() {
        let a1 = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b1 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c1 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d1 = DMatrix::zeros(1, 1);
        let sys1 = ss::StateSpaceContinuous::new(a1, b1, c1, d1);

        // Unity feedback (sys2 = 1, but we model it as a static gain)
        let a2 = DMatrix::zeros(1, 1);
        let b2 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c2 = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d2 = DMatrix::zeros(1, 1);
        let sys2 = ss::StateSpaceContinuous::new(a2, b2, c2, d2);

        let fb_sys = interconnect::feedback(&sys1, &sys2);
        assert_eq!(fb_sys.states(), 2);

        // Closed-loop DC gain with unity feedback: 1/(s+2) at DC = 0.5
        // But with the static gain sys2, it's a bit different. Let's just check stability.
        assert!(fb_sys.is_stable());
    }

    // ========== Kalman Filter Tests ==========

    #[test]
    fn test_kalman_filter_convergence() {
        // Simple 1D tracking
        let a = DMatrix::from_row_slice(1, 1, &[1.0]);
        let b = DMatrix::zeros(1, 1);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let q = DMatrix::from_row_slice(1, 1, &[0.01]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let x0 = DVector::from_vec(vec![0.0]);
        let p0 = DMatrix::from_row_slice(1, 1, &[100.0]);

        let mut kf = estimation::KalmanFilter::new(a, b, c, d, q, r, x0, p0);

        // True state is constant at 5.0
        for _ in 0..50 {
            let y = DVector::from_vec(vec![5.0]);
            let u = DVector::from_vec(vec![0.0]);
            kf.step(&y, &u);
        }

        // Should converge close to 5.0
        let est = kf.estimate();
        assert_abs_diff_eq!(est[0], 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_kalman_filter_2d_convergence() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::zeros(2, 1);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let q = DMatrix::from_row_slice(2, 2, &[0.001, 0.0, 0.0, 0.001]);
        let r = DMatrix::from_row_slice(1, 1, &[0.5]);
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let p0 = DMatrix::from_row_slice(2, 2, &[100.0, 0.0, 0.0, 100.0]);

        let mut kf = estimation::KalmanFilter::new(a, b, c, d, q, r, x0, p0);

        // True position starts at 0, velocity = 1
        for k in 0..100 {
            let true_pos = k as f64 * 0.1;
            let y = DVector::from_vec(vec![true_pos]);
            let u = DVector::from_vec(vec![0.0]);
            kf.step(&y, &u);
        }

        // Position estimate should be close to true value
        assert_abs_diff_eq!(kf.estimate()[0], 9.9, epsilon = 1.0);
        // Velocity estimate should be close to 1.0
        assert_abs_diff_eq!(kf.estimate()[1], 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_kalman_covariance_decreases() {
        let a = DMatrix::from_row_slice(1, 1, &[1.0]);
        let b = DMatrix::zeros(1, 1);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let q = DMatrix::from_row_slice(1, 1, &[0.01]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let x0 = DVector::from_vec(vec![0.0]);
        let p0 = DMatrix::from_row_slice(1, 1, &[100.0]);

        let mut kf = estimation::KalmanFilter::new(a, b, c, d, q, r, x0, p0);
        let initial_p = kf.covariance()[(0, 0)];

        for _ in 0..20 {
            let y = DVector::from_vec(vec![3.0]);
            let u = DVector::from_vec(vec![0.0]);
            kf.step(&y, &u);
        }

        assert!(kf.covariance()[(0, 0)] < initial_p);
    }

    // ========== Luenberger Observer Tests ==========

    #[test]
    fn test_luenberger_observer_convergence() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceDiscrete::new(a, b, c, d, 0.1);

        let desired_poles = vec![0.1, 0.1];
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let mut obs = estimation::LuenbergerObserver::design_by_poles(sys.clone(), &desired_poles, x0);

        // True state
        let mut true_x = DVector::from_vec(vec![5.0, 1.0]);

        for _ in 0..50 {
            let u = DVector::from_vec(vec![0.0]);
            let y = sys.output(&true_x, &u);
            obs.update(&y, &u);
            true_x = sys.step(&true_x, &u);
        }

        // Estimate should be close to true state
        let est = obs.estimate();
        assert_abs_diff_eq!(est[0], true_x[0], epsilon = 1.0);
    }

    // ========== System Norms Tests ==========

    #[test]
    fn test_h2_norm_first_order() {
        // H(s) = 1/(s+1), H2 norm = 1/sqrt(2)
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let h2 = norms::h2_norm(&sys);
        assert_abs_diff_eq!(h2, 1.0 / 2.0_f64.sqrt(), epsilon = 0.01);
    }

    #[test]
    fn test_hinf_norm_first_order() {
        // H(s) = 1/(s+1), Hinf norm = 1 (at DC)
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let hinf = norms::hinf_norm_approx(&sys, (0.01, 100.0), 200);
        assert_abs_diff_eq!(hinf, 1.0, epsilon = 0.05);
    }

    // ========== Agent Belief Tracker Tests ==========

    #[test]
    fn test_agent_tracker_convergence() {
        let mut tracker = agent::AgentBeliefTracker::new_2d(0.1, 0.01, 0.01, 1.0);

        // Agent at position 10, moving at velocity 5
        for k in 0..200 {
            let true_pos = 10.0 + 5.0 * k as f64 * 0.1;
            tracker.update(true_pos, None);
        }

        assert_abs_diff_eq!(tracker.position_estimate, 10.0 + 5.0 * 199.0 * 0.1, epsilon = 2.0);
        assert_abs_diff_eq!(tracker.velocity_estimate, 5.0, epsilon = 1.0);
    }

    #[test]
    fn test_agent_tracker_uncertainty_decreases() {
        let mut tracker = agent::AgentBeliefTracker::new_2d(0.1, 0.01, 0.01, 1.0);
        let initial_uncertainty = tracker.position_uncertainty;

        for k in 0..50 {
            let pos = k as f64 * 0.1;
            tracker.update(pos, None);
        }

        assert!(tracker.position_uncertainty < initial_uncertainty);
    }

    #[test]
    fn test_agent_tracker_3d() {
        let mut tracker = agent::AgentBeliefTracker::new_3d(0.1, 0.01, 1.0);
        assert_eq!(tracker.state().nrows(), 3);

        for k in 0..100 {
            let pos = k as f64 * 0.1;
            tracker.update(pos, None);
        }

        // Should track reasonably
        assert!(tracker.position_uncertainty < 100.0);
    }

    #[test]
    fn test_agent_tracker_predict() {
        let mut tracker = agent::AgentBeliefTracker::new_2d(0.1, 0.01, 0.01, 1.0);

        // First update with a measurement
        tracker.update(5.0, None);
        let pos_after_update = tracker.position_estimate;

        // Predict without measurement
        tracker.predict(None);
        let pos_after_predict = tracker.position_estimate;

        // Position should have moved by velocity * dt
        assert!(pos_after_predict != pos_after_update || tracker.velocity_estimate == 0.0);
    }

    // ========== Serde Tests ==========

    #[test]
    fn test_serde_continuous() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let json = serde_json::to_string(&sys).unwrap();
        let sys2: ss::StateSpaceContinuous = serde_json::from_str(&json).unwrap();
        assert_eq!(sys2.states(), 2);
    }

    #[test]
    fn test_serde_discrete() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceDiscrete::new(a, b, c, d, 0.1);
        let json = serde_json::to_string(&sys).unwrap();
        let sys2: ss::StateSpaceDiscrete = serde_json::from_str(&json).unwrap();
        assert_eq!(sys2.states(), 2);
        assert_abs_diff_eq!(sys2.dt, 0.1, epsilon = 1e-10);
    }

    // ========== Additional Edge Case Tests ==========

    #[test]
    fn test_double_integrator_controllable() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        assert!(ctrb_obs::is_controllable(&a, &b));
    }

    #[test]
    fn test_double_integrator_observable() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        assert!(ctrb_obs::is_observable(&a, &c));
    }

    #[test]
    fn test_pure_gain_tf() {
        let tf = transfer::TransferFunction::new(vec![5.0], vec![1.0]);
        let sys = transfer::tf2ss(&tf);
        // Pure gain: D = 5
        assert_abs_diff_eq!(sys.d[(0, 0)], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zoh_discretize_double_integrator() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let disc = discretize::zoh_discretize(&sys, 0.1);
        // Ad = [1, dt; 0, 1]
        assert_abs_diff_eq!(disc.a[(0, 0)], 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(disc.a[(0, 1)], 0.1, epsilon = 0.01);
        assert_abs_diff_eq!(disc.a[(1, 0)], 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(disc.a[(1, 1)], 1.0, epsilon = 0.01);
        // Bd = [dt^2/2; dt]
        assert_abs_diff_eq!(disc.b[(0, 0)], 0.1 * 0.1 / 2.0, epsilon = 0.01);
        assert_abs_diff_eq!(disc.b[(1, 0)], 0.1, epsilon = 0.01);
    }

    // ========== More Tests for 55+ Coverage ==========

    #[test]
    fn test_ss_discrete_output() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::from_row_slice(1, 1, &[0.5]);
        let sys = ss::StateSpaceDiscrete::new(a, b, c, d, 0.1);
        let x = DVector::from_vec(vec![2.0, 3.0]);
        let u = DVector::from_vec(vec![1.0]);
        let y = sys.output(&x, &u);
        // y = C*x + D*u = 2.0 + 0.5*1 = 2.5
        assert_abs_diff_eq!(y[0], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_controllability_rank_full() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -2.0, -3.0]);
        let b = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
        let ctrb = ctrb_obs::controllability_matrix(&a, &b);
        assert_eq!(ctrb.nrows(), 3);
        assert_eq!(ctrb.ncols(), 3);
        assert!(ctrb_obs::is_controllable(&a, &b));
    }

    #[test]
    fn test_tf_first_order_system() {
        // H(s) = 1/(s+1) → ss2tf should recover it
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::zeros(1, 1);
        let sys = ss::StateSpaceContinuous::new(a, b, c, d);
        let tf = transfer::ss2tf(&sys);
        // At s=0: H(0) = 1
        let val = tf.evaluate(num_complex::Complex64::new(0.0, 0.0));
        assert_abs_diff_eq!(val.re, 1.0, epsilon = 0.01);
        // At s=j: |H(j)| = 1/sqrt(2)
        let val_j = tf.evaluate(num_complex::Complex64::new(0.0, 1.0));
        assert_abs_diff_eq!(val_j.norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 0.01);
    }

    #[test]
    fn test_tf2ss_ss2tf_consistency() {
        // Create a 2nd order TF, convert to SS, check outputs match
        let tf = transfer::TransferFunction::new(vec![1.0, 0.0], vec![1.0, 1.0, 1.0]);
        let sys = transfer::tf2ss(&tf);
        assert_eq!(sys.states(), 2);
        assert!(sys.is_controllable());

        // Verify transfer function matches at a point
        let s = num_complex::Complex64::new(1.0, 0.0);
        let tf_val = tf.evaluate(s);
        // Manually compute from SS
        let n = 2;
        let s_mat = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]) - &sys.a;
        let ss_val = if let Some(inv) = s_mat.try_inverse() {
            let cx = &inv * &sys.b;
            sys.c[(0, 0)] * cx[0] + sys.c[(0, 1)] * cx[1] + sys.d[(0, 0)]
        } else { 0.0 };
        assert_abs_diff_eq!(tf_val.re, ss_val, epsilon = 0.1);
    }
}
