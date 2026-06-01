//! System interconnection: series, parallel, and feedback.

use nalgebra::DMatrix;
use crate::ss::StateSpaceContinuous;

/// Series interconnection: sys1 -> sys2 (output of sys1 feeds into sys2).
pub fn series(sys1: &StateSpaceContinuous, sys2: &StateSpaceContinuous) -> StateSpaceContinuous {
    assert_eq!(sys1.outputs(), sys2.inputs(), "sys1 outputs must match sys2 inputs");

    let n1 = sys1.states();
    let n2 = sys2.states();
    let n = n1 + n2;
    let m = sys1.inputs();
    let p = sys2.outputs();

    // A = [A1, 0; B2*C1, A2]
    let mut a = DMatrix::zeros(n, n);
    a.view_mut((0, 0), (n1, n1)).copy_from(&sys1.a);
    a.view_mut((n1, n1), (n2, n2)).copy_from(&sys2.a);
    // B2*C1
    let b2c1 = &sys2.b * &sys1.c;
    a.view_mut((n1, 0), (n2, n1)).copy_from(&b2c1);

    // B = [B1; B2*D1]
    let mut b = DMatrix::zeros(n, m);
    b.view_mut((0, 0), (n1, m)).copy_from(&sys1.b);
    let b2d1 = &sys2.b * &sys1.d;
    b.view_mut((n1, 0), (n2, m)).copy_from(&b2d1);

    // C = [D2*C1, C2]
    let mut c = DMatrix::zeros(p, n);
    let d2c1 = &sys2.d * &sys1.c;
    c.view_mut((0, 0), (p, n1)).copy_from(&d2c1);
    c.view_mut((0, n1), (p, n2)).copy_from(&sys2.c);

    // D = D2*D1
    let d = &sys2.d * &sys1.d;

    StateSpaceContinuous::new(a, b, c, d)
}

/// Parallel interconnection: y = sys1(u) + sys2(u).
pub fn parallel(sys1: &StateSpaceContinuous, sys2: &StateSpaceContinuous) -> StateSpaceContinuous {
    assert_eq!(sys1.inputs(), sys2.inputs(), "Input dimensions must match");
    assert_eq!(sys1.outputs(), sys2.outputs(), "Output dimensions must match");

    let n1 = sys1.states();
    let n2 = sys2.states();
    let n = n1 + n2;
    let m = sys1.inputs();
    let p = sys1.outputs();

    // A = [A1, 0; 0, A2]
    let mut a = DMatrix::zeros(n, n);
    a.view_mut((0, 0), (n1, n1)).copy_from(&sys1.a);
    a.view_mut((n1, n1), (n2, n2)).copy_from(&sys2.a);

    // B = [B1; B2]
    let mut b = DMatrix::zeros(n, m);
    b.view_mut((0, 0), (n1, m)).copy_from(&sys1.b);
    b.view_mut((n1, 0), (n2, m)).copy_from(&sys2.b);

    // C = [C1, C2]
    let mut c = DMatrix::zeros(p, n);
    c.view_mut((0, 0), (p, n1)).copy_from(&sys1.c);
    c.view_mut((0, n1), (p, n2)).copy_from(&sys2.c);

    // D = D1 + D2
    let d = &sys1.d + &sys2.d;

    StateSpaceContinuous::new(a, b, c, d)
}

/// Feedback interconnection: sys1 in forward path, sys2 in feedback path.
/// y = sys1(e), e = u - sys2(y) => y = sys1(u - sys2(y))
/// Requires (I + D1*D2) to be invertible.
pub fn feedback(sys1: &StateSpaceContinuous, sys2: &StateSpaceContinuous) -> StateSpaceContinuous {
    assert_eq!(sys1.outputs(), sys2.inputs(), "sys1 outputs must match sys2 inputs");
    assert_eq!(sys2.outputs(), sys1.inputs(), "sys2 outputs must match sys1 inputs");

    let n1 = sys1.states();
    let n2 = sys2.states();
    let n = n1 + n2;
    let m = sys1.inputs();
    let p = sys1.outputs();

    // Closed-loop: need (I + D1*D2)^{-1}
    let eye = DMatrix::identity(p, p);
    let m_mat = &eye + &sys1.d * &sys2.d;
    let m_inv = m_mat.try_inverse().expect("(I + D1*D2) must be invertible");

    // Also need (I + D2*D1)^{-1} for the feedback path
    let eye_m = DMatrix::identity(m, m);
    let m2 = &eye_m + &sys2.d * &sys1.d;
    let m2_inv = m2.try_inverse().expect("(I + D2*D1) must be invertible");

    // A_cl = [A1 - B1*D2*M_inv*C1, -B1*M2_inv*C2; B2*M_inv*C1, A2 - B2*D1*M2_inv*C2]
    // Simplified for clarity
    let a_cl = {
        let mut a = DMatrix::zeros(n, n);
        // Top-left: A1 - B1*D2*M_inv*C1
        let tl = &sys1.a - &sys1.b * &sys2.d * &m_inv * &sys1.c;
        a.view_mut((0, 0), (n1, n1)).copy_from(&tl);
        // Top-right: -B1*(I - D2*M_inv*D1)*C2 = -B1*M2_inv*C2
        let tr = -&sys1.b * &m2_inv * &sys2.c;
        a.view_mut((0, n1), (n1, n2)).copy_from(&tr);
        // Bottom-left: B2*M_inv*C1
        let bl = &sys2.b * &m_inv * &sys1.c;
        a.view_mut((n1, 0), (n2, n1)).copy_from(&bl);
        // Bottom-right: A2 - B2*M_inv*D1*C2
        let br = &sys2.a - &sys2.b * &m_inv * &sys1.d * &sys2.c;
        a.view_mut((n1, n1), (n2, n2)).copy_from(&br);
        a
    };

    let b_cl = {
        let mut b = DMatrix::zeros(n, m);
        let top = &sys1.b * &m2_inv;
        b.view_mut((0, 0), (n1, m)).copy_from(&top);
        let bot = -&sys2.b * &m_inv * &sys1.d;
        b.view_mut((n1, 0), (n2, m)).copy_from(&bot);
        b
    };

    let c_cl = {
        let mut c = DMatrix::zeros(p, n);
        let left = &m_inv * &sys1.c;
        c.view_mut((0, 0), (p, n1)).copy_from(&left);
        let right = -&m_inv * &sys1.d * &sys2.c;
        c.view_mut((0, n1), (p, n2)).copy_from(&right);
        c
    };

    let d_cl = &m_inv * &sys1.d;

    StateSpaceContinuous::new(a_cl, b_cl, c_cl, d_cl)
}
