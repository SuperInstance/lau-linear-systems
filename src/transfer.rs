//! Transfer function representation and conversion to/from state-space.

use nalgebra::DMatrix;
use crate::ss::StateSpaceContinuous;

/// SISO transfer function: numerator and denominator polynomial coefficients.
///
/// Represented as b[0]*s^m + b[1]*s^{m-1} + ... + b[m] / a[0]*s^n + a[1]*s^{n-1} + ... + a[n]
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Numerator coefficients (highest power first)
    pub num: Vec<f64>,
    /// Denominator coefficients (highest power first), monic
    pub den: Vec<f64>,
}

impl TransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Self {
        assert!(!den.is_empty(), "Denominator must be non-empty");
        let scale = den[0];
        if scale.abs() > 1e-15 {
            Self {
                num: num.iter().map(|x| x / scale).collect(),
                den: den.iter().map(|x| x / scale).collect(),
            }
        } else {
            Self { num, den }
        }
    }

    /// Evaluate at complex value s.
    pub fn evaluate(&self, s: num_complex::Complex64) -> num_complex::Complex64 {
        let num_val = horner(&self.num, s);
        let den_val = horner(&self.den, s);
        if den_val.norm() < 1e-15 {
            num_complex::Complex64::new(f64::INFINITY, 0.0)
        } else {
            num_val / den_val
        }
    }

    pub fn order(&self) -> usize { self.den.len() - 1 }
    pub fn poles(&self) -> Vec<num_complex::Complex64> { find_roots(&self.den) }
    pub fn zeros(&self) -> Vec<num_complex::Complex64> { find_roots(&self.num) }
}

fn horner(coeffs: &[f64], s: num_complex::Complex64) -> num_complex::Complex64 {
    let mut result = num_complex::Complex64::new(0.0, 0.0);
    for &c in coeffs {
        result = result * s + num_complex::Complex64::new(c, 0.0);
    }
    result
}

fn find_roots(coeffs: &[f64]) -> Vec<num_complex::Complex64> {
    let start = coeffs.iter().position(|&c| c.abs() > 1e-15).unwrap_or(coeffs.len() - 1);
    let c = &coeffs[start..];
    let n = c.len() - 1;
    if n == 0 { return vec![]; }
    if n == 1 { return vec![num_complex::Complex64::new(-c[1] / c[0], 0.0)]; }

    // Companion matrix with super-diagonal ones
    let mut comp = DMatrix::zeros(n, n);
    for i in 0..n - 1 {
        comp[(i, i + 1)] = 1.0;
    }
    for i in 0..n {
        comp[(n - 1, i)] = -c[n - i] / c[0];
    }

    crate::ss::compute_eigenvalues(&comp)
}

/// Convert state-space to transfer function (SISO, first input to first output).
///
/// Uses the Faddeev-LeVerrier algorithm to compute C*adj(sI-A)*B.
pub fn ss2tf(sys: &StateSpaceContinuous) -> TransferFunction {
    let n = sys.states();
    let d_val = sys.d[(0, 0)];

    // Characteristic polynomial via Leverrier: det(sI-A) = s^n + c1*s^{n-1} + ... + cn
    let (den_coeffs, residues) = leverrier(&sys.a);
    // den_coeffs[k] = coefficient of s^{n-k} for k=0..n
    // residues[k] = M_{k-1} for k=1..n (M_0 = I, M_k = A*M_{k-1} + c_k*I)

    // Numerator = sum_{k=1}^{n} C * M_{k-1} * B * c_{den}(k) + d * den(s)
    // where den(s) evaluated at s gives the full denominator
    // Actually: H(s) = (C * adj(sI-A) * B + d * det(sI-A)) / det(sI-A)
    // adj(sI-A) = M_1*s^{n-1} + M_2*s^{n-2} + ... + M_{n-1}*s + M_n... no
    //
    // From Leverrier: adj(sI-A) = sum_{k=0}^{n-1} M_k * s^{n-1-k}
    // where M_0 = I, M_k = A*M_{k-1} + c_k*I
    //
    // Wait, the standard Leverrier gives:
    // det(sI-A) = s^n + c1*s^{n-1} + ... + cn
    // adj(sI-A) = M_0*s^{n-1} + M_1*s^{n-2} + ... + M_{n-1}
    // where M_0 = I, M_k = A*M_{k-1} + c_k*I for k=1..n-1... 
    // Actually there are different conventions. Let me use the one where:
    // c_k = -trace(A*M_{k-1})/k, M_k = A*M_{k-1} + c_k*I
    // with M_0 = I
    // Then: det(sI-A) = s^n + c1*s^{n-1} + ... + cn
    // adj(sI-A) = M_0*s^{n-1} + M_1*s^{n-2} + ... + M_{n-1}
    
    // Numerator of H(s) = C*adj(sI-A)*B + d*det(sI-A)
    // = C*(M_0*s^{n-1} + M_1*s^{n-2} + ... + M_{n-1})*B + d*(s^n + c1*s^{n-1} + ... + cn)
    // = d*s^n + (C*M_0*B + d*c1)*s^{n-1} + (C*M_1*B + d*c2)*s^{n-2} + ... + (C*M_{n-1}*B + d*cn)

    let b_col = sys.b.column(0);
    let c_row = sys.c.row(0);

    let mut num_coeffs = vec![0.0; n + 1]; // index k -> coefficient of s^{n-k}
    num_coeffs[0] = d_val; // s^n term: just d
    for k in 0..n {
        // C * M_k * B for the s^{n-1-k} term
        let cmb: f64 = (&c_row * &(&residues[k] * b_col))[0];
        num_coeffs[k + 1] = cmb + d_val * den_coeffs[k + 1];
    }

    // Convert to highest-power-first
    TransferFunction::new(num_coeffs, den_coeffs)
}

/// Leverrier's algorithm. Returns (char_poly_coeffs, residues).
/// char_poly[k] = coefficient of s^{n-k} (index 0 = s^n, index n = constant)
/// residues[k] = M_k matrix for k=0..n-1
fn leverrier(a: &DMatrix<f64>) -> (Vec<f64>, Vec<DMatrix<f64>>) {
    let n = a.nrows();
    let mut c = vec![0.0; n + 1];
    c[0] = 1.0;

    let mut m = Vec::new();
    m.push(DMatrix::identity(n, n)); // M_0 = I

    for k in 1..=n {
        let am = a * &m[k - 1];
        let trace: f64 = (0..n).map(|i| am[(i, i)]).sum();
        c[k] = -trace / k as f64;
        if k < n {
            m.push(&am + c[k] * DMatrix::identity(n, n));
        }
    }

    (c, m)
}

/// Convert transfer function to state-space (controllable canonical form).
pub fn tf2ss(tf: &TransferFunction) -> StateSpaceContinuous {
    let n = tf.order();
    
    let num_start = tf.num.iter().position(|&c| c.abs() > 1e-15).unwrap_or(tf.num.len() - 1);
    let num_trimmed: Vec<f64> = tf.num[num_start..].to_vec();
    
    if n == 0 {
        let gain = if num_trimmed.len() == 1 { num_trimmed[0] } else { 0.0 };
        return StateSpaceContinuous::new(
            DMatrix::zeros(1, 1),
            DMatrix::zeros(1, 1),
            DMatrix::zeros(1, 1),
            DMatrix::from_element(1, 1, gain),
        );
    }

    // Pad numerator to length n+1 from the front
    let mut num = num_trimmed;
    while num.len() > n + 1 {
        num.remove(0); // Remove leading coefficients if improper
    }
    while num.len() < n + 1 {
        num.insert(0, 0.0);
    }
    // num[0] is coefficient of s^n, num[n] is constant term

    // Controllable canonical form
    let mut a = DMatrix::zeros(n, n);
    for i in 0..n {
        a[(0, i)] = -tf.den[i + 1];
    }
    for i in 1..n {
        a[(i, i - 1)] = 1.0;
    }

    let mut b = DMatrix::zeros(n, 1);
    b[(0, 0)] = 1.0;

    // D = leading coefficient of numerator (coefficient of s^n)
    // C[i] = num[i+1] - num[0] * den[i+1] for i=0..n-1
    // Actually: C = num[1:] - num[0]*den[1:], D = num[0]
    // This handles both strictly proper (num[0]=0) and proper cases
    let mut c = DMatrix::zeros(1, n);
    for i in 0..n {
        c[(0, i)] = num[i + 1] - num[0] * tf.den[i + 1];
    }

    let d_val = num[0];

    StateSpaceContinuous::new(a, b, c, DMatrix::from_element(1, 1, d_val))
}
