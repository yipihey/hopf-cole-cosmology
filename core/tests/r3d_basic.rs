use hcc_core::r3d::{num_moments, voxelize, Poly};

fn tri_moments(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 3] {
    let area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
    [area, area * (a[0] + b[0] + c[0]) / 3.0, area * (a[1] + b[1] + c[1]) / 3.0]
}

#[test]
fn polygon_moments_closed_form() {
    let mut p = Poly::<2>::with_capacity(16);
    p.init_polygon(&[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]);
    let mut m = vec![0.0; 6];
    p.moments(2, &mut m);
    // rectangle [0,2]x[0,1]: area 2, ∫x = 2, ∫y = 1, ∫x² = 8/3, ∫xy = 1, ∫y² = 2/3
    for (got, exp) in m.iter().zip([2.0, 2.0, 1.0, 8.0 / 3.0, 1.0, 2.0 / 3.0]) { assert!((got - exp).abs() < 1e-12, "{got} vs {exp}"); }
}

#[test]
fn tet_and_box_moments() {
    let mut b = Poly::<3>::with_capacity(64);
    b.init_box([0.0, 0.0, 0.0], [2.0, 1.0, 3.0]);
    let mut m = vec![0.0; num_moments(3, 2)];
    b.moments(2, &mut m);
    // volume 6, ∫x = 6, ∫y = 3, ∫z = 9, ∫x² = 8, ∫xy = 3, ∫xz = 9, ∫y² = 2, ∫yz = 4.5, ∫z² = 18
    for (got, exp) in m.iter().zip([6.0, 6.0, 3.0, 9.0, 8.0, 3.0, 9.0, 2.0, 4.5, 18.0]) { assert!((got - exp).abs() < 1e-10, "{got} vs {exp}"); }
    let mut t = Poly::<3>::with_capacity(64);
    t.init_tet([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let mut m = vec![0.0; 4];
    t.moments(1, &mut m);
    for (got, exp) in m.iter().zip([1.0 / 6.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0]) { assert!((got - exp).abs() < 1e-12, "{got} vs {exp}"); }
    // flipped orientation gives the same (positive) moments
    let mut t2 = Poly::<3>::with_capacity(64);
    t2.init_tet([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let mut m2 = vec![0.0; 4];
    t2.moments(1, &mut m2);
    assert!((m2[0] - 1.0 / 6.0).abs() < 1e-12);
}

#[test]
fn voxelize_triangle_conserves_moments() {
    let (a, b, c) = ([0.13, 0.21], [2.71, 0.35], [1.1, 2.9]);
    let mut p = Poly::<2>::with_capacity(32);
    p.init_triangle(a, b, c);
    let mut tot = [0.0; 3];
    let mut cells = 0;
    voxelize(&p, 0.25, 1, |_, m| { tot[0] += m[0]; tot[1] += m[1]; tot[2] += m[2]; cells += 1; assert!(m[0] >= -1e-15); });
    let exp = tri_moments(a, b, c);
    println!("cells {cells}, totals {tot:?} vs {exp:?}");
    for i in 0..3 { assert!((tot[i] - exp[i]).abs() < 1e-12); }
    // unit square aligned with the grid: every cell gets exactly d²
    let mut q = Poly::<2>::with_capacity(32);
    q.init_polygon(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    let mut n = 0;
    voxelize(&q, 0.25, 0, |_, m| { assert!((m[0] - 0.0625).abs() < 1e-14); n += 1; });
    assert_eq!(n, 16);
}

#[test]
fn voxelize_tet_conserves_moments() {
    let v = [[0.1, 0.2, 0.05], [2.3, 0.4, 0.3], [0.7, 2.6, 0.5], [1.1, 1.0, 2.9]];
    let mut t = Poly::<3>::with_capacity(64);
    t.init_tet(v);
    let mut exp = vec![0.0; 4];
    t.moments(1, &mut exp);
    let mut tot = [0.0; 4];
    let mut cells = 0;
    voxelize(&t, 0.3, 1, |_, m| { for i in 0..4 { tot[i] += m[i]; } cells += 1; assert!(m[0] >= -1e-14); });
    println!("cells {cells}, totals {tot:?} vs {exp:?}");
    for i in 0..4 { assert!((tot[i] - exp[i]).abs() < 1e-11, "{} vs {}", tot[i], exp[i]); }
    // a grid-aligned box: every cell exactly d³
    let mut b = Poly::<3>::with_capacity(64);
    b.init_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let mut n = 0;
    voxelize(&b, 0.5, 0, |_, m| { assert!((m[0] - 0.125).abs() < 1e-14); n += 1; });
    assert_eq!(n, 8);
}
