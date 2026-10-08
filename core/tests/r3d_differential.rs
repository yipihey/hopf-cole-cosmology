//! Differential test against R3D.jl (T. Abel's Julia port of r3d): every cell's
//! moments of random triangles/tetrahedra voxelized on random grids.
use hcc_core::r3d::{voxelize, Poly};
use std::collections::HashMap;

#[test]
fn matches_r3d_jl_fixtures() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures_r3d.txt")).expect("fixtures");
    let mut lines = text.lines();
    let mut ncases = 0;
    let mut maxerr: f64 = 0.0;
    while let Some(head) = lines.next() {
        let toks: Vec<&str> = head.split_whitespace().collect();
        if toks.is_empty() { continue; }
        let dim = if toks[0] == "T2" { 2 } else { 3 };
        let nums: Vec<f64> = toks[1..].iter().map(|t| t.parse().unwrap()).collect();
        let nv = dim + 1;
        let d = nums[nv * dim];
        let order = nums[nv * dim + 1] as usize;
        let ncells: usize = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut expect: HashMap<Vec<i64>, Vec<f64>> = HashMap::new();
        for _ in 0..ncells {
            let v: Vec<f64> = lines.next().unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
            let key: Vec<i64> = v[..dim].iter().map(|&x| x as i64).collect();
            expect.insert(key, v[dim..].to_vec());
        }
        let mut got: HashMap<Vec<i64>, Vec<f64>> = HashMap::new();
        if dim == 2 {
            let mut p = Poly::<2>::with_capacity(32);
            p.init_triangle([nums[0], nums[1]], [nums[2], nums[3]], [nums[4], nums[5]]);
            voxelize(&p, d, order, |c, m| { if m[0].abs() > 1e-14 { got.insert(vec![c[0], c[1]], m.to_vec()); } });
        } else {
            let mut p = Poly::<3>::with_capacity(64);
            p.init_tet([[nums[0], nums[1], nums[2]], [nums[3], nums[4], nums[5]], [nums[6], nums[7], nums[8]], [nums[9], nums[10], nums[11]]]);
            voxelize(&p, d, order, |c, m| { if m[0].abs() > 1e-14 { got.insert(vec![c[0], c[1], c[2]], m.to_vec()); } });
        }
        // compare: every expected cell present with the same moments (scale by the largest cell moment)
        // R3D.jl keeps the input orientation (negative moments for clockwise /
        // negatively oriented simplices); the Rust constructors reorient, so
        // compare up to the overall sign of the area.
        let scale = expect.values().map(|m| m[0].abs()).fold(0.0, f64::max);
        let sgn = expect.values().map(|m| m[0]).sum::<f64>().signum();
        for (k, em) in &expect {
            let gm = got.get(k).unwrap_or_else(|| panic!("case {ncases}: cell {k:?} missing in Rust output"));
            for i in 0..em.len() {
                let e = (gm[i] - sgn * em[i]).abs() / scale.max(1e-300);
                maxerr = maxerr.max(e);
                assert!(e < 1e-9, "case {ncases} dim {dim} order {order} cell {k:?} moment {i}: rust {} vs julia {}", gm[i], em[i]);
            }
        }
        for k in got.keys() { assert!(expect.contains_key(k), "case {ncases}: extra cell {k:?} in Rust output"); }
        ncases += 1;
    }
    println!("{ncases} fixture cases matched R3D.jl; max scaled error {maxerr:.2e}");
    assert!(ncases >= 60);
}
