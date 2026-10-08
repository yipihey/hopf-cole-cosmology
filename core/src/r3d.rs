//! Exact polytope clipping, moment integration and conservative
//! voxelization in 2D and 3D: a port of the core of Devon Powell's r3d/r2d
//! (via the Julia port R3D.jl by T. Abel).  Polytopes are stored as a
//! vertex graph: every vertex carries D neighbour indices; in 2D the slots
//! are (next, prev) around the polygon, in 3D the three neighbours are
//! ordered so that walking `(np + 1) % 3` turns around a face.
//!
//! Used for the exact (area/volume-weighted) deposition of the phase-space
//! sheet: `voxelize` returns, for every grid cell, the polynomial moments
//! of the cell ∩ simplex region, from which P0 (order 0) and P1 (order 1)
//! sheet densities follow without any point sampling.

const NONE: i32 = -1;

#[derive(Clone)]
pub struct Poly<const D: usize> {
    pub pos: Vec<[f64; D]>,
    pub nbr: Vec<[i32; D]>,
    pub nverts: usize,
}

impl<const D: usize> Poly<D> {
    pub fn with_capacity(cap: usize) -> Self {
        Poly { pos: vec![[0.0; D]; cap], nbr: vec![[NONE; D]; cap], nverts: 0 }
    }
    fn ensure(&mut self, n: usize) {
        if self.pos.len() < n {
            self.pos.resize(n, [0.0; D]);
            self.nbr.resize(n, [NONE; D]);
        }
    }
    pub fn copy_from(&mut self, src: &Poly<D>) {
        self.ensure(src.nverts);
        self.pos[..src.nverts].copy_from_slice(&src.pos[..src.nverts]);
        self.nbr[..src.nverts].copy_from_slice(&src.nbr[..src.nverts]);
        self.nverts = src.nverts;
    }
    /// Integer bounding box in cells of size `d` (origin at 0): lo = floor(min/d), hi = ceil(max/d).
    pub fn ibox(&self, d: f64) -> ([i64; D], [i64; D]) {
        let mut lo = [i64::MAX; D];
        let mut hi = [i64::MIN; D];
        for v in 0..self.nverts {
            for a in 0..D {
                let x = self.pos[v][a] / d;
                lo[a] = lo[a].min(x.floor() as i64);
                hi[a] = hi[a].max(x.ceil() as i64);
            }
        }
        for a in 0..D { if hi[a] == lo[a] { hi[a] += 1; } }
        (lo, hi)
    }
}

// ---------------------------------------------------------------------------
// 2D

impl Poly<2> {
    /// Counter-clockwise polygon from its vertices.
    pub fn init_polygon(&mut self, verts: &[[f64; 2]]) {
        let n = verts.len();
        self.ensure(n);
        for i in 0..n {
            self.pos[i] = verts[i];
            self.nbr[i] = [((i + 1) % n) as i32, ((i + n - 1) % n) as i32];
        }
        self.nverts = n;
    }
    /// Triangle with positive orientation (vertices reordered if needed).
    pub fn init_triangle(&mut self, a: [f64; 2], b: [f64; 2], c: [f64; 2]) {
        let area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if area2 >= 0.0 { self.init_polygon(&[a, b, c]) } else { self.init_polygon(&[a, c, b]) }
    }
    /// Polygon vertices in order (walking the `next` slot).
    pub fn ordered(&self) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(self.nverts);
        if self.nverts == 0 { return out; }
        let mut v = 0usize;
        for _ in 0..self.nverts {
            out.push(self.pos[v]);
            v = self.nbr[v][0] as usize;
        }
        out
    }
    /// Moments ∫ x^a y^b dA for a+b ≤ order (order ≤ 2) in the r2d ordering
    /// [1, x, y, x², xy, y²], by Green's theorem edge sums.
    pub fn moments(&self, order: usize, out: &mut [f64]) {
        for m in out.iter_mut() { *m = 0.0; }
        let p = self.ordered();
        let n = p.len();
        if n < 3 { return; }
        for i in 0..n {
            let (x0, y0) = (p[i][0], p[i][1]);
            let (x1, y1) = (p[(i + 1) % n][0], p[(i + 1) % n][1]);
            let c = x0 * y1 - x1 * y0;
            out[0] += c / 2.0;
            if order >= 1 {
                out[1] += c * (x0 + x1) / 6.0;
                out[2] += c * (y0 + y1) / 6.0;
            }
            if order >= 2 {
                out[3] += c * (x0 * x0 + x0 * x1 + x1 * x1) / 12.0;
                out[4] += c * (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) / 24.0;
                out[5] += c * (y0 * y0 + y0 * y1 + y1 * y1) / 12.0;
            }
        }
    }
    /// Split along x[ax] = coord into `out0` (x ≤ coord) and `out1` (x > coord).
    /// Mirrors r2d_split_coord.  `self` is consumed.
    pub fn split_coord(&mut self, out0: &mut Poly<2>, out1: &mut Poly<2>, coord: f64, ax: usize, sd: &mut Vec<f64>, side: &mut Vec<i32>, map: &mut Vec<i32>) {
        if self.nverts == 0 { out0.nverts = 0; out1.nverts = 0; return; }
        let onv = self.nverts;
        sd.clear(); sd.resize(onv * 3 + 8, 0.0);
        side.clear(); side.resize(onv * 3 + 8, 0);
        let mut nright = 0;
        for v in 0..onv {
            let s = self.pos[v][ax] - coord;
            sd[v] = s;
            if s > 0.0 { side[v] = 1; nright += 1; } else { side[v] = 0; }
        }
        if nright == 0 { out0.copy_from(self); out1.nverts = 0; return; }
        if nright == onv { out1.copy_from(self); out0.nverts = 0; return; }
        for vcur in 0..onv {
            if side[vcur] != 0 { continue; }
            for np in 0..2 {
                let vnext = self.nbr[vcur][np];
                if vnext < 0 || side[vnext as usize] == 0 { continue; }
                let vnext = vnext as usize;
                let wa = -sd[vnext];
                let wb = sd[vcur];
                let inv = 1.0 / (wa + wb);
                let nx = (wa * self.pos[vcur][0] + wb * self.pos[vnext][0]) * inv;
                let ny = (wa * self.pos[vcur][1] + wb * self.pos[vnext][1]) * inv;
                let other = 1 - np;
                self.ensure(self.nverts + 2);
                let nl = self.nverts; self.nverts += 1;
                self.pos[nl] = [nx, ny];
                self.nbr[nl][other] = vcur as i32;
                self.nbr[nl][np] = NONE;
                self.nbr[vcur][np] = nl as i32;
                if side.len() <= nl + 1 { side.resize(nl + 2, 0); }
                side[nl] = 0;
                let nr = self.nverts; self.nverts += 1;
                self.pos[nr] = [nx, ny];
                self.nbr[nr][other] = NONE;
                self.nbr[nr][np] = vnext as i32;
                self.nbr[vnext][other] = nr as i32;
                side[nr] = 1;
            }
        }
        // link: walk the `next` slot from each unset new vertex until another new vertex
        for vstart in onv..self.nverts {
            if self.nbr[vstart][1] != NONE { continue; }
            let mut vcur = self.nbr[vstart][0] as usize;
            while vcur < onv { vcur = self.nbr[vcur][0] as usize; }
            self.nbr[vstart][1] = vcur as i32;
            self.nbr[vcur][0] = vstart as i32;
        }
        compact_split(self, out0, out1, side, map);
    }
}

// ---------------------------------------------------------------------------
// 3D

impl Poly<3> {
    /// Tetrahedron with positive orientation (vertices reordered if needed).
    pub fn init_tet(&mut self, v: [[f64; 3]; 4]) {
        let e1 = sub(v[1], v[0]); let e2 = sub(v[2], v[0]); let e3 = sub(v[3], v[0]);
        let det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
        let verts = if det >= 0.0 { v } else { [v[0], v[2], v[1], v[3]] };
        self.ensure(4);
        const NB: [[i32; 3]; 4] = [[1, 3, 2], [2, 3, 0], [0, 3, 1], [1, 2, 0]];
        for i in 0..4 { self.pos[i] = verts[i]; self.nbr[i] = NB[i]; }
        self.nverts = 4;
    }
    pub fn init_box(&mut self, lo: [f64; 3], hi: [f64; 3]) {
        self.ensure(8);
        let p = [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]];
        const NB: [[i32; 3]; 8] = [[1, 4, 3], [2, 5, 0], [3, 6, 1], [0, 7, 2], [7, 0, 5], [4, 1, 6], [5, 2, 7], [6, 3, 4]];
        for i in 0..8 { self.pos[i] = p[i]; self.nbr[i] = NB[i]; }
        self.nverts = 8;
    }
    /// Moments ∫ x^i y^j z^k dV, i+j+k ≤ order, r3d ordering
    /// [1, x, y, z, x², xy, xz, y², yz, z²] (Koehl recursion over face fans).
    pub fn moments(&self, order: usize, out: &mut [f64]) {
        for m in out.iter_mut() { *m = 0.0; }
        let nv = self.nverts;
        if nv == 0 { return; }
        let np1 = order + 1;
        let mut emarks = vec![[false; 3]; nv];
        let idx = |i: usize, j: usize, l: usize| (l * np1 + i) * np1 + j;
        let mut s = vec![0.0; np1 * np1 * 2];
        let mut dd = vec![0.0; np1 * np1 * 2];
        let mut cc = vec![0.0; np1 * np1 * 2];
        let mut prev = 0usize;
        let mut cur = 1usize;
        for vstart in 0..nv {
            for pstart in 0..3 {
                if emarks[vstart][pstart] { continue; }
                let mut pnext = pstart;
                let mut vcur = vstart;
                emarks[vcur][pnext] = true;
                let mut vnext = self.nbr[vcur][pnext] as usize;
                let v0 = self.pos[vcur];
                let mut np = 0;
                for k in 0..3 { if self.nbr[vnext][k] as usize == vcur { np = k; break; } }
                vcur = vnext;
                pnext = (np + 1) % 3;
                emarks[vcur][pnext] = true;
                vnext = self.nbr[vcur][pnext] as usize;
                while vnext != vstart {
                    let v2 = self.pos[vcur];
                    let v1 = self.pos[vnext];
                    let sixv = -v2[0] * v1[1] * v0[2] + v1[0] * v2[1] * v0[2] + v2[0] * v0[1] * v1[2] - v0[0] * v2[1] * v1[2] - v1[0] * v0[1] * v2[2] + v0[0] * v1[1] * v2[2];
                    s[idx(0, 0, prev)] = 1.0; dd[idx(0, 0, prev)] = 1.0; cc[idx(0, 0, prev)] = 1.0;
                    out[0] += sixv / 6.0;
                    let mut m = 0usize;
                    for corder in 1..=order {
                        for i in (0..=corder).rev() {
                            for j in (0..=(corder - i)).rev() {
                                let k = corder - i - j;
                                m += 1;
                                let (mut cv, mut dv, mut sv) = (0.0, 0.0, 0.0);
                                if i > 0 { cv += v2[0] * cc[idx(i - 1, j, prev)]; dv += v1[0] * dd[idx(i - 1, j, prev)]; sv += v0[0] * s[idx(i - 1, j, prev)]; }
                                if j > 0 { cv += v2[1] * cc[idx(i, j - 1, prev)]; dv += v1[1] * dd[idx(i, j - 1, prev)]; sv += v0[1] * s[idx(i, j - 1, prev)]; }
                                if k > 0 { cv += v2[2] * cc[idx(i, j, prev)]; dv += v1[2] * dd[idx(i, j, prev)]; sv += v0[2] * s[idx(i, j, prev)]; }
                                dv += cv;
                                sv += dv;
                                cc[idx(i, j, cur)] = cv; dd[idx(i, j, cur)] = dv; s[idx(i, j, cur)] = sv;
                                out[m] += sixv * sv;
                            }
                        }
                        std::mem::swap(&mut cur, &mut prev);
                    }
                    let mut np = 0;
                    for k in 0..3 { if self.nbr[vnext][k] as usize == vcur { np = k; break; } }
                    vcur = vnext;
                    pnext = (np + 1) % 3;
                    emarks[vcur][pnext] = true;
                    vnext = self.nbr[vcur][pnext] as usize;
                }
            }
        }
        // normalization: multinomial × (corder+1)(corder+2)(corder+3)
        cc[idx(0, 0, prev)] = 1.0;
        let mut m = 0usize;
        for corder in 1..=order {
            for i in (0..=corder).rev() {
                for j in (0..=(corder - i)).rev() {
                    let k = corder - i - j;
                    m += 1;
                    let mut cv = 0.0;
                    if i > 0 { cv += cc[idx(i - 1, j, prev)]; }
                    if j > 0 { cv += cc[idx(i, j - 1, prev)]; }
                    if k > 0 { cv += cc[idx(i, j, prev)]; }
                    cc[idx(i, j, cur)] = cv;
                    out[m] /= cv * ((corder + 1) * (corder + 2) * (corder + 3)) as f64;
                }
            }
            std::mem::swap(&mut cur, &mut prev);
        }
    }
    /// Split along x[ax] = coord (mirrors r3d_split_coord).  `self` is consumed.
    pub fn split_coord(&mut self, out0: &mut Poly<3>, out1: &mut Poly<3>, coord: f64, ax: usize, sd: &mut Vec<f64>, side: &mut Vec<i32>, map: &mut Vec<i32>) {
        if self.nverts == 0 { out0.nverts = 0; out1.nverts = 0; return; }
        let onv = self.nverts;
        sd.clear(); sd.resize(onv * 3 + 8, 0.0);
        side.clear(); side.resize(onv * 3 + 8, 0);
        let mut nright = 0;
        for v in 0..onv {
            let s = self.pos[v][ax] - coord;
            sd[v] = s;
            if s > 0.0 { side[v] = 1; nright += 1; } else { side[v] = 0; }
        }
        if nright == 0 { out0.copy_from(self); out1.nverts = 0; return; }
        if nright == onv { out1.copy_from(self); out0.nverts = 0; return; }
        for vcur in 0..onv {
            if side[vcur] != 0 { continue; }
            for np in 0..3 {
                let vnext = self.nbr[vcur][np];
                if vnext < 0 || side[vnext as usize] == 0 { continue; }
                let vnext = vnext as usize;
                let wa = -sd[vnext];
                let wb = sd[vcur];
                let inv = 1.0 / (wa + wb);
                let mut npos = [0.0; 3];
                for a in 0..3 { npos[a] = (wa * self.pos[vcur][a] + wb * self.pos[vnext][a]) * inv; }
                self.ensure(self.nverts + 2);
                if side.len() < self.nverts + 2 { side.resize(self.nverts + 2, 0); }
                let nl = self.nverts; self.nverts += 1;
                self.pos[nl] = npos;
                self.nbr[nl] = [vcur as i32, NONE, NONE];
                self.nbr[vcur][np] = nl as i32;
                side[nl] = 0;
                let nr = self.nverts; self.nverts += 1;
                self.pos[nr] = npos;
                self.nbr[nr] = [vnext as i32, NONE, NONE];
                side[nr] = 1;
                for k in 0..3 { if self.nbr[vnext][k] as usize == vcur { self.nbr[vnext][k] = nr as i32; break; } }
            }
        }
        for vstart in onv..self.nverts {
            let mut vcur = vstart;
            let mut vnext = self.nbr[vcur][0] as usize;
            loop {
                let mut np = 0;
                for k in 0..3 { if self.nbr[vnext][k] as usize == vcur { np = k; break; } }
                vcur = vnext;
                vnext = self.nbr[vcur][(np + 1) % 3] as usize;
                if vcur >= onv { break; }
            }
            self.nbr[vstart][2] = vcur as i32;
            self.nbr[vcur][1] = vstart as i32;
        }
        compact_split(self, out0, out1, side, map);
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }

/// Compact `src` into out0 (side 0) and out1 (side 1), reindexing neighbours.
/// `src` must not alias the outputs.
fn compact_split<const D: usize>(src: &Poly<D>, out0: &mut Poly<D>, out1: &mut Poly<D>, side: &[i32], map: &mut Vec<i32>) {
    let total = src.nverts;
    out0.ensure(total); out1.ensure(total);
    out0.nverts = 0; out1.nverts = 0;
    map.clear(); map.resize(total, 0);
    for v in 0..total {
        if side[v] == 0 { map[v] = out0.nverts as i32; out0.nverts += 1; } else { map[v] = out1.nverts as i32; out1.nverts += 1; }
    }
    let (mut i0, mut i1) = (0usize, 0usize);
    for v in 0..total {
        let mut nb = src.nbr[v];
        for a in 0..D { if nb[a] >= 0 { nb[a] = map[nb[a] as usize]; } }
        if side[v] == 0 { out0.pos[i0] = src.pos[v]; out0.nbr[i0] = nb; i0 += 1; } else { out1.pos[i1] = src.pos[v]; out1.nbr[i1] = nb; i1 += 1; }
    }
}

/// Number of moments up to `order` in D dimensions.
pub fn num_moments(d: usize, order: usize) -> usize {
    match d { 2 => (order + 1) * (order + 2) / 2, _ => (order + 1) * (order + 2) * (order + 3) / 6 }
}

/// Reusable scratch for `voxelize` (no allocations in the hot loop).
pub struct VoxWorkspace<const D: usize> {
    polys: Vec<Poly<D>>,
    scratch: Poly<D>,
    iboxes: Vec<([i64; D], [i64; D])>,
    sd: Vec<f64>,
    side: Vec<i32>,
    map: Vec<i32>,
    mom: Vec<f64>,
}
impl<const D: usize> VoxWorkspace<D> {
    pub fn new(cap: usize) -> Self {
        VoxWorkspace { polys: (0..64).map(|_| Poly::<D>::with_capacity(cap)).collect(), scratch: Poly::<D>::with_capacity(cap), iboxes: vec![([0; D], [0; D]); 64], sd: Vec::new(), side: Vec::new(), map: Vec::new(), mom: Vec::new() }
    }
}
impl<const D: usize> Default for VoxWorkspace<D> {
    fn default() -> Self { Self::new(64) }
}

/// Conservative voxelization: calls `f(cell_index[D], moments)` for every grid
/// cell (cell size `d`, origin 0, unbounded integer indices) that `poly`
/// overlaps, with the moments of the overlap region up to `order`.
pub fn voxelize<const D: usize, F: FnMut([i64; D], &[f64])>(poly: &Poly<D>, d: f64, order: usize, f: F)
where
    Poly<D>: Split<D>,
{
    let mut ws = VoxWorkspace::<D>::new(poly.nverts + 32);
    voxelize_ws(&mut ws, poly, d, order, f)
}

/// `voxelize` with a caller-provided workspace (stack-based bisection along
/// the grid planes, mirrors r3d_voxelize).
pub fn voxelize_ws<const D: usize, F: FnMut([i64; D], &[f64])>(ws: &mut VoxWorkspace<D>, poly: &Poly<D>, d: f64, order: usize, mut f: F)
where
    Poly<D>: Split<D>,
{
    if poly.nverts == 0 { return; }
    let (lo, hi) = poly.ibox(d);
    let nm = num_moments(D, order);
    ws.mom.clear(); ws.mom.resize(nm, 0.0);
    ws.polys[0].copy_from(poly);
    ws.iboxes[0] = (lo, hi);
    let mut nstack = 1usize;
    while nstack > 0 {
        nstack -= 1;
        let (lo, hi) = ws.iboxes[nstack];
        if ws.polys[nstack].nverts == 0 { continue; }
        let mut spax = 0;
        let mut dmax = hi[0] - lo[0];
        for a in 1..D { if hi[a] - lo[a] > dmax { dmax = hi[a] - lo[a]; spax = a; } }
        if dmax <= 1 {
            let (mom, polys) = (&mut ws.mom, &ws.polys);
            polys[nstack].moments(order, mom);
            f(lo, mom);
            continue;
        }
        let half = dmax / 2;
        let split_index = lo[spax] + half;
        let coord = split_index as f64 * d;
        // split scratch (= polys[nstack]) into polys[nstack] (left) and polys[nstack+1] (right)
        if ws.polys.len() < nstack + 2 { let cap = ws.polys[0].pos.len(); ws.polys.push(Poly::<D>::with_capacity(cap)); ws.iboxes.push(([0; D], [0; D])); }
        ws.scratch.copy_from(&ws.polys[nstack]);
        let (left, right) = ws.polys.split_at_mut(nstack + 1);
        ws.scratch.split_coord(&mut left[nstack], &mut right[0], coord, spax, &mut ws.sd, &mut ws.side, &mut ws.map);
        let mut hi_left = hi; hi_left[spax] = split_index;
        let mut lo_right = lo; lo_right[spax] = split_index;
        ws.iboxes[nstack] = (lo, hi_left);
        ws.iboxes[nstack + 1] = (lo_right, hi);
        nstack += 2;
    }
}

/// Dimension-generic access to the split and moment kernels.
pub trait Split<const D: usize> {
    fn split_coord(&mut self, out0: &mut Poly<D>, out1: &mut Poly<D>, coord: f64, ax: usize, sd: &mut Vec<f64>, side: &mut Vec<i32>, map: &mut Vec<i32>);
    fn moments(&self, order: usize, out: &mut [f64]);
}
impl Split<2> for Poly<2> {
    fn split_coord(&mut self, out0: &mut Poly<2>, out1: &mut Poly<2>, coord: f64, ax: usize, sd: &mut Vec<f64>, side: &mut Vec<i32>, map: &mut Vec<i32>) { Poly::<2>::split_coord(self, out0, out1, coord, ax, sd, side, map) }
    fn moments(&self, order: usize, out: &mut [f64]) { Poly::<2>::moments(self, order, out) }
}
impl Split<3> for Poly<3> {
    fn split_coord(&mut self, out0: &mut Poly<3>, out1: &mut Poly<3>, coord: f64, ax: usize, sd: &mut Vec<f64>, side: &mut Vec<i32>, map: &mut Vec<i32>) { Poly::<3>::split_coord(self, out0, out1, coord, ax, sd, side, map) }
    fn moments(&self, order: usize, out: &mut [f64]) { Poly::<3>::moments(self, order, out) }
}
