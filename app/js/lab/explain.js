// explain.js - the "Explain" section: short pedagogic notes in collapsible boxes.

import { el } from './dom.js';

const NOTES = [
  {
    title: 'What is the phase-space sheet?',
    open: true,
    html: `Cold dark matter occupies a thin three-dimensional sheet in six-dimensional phase space (in 2D mode: a two-dimensional sheet in four-dimensional phase space).
      Every Lagrangian label <b>q</b> is carried to <b>x</b>(q, D) = q + ΣₙDⁿ Ψ⁽ⁿ⁾(q), where the displacement terms Ψ⁽ⁿ⁾ are built recursively from the initial field.
      The density is the sum of 1/|J| over all points of the sheet that project onto the same <b>x</b>, with J = det ∂x/∂q. Where J reaches zero the sheet folds over itself (shell crossing) and the density has caustics; beyond it several streams overlap.
      In 3D the sheet density is evaluated exactly the same way (on the GPU, one thread per Lagrangian cell, or in WASM): every Lagrangian cube is cut into six Kuhn tetrahedra, each carrying mass Δq³/6, and every Eulerian sample point inside a deformed tetrahedron receives its mass over its volume (summed over streams); it is point-sampled (about 10 ms at 128³ on the GPU, 1 s at 64³ in WASM), whereas CIC deposits the particles and blurs the result.
      The GPU panel draws every triangle of the deformed grid with additive blending, so overlapping streams add up exactly as they should; the “wire” view shows the folds themselves.
      The growth factor D plays the role of time; with σ0 = 1, D is the rms of the linear density contrast.`,
  },
  {
    title: 'Why is Hopf–Cole equal to Zel’dovich before shell crossing?',
    html: `With D as time, the Zel’dovich velocity u = Ψ⁽¹⁾ = −∇ϕ obeys the inviscid Burgers equation ∂_D u + (u·∇)u = 0. Adding a small viscosity ν gives the adhesion model.
      The Hopf–Cole substitution ψ = exp(−Φ/2ν), u = ∇Φ turns the nonlinear equation into the linear heat equation ∂_Dψ = ν∇²ψ, which can be solved exactly (a Fourier multiplier exp(−νk²D), or a convolution with a Gaussian).
      As ν → 0 the solution reduces to the Hopf–Lax (Lagrangian minimum) formula, and as long as the Lagrangian map q ↦ x is one-to-one, that is the Zel’dovich map. So for LPT order 1, D below D_sc(1) and small ν, the sheet and Hopf–Cole density panels and their spectra coincide.
      After shell crossing the sheet keeps folding and multi-stream regions appear, whereas the Hopf–Cole solution glues the streams into shocks of width ∼√(νD).
      Higher LPT orders improve the sheet before D_sc but do not change what happens at the singularity.
      Lowering Ω_m below one (flat ΛCDM) changes only the time dependence: the growth functions of every LPT term, including the linear D(a), are integrated numerically from the exact equations instead of being the pure powers Dⁿ of Einstein–de Sitter, so the readout g_τ(D)/Dⁿ departs from the EdS constants (−3/7, …) as D grows; the Hopf–Cole solution depends on D only and is unchanged.`,
  },
  {
    title: 'Legendre transform: Hopf–Cole for the exact nLPT density',
    html: `The Hopf–Cole / Hopf–Lax construction inverts any <i>gradient</i> Lagrangian map: for x = q + ∇S(q), the inverse map is the gradient of the Legendre transform of q²/2 + S(q), i.e. Φ(x) = min_q [S(q) + |x−q|²/2] (and its heat-kernel smoothing at finite ν), and the Eulerian density is 1+δ = det(I − ∇∇Φ).
      The Zel’dovich approximation is the case S = −Dϕ, where this is also the Burgers equation in D. Replacing S by the longitudinal displacement potential of nLPT gives the exact nLPT Eulerian density before shell crossing, particle-free and on the Eulerian grid; afterwards the Legendre transform keeps one stream per point (the “adhesion” selection) whereas the sheet sums all streams.
      Through 2LPT the displacement is exactly a gradient; at 3LPT and 4LPT a small transverse part Ψ_T appears (rms ratio ∼10⁻³), which can be corrected to first order in the inverse map, q(x) ≈ q_L(x) − Ψ_T(q_L(x)). Switch on <i>Legendre lab</i> and choose the “HC source ϕ” under Dynamics to compare the Legendre density with the sheet and with the Zel’dovich Hopf–Cole density: the residual against the sheet before shell crossing is the sheet’s own rasterization noise, which shrinks with N.`,
  },
  {
    title: 'The Fourier method and its dynamic range',
    html: `The multiplier solution evaluates exp(−Φ/2ν) on the grid and then multiplies its transform by exp(−νk²D). For small ν the exponent range (Φmax − Φmin)/2ν is enormous (the readout “exponent range”), and double precision cannot hold e^{range}.
      The Fourier method therefore limits the range to max_exp by raising the effective viscosity ν_eff, which is why its result departs from the requested ν. The log-domain kernel (×1, ×2, ×4 refinement) evaluates the same heat-kernel convolution with log-sum-exp and remains accurate down to a grid floor of order Δx²/(4D), refined by the upsampling factor.`,
  },
  {
    title: 'Mode coupling with two plane waves',
    html: `A linear field made of two plane waves has power only at k₁ and k₂. In the linear panel the Fourier amplitude map shows just those two modes (and their mirror images).
      Nonlinear evolution multiplies fields, so new modes appear at integer combinations m·k₁ + n·k₂: harmonics 2k₁, 3k₁, …, and the sum and difference k₁ ± k₂. In perturbation theory the combination (m, n) first appears at order |m| + |n| and its amplitude grows like D^(|m|+|n|), which you can see by increasing D or the LPT order.
      The phase map shows the other side of the coupling: the phase of a generated mode is fixed by the phases of its parents, m φ₁ + n φ₂ (plus a constant), so phases of small-scale modes become correlated with the large-scale ones.
      If k₁ and k₂ are parallel the dynamics is one-dimensional and the Zel’dovich solution is exact until the first shell crossing.`,
  },
  {
    title: 'What does the EFT counterterm do?',
    html: `Standard perturbation theory treats matter as a pressureless fluid and has no description of shell-crossed regions. The one-loop integrals P22 and P13 also receive contributions from small-scale modes, where the theory is invalid; this makes them sensitive to the smoothing R.
      In the effective field theory (EFT) approach the unknown small-scale physics enters through an effective stress tensor, which on large scales contributes a term proportional to k²P_lin(k). It is parametrised by one coefficient c_s²: P_EFT = P_lin + P22 + P13 − 2c_s²k²P_lin.
      Here c_s² is obtained by least squares from the difference between the measured spectrum (sheet in 2D, CIC in 3D) and the 1-loop SPT curve for k < k_max. If the loop and the counterterm are working, the EFT curve follows the data to larger k than 1-loop alone; changing R moves the loop and the fitted c_s² together.`,
  },
  {
    title: 'How to read the Fourier maps and r(k)',
    html: `Fourier maps show k_x horizontally and k_y vertically with k = 0 at the centre; the amplitude is log₁₀|δ̂|/max and the phase is arg δ̂. Hover over a map to read the integer mode numbers.
      The cross-correlation coefficient r(k) between an evolved field and the linear field is 1 when the evolved field is a pure rescaling of the linear one in that mode and falls as nonlinear evolution scrambles its phases. For the Zel’dovich map of a Gaussian field the decay is approximately exp(−k²σ_Ψ²D²/2), where σ_Ψ² is the variance of one component of the displacement.`,
  },
];

export function buildExplain(host) {
  const root = el('div', 'lab-explain', host);
  for (const n of NOTES) {
    const d = el('details', 'lab-note', root);
    if (n.open) d.open = true;
    el('summary', null, d, n.title);
    const p = el('p', null, d);
    p.innerHTML = n.html.replace(/\s+/g, ' ');
  }
  return root;
}
