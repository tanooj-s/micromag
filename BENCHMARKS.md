# BENCHMARKS

Published numbers we compare against, with our results. They are kept here so the source and the conditions travel with them. The decisions behind each check are in `DECISIONS.md` (D38).

### µMAG Standard Problem 3: single-domain limit of a cube
Source: https://www.ctcms.nist.gov/~rdm/results3.html (fetched 2026-10-17). L is the cube edge where the flower and vortex energies are equal, in units of l_ex = √(A/K_m), with K_m = µ₀Ms²/2 and K_u = 0.1 K_m along one cube axis. e is the reduced energy E/(K_m L³).

| Submission | Grid | L / l_ex | e at L |
|---|---|---|---|
| Rave, Hubert & Fabian (1998) | not stated | 8.47 | 0.3027 |
| Martins et al. (1998) | 10³ | 8.4073 | 0.2937 |
| Martins et al. | extrapolated from 10³–70³ | 8.4687 | 0.3026 |
| Hertel & Kronmüller (1998), symmetric flower (metastable) | not stated | 8.52 | 0.3049 |
| Hertel & Kronmüller, twisted flower | not stated | 8.57 | 0.3032 |
| **Bjørk, Poulsen & Insinga (2021), MagTense** | **20³** | **8.4556** | 0.3029 |
| Bjørk et al., MagTense | power-law fit, 5³–20³ | 8.477 ± 0.007 | — |

Rave et al.'s partial energies at L = 8.47, in units of K_m L³:

| State | Demag | Exchange | Anisotropy |
|---|---|---|---|
| Flower | 0.2794 | 0.0177 | 0.0056 |
| Vortex | 0.0783 | 0.1723 | 0.0521 |

**Ours (iteration 4, 20³):** L = 8.4551 (MagTense 20³: 8.4556). Flower 0.2797/0.0176/0.0056, total 0.3029; vortex 0.0783/0.1725/0.0520, total 0.3028 (demag/exchange/anisotropy, anisotropy in Rave's convention).

**Ours (iteration 5, GPU, Tesla T4):** identical to the CPU to the printed digits: L = 8.4551, same partial energies.

Coarse grids bias L low (10³ gives about 0.7% low). Hertel's twisted flower is a different, lower-energy state, so it is not comparable to a symmetric-flower computation.

### Stoner–Wohlfarth astroid
The analytic switching field of a single-domain particle with uniaxial anisotropy, with the field at angle ψ to the easy axis: h_sw = (cos^{2/3}ψ + sin^{2/3}ψ)^{−3/2}, in units of H_K = 2K/(µ₀Ms).

**Ours (iteration 4):** one cubic cell, easy axis z, Ms = 8e5 A/m, K = 5e5 J/m³. The field is swept from 0 to −1.2 H_K in steps of 10⁻³ H_K, and the switch is placed at the midpoint of the step with the largest |Δm|. Measured on 2026-10-17.

| ψ | h_sw analytic | h_sw ours | error | max minimizer iterations per field step |
|---|---|---|---|---|
| 15° | 0.6147 | 0.6145 | −2×10⁻⁴ | 163 |
| 30° | 0.5240 | 0.5245 | +5×10⁻⁴ | 151 |
| 45° | 0.5000 | 0.5005 | +5×10⁻⁴ | 139 |
| 60° | 0.5240 | 0.5245 | +5×10⁻⁴ | 128 |
| 75° | 0.6147 | 0.6145 | −2×10⁻⁴ | 112 |

Every error is within one field step, which is the resolution of the sweep. 0° and 90° are excluded, because there the reversed state is an exact (unstable) equilibrium (D23).

### µMAG Standard Problem 4, field 1
Source: OOMMF 1.2 (Donahue & Porter), 1 nm cells. The trimmed time series is `tests/data/sp4_field1_oommf.csv`.
- S-state ⟨m⟩ = (0.9666, 0.1260, 0.0000).
- First ⟨mx⟩ = 0 crossing at 138.4 ps (interpolated from the 1 ps samples).
- Ours (iteration 3, 128×32×1): S-state ⟨m⟩ = (0.9670, 0.1253, 0.0000); crossing at 138.6 ps; RMS ⟨m⟩ difference over 0–1 ns = 0.0051.
- Ours (iteration 5, GPU, Tesla T4): identical to the CPU, including the step count (1979 steps, 1 rejected) and the 2.80 ns S-state relaxation.

### µMAG Standard Problem 2: coercivity and remanence of a thin film vs size
Source: https://www.ctcms.nist.gov/~rdm/std2/spec2.html and `results.html` (fetched 2026-10-18).
- **Specification:**
  - geometry: a film with t/d = 0.1 and L/d = 5
  - energies: exchange and magnetostatic only (K = 0); l_ex = √(A/K_m) with K_m = µ₀Ms²/2
  - procedure: the field along [1,1,1]; report H_c/Ms, where m·[1,1,1] = 0, and the remanence ⟨m⟩ at H = 0, against d/l_ex
  - expected behaviour: the film switches as one rigid spin near d/l_ex ≈ 0.1, rotates non-uniformly around 2, and forms vortices above that
- **Exact small-particle limit:** H_c/Ms = 0.05707; switching field 0.05714.
- **OOMMF 1.1** (Donahue et al., cells of 0.03–0.6 l_ex; 0.60 l_ex at d/l_ex = 30, per the results page, checked 2026-10-20), selected rows:

| d/l_ex | H_c/Ms | switching field / Ms | mx | my |
|---|---|---|---|---|
| 1 | 0.05704 | 0.05716 | 1.0000 | 0.0000 |
| 10 | 0.05454 | 0.05562 | 0.9989 | 0.0000 |
| 20 | 0.04991 | 0.04991 | 0.9701 | 0.0819 |
| 30 | 0.04594 | 0.04594 | 0.9627 | 0.0756 |

- **MagTense** (Bjørk et al., 100×20×3 cells at every size): H_c/Ms = 0.05705 at d/l_ex = 3.00, 0.04938 at 20.36, 0.04558 at 30.00. It agrees with OOMMF 1.1 to about 0.8%.
- **The older OOMMF 1.0 submission** is off in the small-particle limit because of its demag model (Donahue's note).
- **Ours (iteration 6b, cell = min(0.5 l_ex, t)):** H_c/Ms = 0.05704 at d/l_ex = 1 and 0.05446 at 10 on both devices; remanence (1.0000, 0.0000) and (0.9989, 0.0000). On the GPU (Colab T4): 0.04960 at 20 (−0.6%, remanence 0.9701, 0.0820) and 0.04533 at 30 (−1.3% from OOMMF 1.1, −0.55% from MagTense; remanence 0.9626, 0.0756). The d/l_ex = 30 miss is open: our cells are 0.5 l_ex, OOMMF's 0.60 (D32).

### Nd₂Fe₁₄B cubes: switching and coercive fields
- **Pure 40 nm cube** (Fischbacher et al., APL 111, 072404 (2017), arXiv:1708.01880):
  - setup: finite elements, 1.5 nm mesh, T = 300 K; µ₀Ms = 1.61 T, K = 4.30 MJ/m³, A = 7.7 pJ/m; field one degree off the easy axis, which is parallel to one edge
  - results: static switching field µ₀H₀ = 5.29 T, and 6.09 T without magnetostatics (= Stoner–Wohlfarth at 1°)
- **Core–shell cube, Fig. 8a of the review** (finite elements):
  - setup: core K = 4.9 MJ/m³, µ₀Ms = 1.61 T, A = 8 pJ/m; a 3 nm shell with K = 0, µ₀Ms = 0.477 T, A = 6.12 pJ/m. The field direction is not stated.
  - results, read off the figure at 600 dpi on the finest mesh (1.3 nm): µ₀H_c ≈ 2.76 T at 25 nm, 2.71 T at 50 nm, 2.58 T at 100 nm.
  - mesh dependence: the 4 nm mesh sits up to 0.13 T higher, and the curve falls by about 0.002 T per nm.
- **Ours (iteration 6b, field 1° off z, the same on CPU and GPU):** the pure 40 nm cube (cells of 1.33 nm) switches at 5.275 T (−0.3%); the 25 nm core–shell cube (1 nm cells) has H_c = 2.721 T (Fig. 8a: 2.76 T); the 50 nm one (GPU only) 2.675 T (Fig. 8a: 2.71 T).

### Stoner–Wohlfarth ensemble of randomly oriented particles
- **Stoner & Wohlfarth (1948)**, non-interacting uniaxial particles with isotropic easy axes: remanence M_r = 0.5 M_s and coercivity H_c = 0.479 H_K (H_K = 2K/(µ₀M_s)).
- **Ours (iteration 6c, CPU):** 255 decoupled Voronoi grains without demag give M_r = 0.4705 M_s and H_c = 0.4627 H_K. Both are within 3σ of the isotropic values for this finite draw of axes, and both agree with the per-axis oracle to 10⁻⁷ (D37).
