# Thermal runs

An `llg-dynamics` stage with `temperature > 0` adds Brown's thermal field to the effective field. This page explains what that field is, how it is integrated, and what it does **not** account for. The card keys are in [cards.md](cards.md#type--llg-dynamics); the decisions are D41–D44 in [DECISIONS.md](../DECISIONS.md).

## The physics

Each cell's magnetisation is like a compass needle in honey. The honey is the damping α: it slows the needle down until it settles along its field. Heat is like someone randomly shaking the table, and Brown (1963) modelled the shaking as an extra random field H_th, added to H_eff in the Landau–Lifshitz–Gilbert equation (eq. 25).

How hard to shake is not a free choice. The kicks must keep the needles exactly as wobbly as the temperature says (the fluctuation–dissipation theorem), which fixes the variance per cell, per component and per time step:

σ² = 2 α k_B T / (µ0 γ0 Ms V Δt), with γ0 = µ0 |γ| and V the cell volume.

Thicker honey (larger α) needs harder shaking; a larger moment (Ms V) moves less for the same kick; a shorter step needs a larger field, because white noise averaged over a shorter time is larger. The kicks are independent between cells, components and steps, and vacuum cells get none. There is no 1+α² factor: the noise enters H in Gilbert's form, as in MuMax3.

The random numbers come from a counter-based Philox generator keyed by the stage's `seed` and its position in the card, so a run is reproducible, and the CPU and GPU draw the same numbers.

## The integrators

The adaptive RK45 used at T = 0 can't take noise: its error estimate assumes a smooth equation. At T > 0 there are two methods:

- **`heun`** (the default): a fixed step `dt`. Each step draws one kick and holds it over the predictor and the corrector, which makes the scheme converge to Brown's (Stratonovich) equation.
- **`rk23`**: adaptive Bogacki–Shampine steps with the kick frozen over each step. When a step is rejected, its kick is not thrown away, since that would favour steps where the noise happened to be small. A Brownian bridge splits it exactly between the shorter retry and the rest, and the rest is used by later steps. `max_bridge_depth` in the provenance records how many pieces were waiting at most. This matters most for small systems: a step is rejected when its worst cell is, so on a large grid a rejection says little about any one cell's noise, while a lone macrospin that redrew its noise on rejection would come out far too cold (D44).

**Choosing `dt` for `heun`.** It must resolve the fastest precession: a small fraction (about 1/20 or less) of 2π / (γ0 H_max), where H_max includes the exchange field, roughly 2A / (µ0 Ms Δx²) per neighbour. It should also be well below a cell's free-diffusion time τ_N = (1+α²) µ0 Ms V / (2 α γ0 k_B T); the tests use τ_N / 200. Check a result by halving `dt` once.

**What is within reach.** The step must resolve precession on femtosecond to picosecond scales, so direct runs reach nanoseconds to microseconds. Thermally activated reversal over a barrier of 25 k_B T, which takes about a second (the paper's coercivity criterion), is far out of reach: use barriers of a few k_B T. Energy-barrier methods (the string method) are optional roadmap iteration 8.

## Caveats

> [!WARNING]
> **Some fluctuations are counted twice.** Material parameters measured at temperature, such as Ms(300 K), A(300 K) and K(300 K), already contain the average effect of the atoms' small-scale jiggling. Adding the thermal field on top shakes the magnet again. It is like a recipe that says "use salted butter" when you also add the full amount of salt. Neither choice is exactly right: 0 K parameters with noise miss the fluctuations shorter than a cell, and 300 K parameters with noise count part of them twice. micromag does not correct for this; choose your parameters knowing it.

> [!WARNING]
> **Results depend on the cell size.** Each cell gets its own independent kick, so smaller cells carry shorter-wavelength fluctuations and the magnet looks somewhat "more thermal". A real magnet doesn't care about our mesh. There is no settled correction (cell-size renormalisation is research-level); this is an open question for the project (NOTES.md). Report the cell size with any thermal result, and compare at least two.

- **One temperature per stage.** T is the same in every cell; there is no per-cell temperature (hot spots, gradients). Change it between stages.
- **Thermal settings are only on `llg-dynamics`.** `minimize` and `hysteresis` stay at T = 0, and `until_torque` is refused at T > 0, since the torque never settles under noise.

## How it is checked

See D44 and [verification.md](verification.md): Philox against its published test vectors; the noise's mean, variance and independence; second- and third-order convergence of `heun` and `rk23` at T = 0; an ensemble of independent macrospins against the Boltzmann distribution and the Langevin curve, at two damping values; a single macrospin under heavy step rejection, and the bridge's exact Wiener increments; the escape rate over a 3 k_B T barrier against the Fokker–Planck operator; and CPU against CUDA.
