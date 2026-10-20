# The results folder

`micromag run CARD -o DIR` writes everything about one run into `DIR`, which must be new or empty. A non-empty folder is refused, so a result is never overwritten. The layout borrows from LAMMPS: a thermo table per stage, and one dump file. Source: [`python/micromag/run.py`](../python/micromag/run.py); design: D28 in [DECISIONS.md](../DECISIONS.md).

```
DIR/
├── card.toml                  # the card, copied byte for byte
├── provenance.json            # versions, git state, timings, status, per-stage summaries
├── git.diff                   # only if the checkout had uncommitted changes
├── stage1_llg-dynamics.csv    # one table per stage: stage<k>_<type>.csv
├── stage2_minimize.csv
└── trajectory.h5              # magnetisation snapshots and the region map
```

## Stage tables: `stage<k>_<type>.csv`

One row per sample, flushed as it is written, so `tail -f` works while a run is going. Floats are written with `repr`, so a value read back is exactly the value computed. Time restarts at 0 in each stage.

| Stage type | Columns |
|---|---|
| `llg-dynamics` | `t, mu0_Hx, mu0_Hy, mu0_Hz, mx, my, mz, E_exchange, E_anisotropy, E_zeeman, E_demag, E_total, steps, rejected` |
| `minimize`, `hysteresis` | `mu0_Hx, mu0_Hy, mu0_Hz, mx, my, mz, E_exchange, E_anisotropy, E_zeeman, E_demag, E_total, iterations` |

- `t` is in s and `mu0_H*` in T. The field in each row is the one that row was computed at.
- `mx, my, mz` is ⟨m⟩ averaged over magnetic cells (Ms > 0).
- Energies are in J.
- `steps`, `rejected` and `iterations` count cumulatively within the stage.
- A refined `hysteresis` stage revisits fields: every field it computed is a row, in the order computed.

## `trajectory.h5`

float64 throughout, so restarts are exact, compressed losslessly with shuffle + gzip.

| Path | Contents |
|---|---|
| `/` attributes | `n`, `cell` (m), units, and the `m` layout: `m[snapshot, component, z, y, x]`, x fastest |
| `/regions` | the region map, `uint8 (nz, ny, nx)`; attribute `materials` is JSON `{"id": "material name"}` |
| `/stage<k>/m` | snapshots, `(count, 3, nz, ny, nx)` unit vectors; attribute `type` is the stage type |
| `/stage<k>/t` or `/stage<k>/mu0_H` | the time (s) or field (T) of each snapshot |
| `/final/m` | the state at the end of the run |
| `/failed/m` | the last state, if a stage failed |

By default, each stage saves one snapshot at its end. `dump_every` (`llg-dynamics`) or `dump = true` (`hysteresis`) saves more.

```python
import h5py
with h5py.File("runs/sp4/trajectory.h5") as f:
    m = f["final/m"][()]            # (3, nz, ny, nx)
    states = f["stage2/m"]          # read snapshots lazily
    regions = f["regions"][()]
```

A later card can start from any saved state:

```toml
[initial]
file  = "../runs/sp4/trajectory.h5"
state = "stage1"                    # or "final" (the default)
```

## `provenance.json`

| Field | Meaning |
|---|---|
| `card` | the card's file name |
| `status`, `error` | `"running"`, `"completed"` or `"failed"`, with the error message |
| `started`, `finished` | UTC timestamps |
| `device`, `gpu` | `"cpu"` or `"cuda"`, and the GPU name from `nvidia-smi` |
| `versions` | micromag, numpy, h5py and Python |
| `git` | `{hash, dirty}` of the checkout, or `null` outside one |
| `initial` | the uniform `m`, or the restart file, state and SHA-256 |
| `regions` | per `[[region]]`: `id`, `material`, `shape`, exact `volume_m3` (null for `all` and inverted shapes) and painted `cells_m3` |
| `stages` | per stage: `type`, `wall_s`, and for `hysteresis` a `summary` with `Hc_T`, `switch_T` and `remanence` |

The file is rewritten after each stage, so a killed run still shows how far it got.

## Failures

A stage fails if `max_duration` or `max_iter` is reached, or on a solver error. The run then stops, keeps everything written so far, saves the last state in `/failed`, records `status: "failed"` with the error, and exits with code 1. A bad card or a non-empty output folder exits with code 2 before anything is written.
