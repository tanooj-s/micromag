"""Energy barriers by geodesic NEB with a climbing image (D48), moved by L-BFGS over the whole chain (D49).

A path is a chain of images, each a full state m. Images feel the true force across the path and a spring along it, and
the highest image climbs to the saddle. States live on a product of spheres, so distances are great-circle angles and
every step is a per-cell rotation. All vectors here are (image, component, cell) over the magnetic cells only.
"""

from dataclasses import dataclass

import numpy as np

from . import energy, mu0

HISTORY = 15     # L-BFGS pairs kept (D49 Correction)
MAX_ROTATION = 0.1  # rad: cap on any one cell's rotation in a step, in place of a line search (D49 Correction)
NOISE = 1e-2     # rad per cell on the first path's inner images: a uniform path can't leave itself (D48 Correction)
TRUNCATE_RATIO = 10   # shorten the path while its energy range exceeds this many barriers (D48 Correction)
TRUNCATE_EVERY = 100  # iterations between checks


@dataclass
class Path:
    m: np.ndarray       # (images, 3, nz, ny, nx)
    energy: np.ndarray  # J, per image
    s: np.ndarray       # rad: arc length along the path, 0 at the start
    climbing: int       # the image on the saddle
    iterations: int
    torque: float       # max |force| / Ms at the end
    resets: int         # L-BFGS restarts, the climbing image's changes included
    truncations: int    # times the far end moved in towards the saddle

    @property
    def barrier(self):
        return float(self.energy.max() - self.energy[0])


def _angle(a, b):
    """Great-circle angle per cell between unit vectors, component axis first; atan2 stays exact near 0 and π."""
    return np.arctan2(np.linalg.norm(np.cross(a, b, axis=0), axis=0), (a * b).sum(axis=0))


def distance(a, b, w):
    """L = √(Σ wᵢ θᵢ²), w the Ms weights (D48)."""
    return float(np.sqrt((w * _angle(a, b) ** 2).sum()))


def _slerp(a, b, t):
    """Each cell a fraction t of the way from a to b along its great circle, t of shape (k, 1, ...). An exactly antiparallel
    cell has no plane of its own, so it turns about a fixed axis perpendicular to it."""
    axis = np.cross(a, b, axis=0)
    size = np.linalg.norm(axis, axis=0)
    fixed = np.zeros_like(a)
    along = np.abs(a[1]) > 1 - 1e-9                  # ŷ × a vanishes on ŷ, so x̂ there
    fixed[0], fixed[1] = along, ~along
    side = np.cross(fixed, a, axis=0)
    side /= np.linalg.norm(side, axis=0)
    turned = np.cross(axis / np.where(size > 1e-12, size, 1.0), a, axis=0)
    side = np.where(size > 1e-12, turned, side)      # the unit vector in the plane, a quarter turn from a towards b
    theta = _angle(a, b)
    path = a * np.cos(t * theta) + side * np.sin(t * theta)
    return path / np.linalg.norm(path, axis=1, keepdims=True)


def great_circle(a, b, images):
    """The first path (D48): each cell turns from a to b at a constant rate."""
    path = _slerp(a, b, np.linspace(0, 1, images).reshape(-1, 1, *[1] * (a.ndim - 1)))
    path[0], path[-1] = a, b
    return path


def _along(m, s, targets):
    """States at arc lengths `targets` on the piecewise great-circle path through m, whose images sit at arc lengths s."""
    out = np.empty((len(targets), *m.shape[1:]))
    for i, t in enumerate(targets):
        k = min(max(int(np.searchsorted(s, t, side="right")) - 1, 0), len(s) - 2)
        out[i] = _slerp(m[k], m[k + 1], np.array([[[(t - s[k]) / max(s[k + 1] - s[k], 1e-300)]]]))[0]
    return out


def tangents(m, E, w):
    """Unit tangent at each inner image, zero at the ends (D48): the difference to the higher neighbour, blended by the
    energy differences at an extremum (Henkelman & Jónsson 2000), projected per cell and normalised over the state."""
    t = np.zeros_like(m)
    for i in range(1, len(m) - 1):
        up, down = m[i + 1] - m[i], m[i] - m[i - 1]
        dp, dm = E[i + 1] - E[i], E[i] - E[i - 1]
        if dp > 0 and dm > 0:
            v = up
        elif dp < 0 and dm < 0:
            v = down
        else:
            big, small = max(abs(dp), abs(dm)), min(abs(dp), abs(dm))
            v = big * up + small * down if E[i + 1] > E[i - 1] else small * up + big * down
        v = v - (v * m[i]).sum(axis=0) * m[i]
        size = np.sqrt((w * v * v).sum())
        t[i] = v / size if size > 0 else v
    return t


def _project(v, m):
    return v - (v * m).sum(axis=1, keepdims=True) * m


def _rotate(m, d):
    """Each cell turned along its tangent vector d by |d|: the exponential map on the sphere."""
    theta = np.linalg.norm(d, axis=1, keepdims=True)
    out = m * np.cos(theta) + d * np.sinc(theta / np.pi)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class _LBFGS:
    """Two-loop L-BFGS on force vectors, in the Ms-weighted metric. History is kept as plain 3-vectors per cell and
    projected onto the current tangent planes when used (D49)."""

    def __init__(self, w):
        self.w, self.pairs, self.gamma = w, [], None

    def dot(self, a, b):
        return float((self.w * a * b).sum())

    def clear(self):
        self.pairs.clear()

    def add(self, s, y):
        if self.dot(s, y) > 0:  # no curvature along this step: skip it
            self.pairs.append((s, y))
            del self.pairs[:-HISTORY]
            self.gamma = self.dot(s, y) / self.dot(y, y)

    def direction(self, f, m):
        q = -f
        pairs = [(_project(s, m), _project(y, m)) for s, y in self.pairs]
        pairs = [(s, y, 1 / self.dot(s, y)) for s, y in pairs if self.dot(s, y) > 0]
        alpha = []
        for s, y, rho in reversed(pairs):
            alpha.append(rho * self.dot(s, q))
            q = q - alpha[-1] * y
        q = self.gamma * q
        for (s, y, rho), a in zip(pairs, reversed(alpha)):
            q = q + (a - rho * self.dot(y, q)) * s
        return -q


def gneb(prob, grid, start, end, images=20, torque_tol=1e-4, max_iter=100_000, seed=0):
    """The minimum energy path from start to end (both minima, kept fixed) at prob.H_ext. Raises at max_iter. seed keys
    the first path's noise."""
    Ms = prob.Ms
    cells = Ms > 0
    w = (Ms[cells] / Ms[cells].sum())[None, None, :]
    size = float(grid.dx * grid.dy * grid.dz * mu0 * Ms[cells].sum())  # force in J/rad = size × force in A/m (D48)
    H_ext = np.array(prob.H_ext).reshape(3, 1, 1, 1)
    full, H = np.array(start), np.empty_like(start)

    def field(state):
        """(m × (m × H) up to sign, E) of one state on the magnetic cells: E is -½ µ0 dV Σ Ms m·(H + H_ext), since every
        term but Zeeman is quadratic in m (D10), so a step needs one field evaluation per image."""
        full[:, cells] = state
        prob.effective_field(full, H)
        return H[:, cells], energy(grid, full, Ms, H + H_ext, 0.5)

    m = great_circle(start, end, images)[:, :, cells]
    n = images - 1
    m[1:n] = _rotate(m[1:n], _project(NOISE * np.random.default_rng(seed).normal(size=m[1:n].shape), m[1:n]))
    E = np.zeros(images)
    F = np.zeros_like(m)  # the true field at each image
    _, E[0] = field(m[0])
    _, E[-1] = field(m[-1])

    def evaluate(m):
        for i in range(1, n):
            F[i], E[i] = field(m[i])
        return _project(F, m)

    def force(m, perp, climbing):
        """The GNEB force in A/m: the true force across the path, the spring along it; inverted along it on the climbing image."""
        t = tangents(m, E, w)
        gaps = np.array([distance(m[i], m[i + 1], w[0]) for i in range(n)])
        kappa = 10 * n * (E.max() - E.min()) / max(gaps.sum() ** 2, 1e-300)  # J/rad², D48
        out = np.zeros_like(m)
        for i in range(1, n):
            along = float((w * perp[i] * t[i]).sum())
            if i == climbing:
                out[i] = perp[i] - 2 * along * t[i]
            else:
                out[i] = perp[i] - along * t[i] + kappa / size * (gaps[i] - gaps[i - 1]) * t[i]
        return out, gaps

    def truncate(gaps):
        """Moves the far end to the first image lower than the start, and re-spaces the images between (D48 Correction): the
        path has crossed its first pass by then, and a 25 k_BT pass on a 10⁵ k_BT slope would otherwise fit between two
        images. Returns False if the energy range is already comparable to the barrier, or no image can serve as the end."""
        past = 1 + np.flatnonzero(E[1:] < E[0])
        if E.max() - E.min() <= TRUNCATE_RATIO * (E.max() - E[0]) or len(past) == 0 or past[0] == n:
            return False
        s = np.concatenate([[0.0], np.cumsum(gaps)])
        first, last = m[0].copy(), m[past[0]].copy()
        m[:] = _along(m, s, np.linspace(0, s[past[0]], images))
        m[0], m[-1] = first, last
        _, E[-1] = field(last)
        return True

    lbfgs = _LBFGS(w)
    inner = slice(1, n)
    climbing, resets, truncations, previous = None, 0, 0, None
    for iteration in range(max_iter + 1):
        perp = evaluate(m)
        top = 1 + int(np.argmax(E[1:-1]))
        if climbing is not None and top != climbing:
            climbing = top
            lbfgs.clear()
            resets += 1
            previous = None
        f, gaps = force(m, perp, climbing)
        torque = float((np.linalg.norm(f[inner], axis=1) / Ms[cells]).max())
        if (torque < 10 * torque_tol or (iteration + 1) % TRUNCATE_EVERY == 0) and truncate(gaps):
            truncations += 1
            climbing = None
            lbfgs.clear()
            previous = None
            continue
        if climbing is None and torque < 10 * torque_tol:
            climbing = top
            lbfgs.clear()
            previous = None
            f, gaps = force(m, perp, climbing)
            torque = float((np.linalg.norm(f[inner], axis=1) / Ms[cells]).max())
        if climbing is not None and torque < torque_tol:
            break
        if iteration == max_iter:
            raise RuntimeError(f"GNEB: max_iter = {max_iter} reached with max |force|/Ms = {torque:.2e} > torque_tol = {torque_tol:g}")

        v, x = f[inner], m[inner]
        if previous is not None:
            step, f_old = previous
            lbfgs.add(step, f_old - v)
        if lbfgs.gamma is None:  # the first step: a small fraction of the cap
            d = v * (0.1 * MAX_ROTATION / max(np.sqrt(lbfgs.dot(v, v) / len(v)), 1e-300))
            lbfgs.gamma = 0.1 * MAX_ROTATION / max(np.sqrt(lbfgs.dot(v, v) / len(v)), 1e-300)
        else:
            d = lbfgs.direction(v, x)
            if lbfgs.dot(d, v) <= 0:  # against the force: no gradient to fall back on, so start over (D49)
                lbfgs.clear()
                resets += 1
                d = lbfgs.gamma * v
        largest = float(np.linalg.norm(d, axis=1).max())
        if largest > MAX_ROTATION:
            d = d * (MAX_ROTATION / largest)
        d = _project(d, x)
        m[inner] = _rotate(x, d)
        previous = (d, v)

    out = np.repeat(np.asarray(start)[None], images, axis=0)
    out[:, :, cells] = m
    return Path(out, E.copy(), np.concatenate([[0.0], np.cumsum(gaps)]), climbing, iteration, torque, resets, truncations)
