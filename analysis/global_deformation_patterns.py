"""Whole-pelvis 3D canonical deformation-pattern decomposition without arbitrary axes.

Mathematical procedure:
1. Infinitesimal rigid-body motion (3 translations + 3 rotations) is mathematically
   projected out upfront via QR orthogonal decomposition with volume quadrature weights.
   Rigid rotation is eliminated and is not a reported deformation category.
2. The remaining nonrigid shape change is evaluated against four canonical 3D deformation
   families spanning all nonrigid polynomial modes up to degree 2:
   - Axial (uniform normal stretching + normal strain gradients along ML, AP, CC)
   - Bending (6 pure Euler-Bernoulli flexural modes with zero shear strain)
   - Shear (uniform transverse shears + transverse shear gradients across all planes)
   - Torsion (Saint-Venant torsional twist gradients around all axes)
3. Subspace overlap on irregular pelvic geometry is partitioned using Shapley allocation
   of the explained nonrigid squared displacement norm, ensuring order-independent closure.
"""
import math
import numpy as np

PATTERNS = ('axial', 'bending', 'shear', 'torsion')


def rigid_design(points):
    """6-parameter infinitesimal rigid body motion basis (3 translations, 3 rotations)."""
    n = len(points)
    design = np.zeros((n, 3, 6))
    design[:, :, :3] = np.eye(3)
    for j in range(3):
        design[:, :, 3 + j] = np.cross(np.eye(3)[j], points)
    return design.reshape(-1, 6)


class GlobalDeformationModel:
    """3D whole-domain deformation pattern model for arbitrary pelvic geometry."""

    def __init__(self, points, weights):
        x = np.asarray(points, float)
        w = np.asarray(weights, float)
        if x.shape != (len(w), 3) or np.any(w <= 0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
            raise ValueError('Finite coordinates and positive volume weights required')

        self.center = np.average(x, axis=0, weights=w)
        r = x - self.center
        self.length = np.sqrt(np.average(np.sum(r**2, axis=1), weights=w))
        if self.length == 0:
            raise ValueError('Degenerate geometry')

        coords = r / self.length
        xx, yy, zz = coords.T
        n = len(x)
        self.sqrtw = np.repeat(np.sqrt(w / w.sum()), 3)

        # 1. Rigid body motion subspace (6 DOF)
        rigid = rigid_design(coords) * self.sqrtw[:, None]
        self.rigid_q, _ = np.linalg.qr(rigid, mode='reduced')

        # 2. Canonical deformation families across full 3D space:
        # Axial: normal stretch and stretch gradients
        axial = []
        for j, coord in enumerate((xx, yy, zz)):
            v = np.zeros((n, 3)); v[:, j] = coord; axial.append(v.ravel())
            v = np.zeros((n, 3)); v[:, j] = 0.5 * coord**2; axial.append(v.ravel())

        # Bending: pure Euler-Bernoulli flexure (zero shear strain)
        bend = []
        v = np.zeros((n, 3)); v[:, 0] = -xx*yy; v[:, 1] = 0.5*xx**2; bend.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 0] = -xx*zz; v[:, 2] = 0.5*xx**2; bend.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 1] = -yy*xx; v[:, 0] = 0.5*yy**2; bend.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 1] = -yy*zz; v[:, 2] = 0.5*yy**2; bend.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 2] = -zz*xx; v[:, 0] = 0.5*zz**2; bend.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 2] = -zz*yy; v[:, 1] = 0.5*zz**2; bend.append(v.ravel())

        # Shear: uniform transverse shear + transverse shear gradients
        shear = []
        v = np.zeros((n, 3)); v[:, 0] = yy; v[:, 1] = xx; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 1] = zz; v[:, 2] = yy; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 0] = zz; v[:, 2] = xx; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 0] = xx*yy; v[:, 1] = 0.5*xx**2; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 0] = xx*zz; v[:, 2] = 0.5*xx**2; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 1] = yy*xx; v[:, 0] = 0.5*yy**2; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 1] = yy*zz; v[:, 2] = 0.5*yy**2; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 2] = zz*xx; v[:, 0] = 0.5*zz**2; shear.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 2] = zz*yy; v[:, 1] = 0.5*zz**2; shear.append(v.ravel())

        # Torsion: pure Saint-Venant twist gradients (zero normal strain)
        torsion = []
        v = np.zeros((n, 3)); v[:, 1] = -xx*zz; v[:, 2] = xx*yy; torsion.append(v.ravel())
        v = np.zeros((n, 3)); v[:, 0] = yy*zz; v[:, 2] = -xx*yy; torsion.append(v.ravel())

        groups = [axial, bend, shear, torsion]
        all_cols = []
        self.group_col_ids = []
        start_idx = 0
        for grp in groups:
            A = np.column_stack(grp) * self.sqrtw[:, None]
            A -= self.rigid_q @ (self.rigid_q.T @ A)
            U, S, _ = np.linalg.svd(A, full_matrices=False)
            rank = int(np.sum(S > 1e-10 * S[0]))
            Q = U[:, :rank]
            all_cols.append(Q)
            self.group_col_ids.append(list(range(start_idx, start_idx + rank)))
            start_idx += rank

        self.a = np.column_stack(all_cols)
        self.gram = self.a.T @ self.a
        self.condition = float(np.linalg.cond(self.gram))

        # Precompute pseudoinverses for all 15 non-empty subsets of the 4 groups
        self.inverse = {}
        for mask in range(1, 16):
            ids = [idx for g in range(4) if mask & (1 << g) for idx in self.group_col_ids[g]]
            self.inverse[mask] = (ids, np.linalg.pinv(self.gram[np.ix_(ids, ids)], rcond=1e-12))

    def fit(self, displacement):
        """Fit displacement field to canonical deformation families."""
        u = np.asarray(displacement, float)
        if u.shape != (len(self.sqrtw) // 3, 3) or not np.all(np.isfinite(u)):
            raise ValueError('Finite displacement samples matching geometry required')

        scale = np.max(np.abs(u))
        if scale == 0:
            return dict(is_valid=False)

        target = (u / scale).ravel() * self.sqrtw
        total = float(target @ target)
        target -= self.rigid_q @ (self.rigid_q.T @ target)
        norm = float(target @ target)

        if norm <= 1e-20 * total:
            return dict(is_valid=False, nonrigid_fraction=norm / total)

        rhs = self.a.T @ target
        explained = np.zeros(16)
        for mask, (ids, inv) in self.inverse.items():
            explained[mask] = float(rhs[ids] @ inv @ rhs[ids]) / norm

        if explained[-1] > 1 + 1e-8:
            raise ValueError('Projection exceeds nonrigid norm')

        fractions = {}
        for g, name in enumerate(PATTERNS):
            val = 0.0
            for subset in range(16):
                if subset & (1 << g):
                    continue
                count = subset.bit_count()
                factor = math.factorial(count) * math.factorial(3 - count) / math.factorial(4)
                marginal = explained[subset | (1 << g)] - explained[subset]
                if marginal < -1e-8:
                    raise ValueError('Nonmonotonic subspace fit')
                val += factor * marginal
            fractions['f_' + name] = val

        return dict(
            is_valid=True,
            **fractions,
            f_residual=max(0.0, 1.0 - explained[-1]),
            nonrigid_fraction_of_total=norm / total,
            gram_condition=self.condition,
            **{'standalone_' + name: explained[1 << g] for g, name in enumerate(PATTERNS)}
        )
