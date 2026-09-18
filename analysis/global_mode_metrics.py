"""Volume-weighted whole-domain motion descriptors, independent of bone labels."""
import numpy as np


def global_motion_profile(points, displacement, weights):
    """Orthogonal translation + best infinitesimal rotation + nonrigid remainder.

    Input samples may be tetrahedral quadrature points. Fractions describe squared
    displacement, not strain or elastic energy. Axis position is not estimated:
    the direction is displayed through the integration centroid by convention.
    """
    x, u, w = (np.asarray(a, dtype=float) for a in (points, displacement, weights))
    if x.shape != u.shape or x.ndim != 2 or x.shape[1] != 3 or w.shape != (len(x),):
        raise ValueError('Expected matching (n,3) positions/displacements and (n,) weights')
    if not all(np.all(np.isfinite(a)) for a in (x,u,w)) or np.any(w <= 0):
        raise ValueError('Finite samples and positive weights required')
    center = np.average(x, axis=0, weights=w)
    translation = np.average(u, axis=0, weights=w)
    r = x-center
    inertia = np.eye(3)*np.sum(w*np.sum(r*r,axis=1)) - (r*w[:,None]).T@r
    omega = np.linalg.solve(inertia, np.sum(w[:,None]*np.cross(r,u-translation),axis=0))
    rotation = np.cross(omega,r)
    residual = u-translation-rotation
    total = np.sum(w[:,None]*u*u)
    if total <= 0:
        raise ValueError('Zero displacement has no modal character')
    terms = np.array([w.sum()*np.dot(translation,translation),
                      np.sum(w[:,None]*rotation**2), np.sum(w[:,None]*residual**2)]) / total
    if not np.isclose(terms.sum(),1,atol=1e-10):
        raise ValueError('Global motion projection did not close')
    # Sign-invariant orientation, reproducible under eigenvector reversal.
    axis = omega / np.linalg.norm(omega) if terms[1] > 1e-12 else np.full(3,np.nan)
    if np.all(np.isfinite(axis)):
        axis *= np.sign(axis[np.argmax(np.abs(axis))])
    return dict(center=center, axis=axis, omega=omega, translation=translation,
                f_translation=terms[0], f_rotation=terms[1], f_nonrigid=terms[2],
                rotation_fit=terms[1]/(terms[1]+terms[2]) if terms[1]+terms[2]>1e-12 else 0.)
