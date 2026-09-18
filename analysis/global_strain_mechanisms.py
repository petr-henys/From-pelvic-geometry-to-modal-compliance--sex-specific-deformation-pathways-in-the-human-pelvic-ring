"""Whole-domain, axis-conditioned strain-pattern projections.

Sections discretise a longitudinal coordinate, not anatomical structures. A single
axis and all continuum cells define each global profile. This is a beam-pattern
approximation, not a unique constitutive decomposition of a pelvic ring.
"""
import numpy as np
from analysis.mode_mechanism_metrics import RegionalBeamModel

MECHANISMS = ('axial', 'bending', 'shear', 'twist', 'residual')


def modal_cell_strain(displacement, cells, inverse_edges):
    u = np.asarray(displacement)
    differences = np.stack([u[cells[:, j]]-u[cells[:, 0]] for j in (1,2,3)], axis=-1)
    grad = differences @ inverse_edges
    return .5*(grad+grad.transpose(0,2,1))


class GlobalStrainModel:
    """Fit axial, bending, shear and torsional strain patterns on the whole mesh.

    Equal-width longitudinal sections allow coefficients to vary along the axis.
    Patterns are sampled at cell centroids and projected with positive cell-volume
    weights. Fractions sum global squared strain norms, never regional percentages.
    """
    def __init__(self, centroids, volumes, axis, n_sections=8):
        self.centroids = np.asarray(centroids, dtype=float)
        self.volumes = np.asarray(volumes, dtype=float)
        self.axis = np.asarray(axis, dtype=float).copy()
        if self.centroids.shape != (len(self.volumes),3) or not np.all(np.isfinite(self.centroids)):
            raise ValueError('Finite (n,3) centroids and matching volumes required')
        if not np.all(np.isfinite(self.volumes)) or np.any(self.volumes <= 0):
            raise ValueError('Finite, strictly positive volumes required')
        if self.axis.shape != (3,) or not np.all(np.isfinite(self.axis)) or np.linalg.norm(self.axis)==0:
            raise ValueError('A finite nonzero axis is required')
        if not isinstance(n_sections,int) or n_sections<1:
            raise ValueError('Positive integer section count required')
        self.axis /= np.linalg.norm(self.axis)
        self.center = np.average(self.centroids,axis=0,weights=self.volumes)
        # Scale geometry for numerical conditioning, preserving the pattern span.
        length = np.linalg.norm(np.ptp(self.centroids,axis=0))
        if length <= 0: raise ValueError('Degenerate geometry')
        coords = (self.centroids-self.center)/length
        s = coords@self.axis
        edges = np.linspace(s.min(),s.max(),n_sections+1)
        membership = np.clip(np.searchsorted(edges,s,side='right')-1,0,n_sections-1)
        self.sections=[]
        for index in range(n_sections):
            ids = np.flatnonzero(membership==index)
            if len(ids):
                self.sections.append(RegionalBeamModel(cell_indices=ids,centroids=coords[ids],e_long=self.axis))

    def decompose(self, strain):
        eps = np.asarray(strain,dtype=float)
        if eps.shape != (len(self.volumes),3,3) or not np.all(np.isfinite(eps)):
            raise ValueError('Finite (n,3,3) strains required')
        if not np.allclose(eps,eps.transpose(0,2,1)):
            raise ValueError('Symmetric strains required')
        scale = np.max(np.abs(eps))
        if scale == 0:
            return dict(is_valid=False,D_total=0.,**{'f_'+k:np.nan for k in MECHANISMS})
        # Removing eigenvector amplitude prevents absolute small-number cutoffs
        # in the local least-squares kernel from affecting the global fractions.
        eps = eps/scale
        weights = self.volumes/self.volumes.sum()
        total = float(np.sum(weights[:,None,None]*eps**2))
        contributions = dict.fromkeys(MECHANISMS,0.)
        for section in self.sections:
            result = section.decompose(eps,weights)
            if result['is_valid']:
                for key in MECHANISMS: contributions[key] += result['D_'+key]
            else:
                ids=section.cell_indices
                contributions['residual'] += float(np.sum(weights[ids,None,None]*eps[ids]**2))
        fractions = {'f_'+key:value/total for key,value in contributions.items()}
        if not np.isclose(sum(fractions.values()),1,atol=1e-9):
            raise ValueError('Whole-domain strain decomposition did not close')
        return dict(is_valid=True,D_total=total*scale**2*self.volumes.sum(),**fractions)
