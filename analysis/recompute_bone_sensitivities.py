"""Recompute bone-law eigenvalue derivatives from archived eigenvectors.

The source-point material law is differentiated before its RBF interpolation.
The FE integrals use the same P1 mapping and signed volume measure as the
archived solver. No eigenproblem is re-solved.
"""
from pathlib import Path
import json
import argparse
import sys
import numpy as np
from scipy.interpolate import RBFInterpolator
import zarr
from dolfinx import fem
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.spectral_data import load_reference_space, load_template_and_shapes, load_eigenvectors_obj
from analysis.spectral_config import DATA_DIR_FULL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis_outputs/material_uncertainty/corrected_bone_sensitivities.npz'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--stop',type=int,default=None)
    args=ap.parse_args()
    meta = json.loads((ROOT/'results/ref_S1P_fixed_new2/simulation_metadata.json').read_text())
    cfg = meta['config']; alpha=float(cfg['BONE_MODULUS_ALPHA']); beta=float(cfg['BONE_MODULUS_BETA'])
    threshold=float(cfg['BONE_MODULUS_THRESHOLD'])
    parser,V=load_reference_space(); mesh=V.mesh
    X=mesh.geometry.x; cells=mesh.geometry.dofmap
    if not np.allclose(V.tabulate_dof_coordinates(),X):
        raise ValueError('DOF/geometry coordinates differ')
    W=fem.functionspace(mesh,('DG',0))
    dc=W.tabulate_dof_coordinates()
    dg=np.array([W.dofmap.cell_dofs(i)[0] for i in range(len(cells))])
    labels=np.char.lower(np.asarray(parser.material_labels,dtype=str))
    bone=(np.char.find(labels,'bone')>=0)&(np.char.find(labels,'cartilage')<0)&(np.char.find(labels,'symph')<0)
    nu=np.full(len(cells),float(cfg['BONE_POISSON_RATIO']))
    nu[np.char.find(labels,'cartilage')>=0]=float(cfg['SIJ_CARTILAGE_POISSON_RATIO'])
    nu[np.char.find(labels,'symph')>=0]=float(cfg['SYMPHYSIS_POISSON_RATIO'])
    v=X[cells]; edges=np.stack([v[:,j]-v[:,0] for j in (1,2,3)],axis=-1)
    inv_edges=np.linalg.inv(edges); vol=np.abs(np.linalg.det(edges))/6
    tpl,shapes=load_template_and_shapes()
    density=np.load(ROOT/cfg['DENSITY_DATA_PATH'],mmap_mode='r')
    modes=load_eigenvectors_obj(DATA_DIR_FULL)
    n=modes.shape[0]
    start=args.start;stop=n if args.stop is None else args.stop
    if not (0 <= start < stop <= n):raise ValueError('Invalid subject interval')
    out_a=np.empty((stop-start,15));out_b=np.empty((stop-start,15))
    target=OUT.with_name(f'corrected_bone_{start:03d}_{stop:03d}.npz')
    opts=dict(smoothing=float(cfg['RBF_DATA_SMOOTHING']),neighbors=int(cfg['RBF_DATA_NEIGHBORS']))
    for local,i in enumerate(range(start,stop)):
        phi=RBFInterpolator(tpl,shapes[i]-tpl,**opts)(X)
        dphi=np.stack([phi[cells[:,j]]-phi[cells[:,0]] for j in (1,2,3)],axis=-1)
        F=np.eye(3)+dphi@inv_edges
        J=np.linalg.det(F); Finv=np.linalg.inv(F)
        rho=(np.maximum(density[i],0)+.09)/1.14
        active=rho>threshold
        power=rho**beta
        dEa=RBFInterpolator(tpl,np.where(active,power,0.),**opts)(dc)[dg]
        dEb=RBFInterpolator(tpl,np.where(active,alpha*power*np.log(rho),0.),**opts)(dc)[dg]
        dEa[~bone]=0.;dEb[~bone]=0.
        factor=vol*J
        for k,u in enumerate(np.asarray(modes[i],dtype=float)):
            du=np.stack([u[cells[:,j]]-u[cells[:,0]] for j in (1,2,3)],axis=-1)
            grad=du@inv_edges
            A=grad@Finv;eps=.5*(A+np.swapaxes(A,1,2))
            tr=np.trace(eps,axis1=1,axis2=2)
            energy_per_E=nu/((1+nu)*(1-2*nu))*tr**2 + np.sum(eps*eps,axis=(1,2))/(1+nu)
            nodal_u=u[cells]
            sum_u=nodal_u.sum(axis=1)
            mass=np.sum(factor*(np.sum(sum_u*sum_u,axis=1)+np.sum(nodal_u*nodal_u,axis=(1,2)))/20)
            out_a[local,k]=np.sum(factor*dEa*energy_per_E)/mass
            out_b[local,k]=np.sum(factor*dEb*energy_per_E)/mass
        if local%10==0:
            print(f'{i+1}/{n} (shard {start}:{stop})',flush=True)
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,alpha=out_a[:local+1],beta=out_b[:local+1],indices=np.arange(start,i+1))
    np.savez_compressed(target,alpha=out_a,beta=out_b,indices=np.arange(start,stop))

if __name__=='__main__':main()
