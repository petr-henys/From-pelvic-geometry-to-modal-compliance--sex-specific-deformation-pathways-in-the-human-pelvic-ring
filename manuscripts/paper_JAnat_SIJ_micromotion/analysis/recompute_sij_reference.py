"""Re-extract SIJ endpoints from saved FE fields using unloaded subject geometry.

Original FE/Zarr data are read-only. Each subject checkpoint is independently
resumable. Reference VTK displacement arrays establish the saved DOF ordering.
"""
from pathlib import Path
import sys, json, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyvista as pv
import zarr
from scipy.interpolate import RBFInterpolator
from rbf_weights import weights
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from anatomy_analyser import sij_relative_angles
PAPER=Path(__file__).resolve().parents[1]
OUT=PAPER/'tables'/'corrected'
LOADS=['SP2leg','SP1leg','LAB_phase1','LAB_phase2','LAB_phase3']
CHANNELS={'full':'ref_S1P_fixed_new2','shape_only':'ref_S1P_fixed_new2_shape_only','material_only':'ref_S1P_fixed_new2_material_only'}

def arrays(stats):
    return (np.array([list(stats[s]['angles_deg'].values()) for s in ['SIJ_left','SIJ_right']]),
            np.array([list(stats[s]['displacement_mm'].values()) for s in ['SIJ_left','SIJ_right']]))

def init():
    global TEMPLATE, DOFS, SHAPES, STORES, CFG, W_SHAPE, W_SURFACE
    TEMPLATE=pv.read(ROOT/'data/pelvic.vtk'); SHAPES=np.load(ROOT/'data/X.npy',mmap_mode='r')
    DOFS={}; STORES={}; CFG={}; W_SHAPE={}; W_SURFACE={}
    for ch,folder in CHANNELS.items():
        base=ROOT/'results'/folder
        mesh=pv.read(base/'paraview/reference_fields.vtk')
        ref=np.load(base/'reference_solution.npz')
        for load in LOADS:
            np.testing.assert_array_equal(mesh['ref_u_'+load],ref['displacement_'+load])
        DOFS[ch]=mesh.points
        CFG[ch]=json.loads((base/'simulation_metadata.json').read_text())['config']
        STORES[ch]={load:zarr.open_group(str(base/'data'/f'displacements_{load}.zarr'),mode='r')['data'] for load in LOADS}
        cfg=CFG[ch]
        if ch!='full' and np.array_equal(DOFS[ch],DOFS['full']):
            W_SHAPE[ch]=W_SHAPE['full']; W_SURFACE[ch]=W_SURFACE['full']
        else:
            W_SHAPE[ch]=weights(TEMPLATE.points,DOFS[ch],cfg['RBF_DATA_NEIGHBORS'],cfg['RBF_DATA_SMOOTHING'])
            W_SURFACE[ch]=weights(DOFS[ch],TEMPLATE.points,cfg['RBF_SOLVER_NEIGHBORS'],cfg['RBF_SOLVER_SMOOTHING'])
            probe=np.sin(DOFS[ch]/31)
            actual=RBFInterpolator(DOFS[ch],probe,neighbors=cfg['RBF_SOLVER_NEIGHBORS'],smoothing=cfg['RBF_SOLVER_SMOOTHING'])(TEMPLATE.points[::301])
            np.testing.assert_allclose((W_SURFACE[ch]@probe)[::301],actual,atol=1e-8,rtol=1e-8)


def subject(i):
    path=OUT/f'subject_{i:03d}.npz'
    if path.exists(): return i
    values={}
    for ch in CHANNELS:
        cfg=CFG[ch]; coords=DOFS[ch]
        if ch=='material_only': phi=np.zeros_like(coords)
        else:
            phi=W_SHAPE[ch]@(SHAPES[i]-TEMPLATE.points)
        # Batch all five loads and geometry through the identical linear RBF map.
        fields=np.concatenate([phi]+[np.asarray(STORES[ch][l][i]) for l in LOADS],axis=1)
        surface=W_SURFACE[ch]@fields
        unloaded=TEMPLATE.copy(deep=True); unloaded.points=TEMPLATE.points+surface[:,:3]
        if ch=='full':
            a,t=arrays(sij_relative_angles(TEMPLATE,unloaded))
            values['zero_old_angles']=a; values['zero_old_trans']=t
            a,t=arrays(sij_relative_angles(unloaded,unloaded))
            if max(np.abs(a).max(),np.abs(t).max())>1e-8: raise ValueError('Zero-motion check failed')
        for j,load in enumerate(LOADS):
            loaded=unloaded.copy(deep=True); loaded.points=unloaded.points+surface[:,3*(j+1):3*(j+2)]
            a,t=arrays(sij_relative_angles(unloaded,loaded))
            values[f'{ch}_{load}_angles']=a; values[f'{ch}_{load}_trans']=t
    np.savez_compressed(path,**values)
    return i

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=278);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
        for n,i in enumerate(pool.map(subject,range(args.limit)),1):
            if n%10==0 or n==args.limit: print(f'{n}/{args.limit} subjects complete',flush=True)
    records=[]
    for i in range(args.limit):
        with np.load(OUT/f'subject_{i:03d}.npz') as d:
            for ch in CHANNELS:
                for load in LOADS:
                    a=d[f'{ch}_{load}_angles'];t=d[f'{ch}_{load}_trans']
                    r=np.linalg.norm(a,axis=1);v=np.linalg.norm(t,axis=1)
                    row=dict(subject_idx=i,channel=ch,load_case=load,rot_mag_deg=r.mean(),trans_mag_mm=v.mean(),
                             lr_rot_asym_deg=abs(r[0]-r[1]),lr_trans_asym_mm=abs(v[0]-v[1]))
                    for j,name in enumerate(['nut_abs_deg','ap_rot_abs_deg','cc_rot_abs_deg']):row[name]=np.abs(a[:,j]).mean()
                    for j,axis in enumerate(['ml','ap','cc']):
                        row[f'{axis}_trans_abs_mm']=np.abs(t[:,j]).mean();row[f'{axis}_trans_signed_mm']=t[:,j].mean()
                    records.append(row)
    pd.DataFrame(records).to_csv(OUT/'kinematics.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='X + phi',loaded='X + phi + u',
        affine_sacrum_removal=True,original_results_modified=False,n_subjects=args.limit,
        dof_order_check='exact equality of all five reference displacement arrays and VTK point arrays'),indent=2))
if __name__=='__main__':main()
