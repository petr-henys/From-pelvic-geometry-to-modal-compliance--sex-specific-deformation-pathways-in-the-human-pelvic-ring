"""Independent saved-output reproduction and rigid-only correction sensitivity."""
import recompute_sij_reference as r
from anatomy_analyser import sij_relative_angles
from scipy.interpolate import RBFInterpolator
import numpy as np,pandas as pd,zarr,json
r.init(); rows=[]; errors=[]
for i in [0,55,110,165,220,277]:
    ch='full';cfg=r.CFG[ch];coords=r.DOFS[ch];m=r.TEMPLATE
    phi=r.W_SHAPE[ch]@(r.SHAPES[i]-m.points)
    # Verify shape interpolation independently of the sparse implementation.
    direct=RBFInterpolator(m.points,r.SHAPES[i]-m.points,smoothing=cfg['RBF_DATA_SMOOTHING'],neighbors=cfg['RBF_DATA_NEIGHBORS'])(coords[::199])
    np.testing.assert_allclose(phi[::199],direct,atol=1e-8,rtol=1e-8)
    fields=np.concatenate([phi]+[np.asarray(r.STORES[ch][l][i]) for l in r.LOADS],axis=1)
    surface=r.W_SURFACE[ch]@fields
    unloaded=m.copy(deep=True);unloaded.points=m.points+surface[:,:3]
    for j,load in enumerate(r.LOADS):
        loaded=unloaded.copy(deep=True);loaded.points=unloaded.points+surface[:,3*(j+1):3*(j+2)]
        a,t=r.arrays(sij_relative_angles(unloaded,loaded))
        ar,tr=r.arrays(sij_relative_angles(unloaded,loaded,sacrum_correction='rigid'))
        rows.append(dict(subject_idx=i,load_case=load,affine_rotation=np.linalg.norm(a,axis=1).mean(),rigid_rotation=np.linalg.norm(ar,axis=1).mean(),affine_translation=np.linalg.norm(t,axis=1).mean(),rigid_translation=np.linalg.norm(tr,axis=1).mean()))
        if load=='SP2leg':
            olda,oldt=r.arrays(sij_relative_angles(m,loaded))
            for metric,arr in [('angles',olda),('trans',oldt)]:
                archived=np.asarray(zarr.open_group(str(r.ROOT/'results'/r.CHANNELS[ch]/'data'/f'sij_{metric}_{load}.zarr'),mode='r')['data'][i])
                err=float(np.max(np.abs(arr-archived)))
                np.testing.assert_allclose(arr,archived,atol=1e-8,rtol=1e-8)
                errors.append(dict(subject_idx=i,metric=metric,max_error=err))
    print('Checked',i,flush=True)
r.OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(r.OUT/'rigid_correction_sensitivity.csv',index=False)
(r.OUT/'saved_output_reproduction.json').write_text(json.dumps(errors,indent=2))
