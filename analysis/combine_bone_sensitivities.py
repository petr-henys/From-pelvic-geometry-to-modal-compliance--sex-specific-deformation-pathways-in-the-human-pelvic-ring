"""Combine validated subject shards of corrected bone-law sensitivities."""
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_outputs/material_uncertainty'
paths=sorted(OUT.glob('corrected_bone_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz'))
full=OUT/'corrected_bone_000_278.npz'
parts=[]
if full.exists():
    paths=[full]
else:
    partial=OUT/'corrected_bone_sensitivities.npz'
    if partial.exists():
        z=np.load(partial)
        count=int(z['completed'])
        parts.append((np.arange(count),z['alpha'],z['beta']))
for path in paths:
    z=np.load(path)
    expected=int(path.stem[-3:])-int(path.stem[-7:-4])
    if len(z['indices'])!=expected:raise RuntimeError(f'Incomplete shard {path}')
    parts.append((z['indices'],z['alpha'],z['beta']))
if not parts:raise RuntimeError('No sensitivity shards found')
ids=np.concatenate([p[0] for p in parts])
if sorted(ids.tolist())!=list(range(278)):raise RuntimeError('Missing or duplicated subject')
a=np.empty((278,15));b=np.empty((278,15))
for indices,alpha,beta in parts:
    a[indices]=alpha;b[indices]=beta
if not (np.isfinite(a).all() and np.isfinite(b).all()):raise RuntimeError('Nonfinite sensitivity')
np.savez_compressed(OUT/'corrected_bone_sensitivities_complete.npz',alpha=a,beta=b,completed=278)
(OUT/'corrected_bone_provenance.json').write_text(json.dumps({
    'source_script':'analysis/recompute_bone_sensitivities.py',
    'source_sha256':hashlib.sha256((ROOT/'analysis/recompute_bone_sensitivities.py').read_bytes()).hexdigest(),
    'source_eigenvectors':'results/ref_S1P_fixed_new2/data/eigenvectors.zarr',
    'subjects':278,'modes':15,'validation':'derivative_validation.json',
    'method':'Derivative of source modulus RBF-mapped to FE cells; mapped P1 strain energy and consistent-mass normalization'
},indent=2))
print('Combined 278 subjects; alpha range',a.min(),a.max(),'beta range',b.min(),b.max())
