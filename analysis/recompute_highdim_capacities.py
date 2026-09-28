"""Update four-mode anatomical capacities in the cached cohort table.

All other metrics use unchanged algorithms and remain from the archived table.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.spectral_data import (load_fe_mesh_coords,load_mass_matrix,load_template_and_shapes,
    load_inlet_landmarks,load_outlet_landmarks,load_eigenvectors_obj)
from analysis.spectral_metrics import (orthonormalize_l2,build_patient_measurement_vectors,
    single_functional_capacity_max_nodal)
from analysis.spectral_config import DATA_DIR_FULL
from analysis.plos_revision import ROOT,TAB,OUT,AXES,source_fingerprint

def main():
    path=TAB/'subject_metrics.csv'; df=pd.read_csv(path)
    rows=df.index[df.block=='12-15'].to_numpy()
    if len(rows)!=278:raise ValueError('Expected 278 four-mode rows')
    M=load_mass_matrix();coords=load_fe_mesh_coords();tree=cKDTree(coords)
    tpl,shapes=load_template_and_shapes();lm={**load_inlet_landmarks(),**load_outlet_landmarks()}
    measurement={axis:build_patient_measurement_vectors(tree,len(coords),tpl,shapes,*lm[axis]) for axis in AXES}
    eig=load_eigenvectors_obj(DATA_DIR_FULL)
    for j,row in enumerate(rows):
        subject=int(df.at[row,'subject'])
        Q=orthonormalize_l2(np.asarray(eig[subject,11:15]).reshape(4,-1).T,M=M)
        for axis in AXES:
            cap,_,sigma=single_functional_capacity_max_nodal(Q,measurement[axis][subject],M=M)
            df.at[row,axis]=cap
            if axis in ['AP','ML','BIS','BIT']:
                df.at[row,axis+'_scalar']=cap
            if axis in ['AP','ML']:
                df.at[row,axis+'_mass_norm']=sigma
        if j%25==0:print(f'{j+1}/278',flush=True)
    df.to_csv(path,index=False)
    provenance=json.loads((OUT/'provenance.json').read_text())
    provenance['source_hashes']=source_fingerprint()
    provenance['four_mode_capacity']='Convex nodal constraint generation, all nodes checked, dual gap <= 1e-5'
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
