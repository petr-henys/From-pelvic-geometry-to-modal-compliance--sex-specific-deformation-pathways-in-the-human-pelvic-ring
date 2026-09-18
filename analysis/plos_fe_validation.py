#!/usr/bin/env python3
"""Targeted, isolated FE checks for the PLOS revision; never overwrites cohort data."""
from pathlib import Path
import sys,json,logging
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analysis.spectral_data import load_eigenvalues,_open_zarr_array
from analysis.spectral_config import DATA_DIR_FULL
from simulation.simulation_core import EigenstiffnessSimulation
from simulation.simulation_reference import ReferenceSolver
from simulation.eigensolver import ElasticEigenSolver
from simulation.elasticsolver import ElasticSolver
from petsc4py import PETSc


def main():
 out=Path('analysis_outputs/plos_revision/fe_validation');out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(Path('results/ref_S1P_fixed_new2/simulation_metadata.json').read_text())['config']
 cfg.update(RESULTS_DIR=str(out/'working'),FULL_DATASET_FLAG=False,STATIC_ANAL=False,COMPUTE_SENSITIVITIES=False,VERBOSE=False)
 sim=EigenstiffnessSimulation(cfg);logging.basicConfig(level=logging.WARNING)
 sim._setup_simulation();sim._prepare_materials();ReferenceSolver(sim)._initialize_ligaments()
 sim.eigen_solver=ElasticEigenSolver(sim.V,sim.E_func,sim.nu_func,sim.phi_func,sim.config.config,ligament_system=sim.ligament_system)
 sim.eigen_solver.fixed_dirichlet(gamma=cfg['SURFACE_PENALTY_GAMMA'],ds=sim.f2x_model.ds_named(cfg['BOUNDARY_FIXED']))
 old=load_eigenvalues(DATA_DIR_FULL); gaps=(old[:,9]-old[:,8])/old[:,8]
 ids=list(dict.fromkeys(int(np.argmin(abs(gaps-np.quantile(gaps,q)))) for q in [.1,.5,.9]))
 energy=_open_zarr_array(DATA_DIR_FULL/'mode_energy.zarr')
 rows=[]
 for idx in ids:
  print('Validation subject',idx,flush=True)
  sim._apply_sample_material_properties(idx);sim.shape_mapper.apply_sample(idx)
  sim.ligament_system.update();sim.eigen_solver.update()
  val,vec=sim.eigen_solver.solve(nev=40,target=cfg['EIGENVALUE_TARGET'])
  mismatch=float(np.max(abs(val[:15]/old[idx]-1)))
  if mismatch>1e-4:raise ValueError(f'Reconstruction mismatch {mismatch}: cannot validate archived spectrum')
  for j,load in enumerate(['SP2leg','SP1leg','LAB_phase1','LAB_phase2','LAB_phase3']):
   u=_open_zarr_array(DATA_DIR_FULL/f'displacements_{load}.zarr')[idx].ravel()
   v=sim.eigen_solver.K.createVecRight();v.setArray(u.copy());total=.5*v.dot(sim.eigen_solver.K*v)
   M=sim.eigen_solver.M; ai,aj,av=M.getValuesCSR()
   from scipy.sparse import csr_matrix
   mass=csr_matrix((av,aj,ai),shape=M.getSize());Phi=vec.reshape(len(val),-1).T
   gram=Phi.T@(mass@Phi);rhs=Phi.T@(mass@u);q=np.linalg.solve(gram,rhs)
   modal=.5*val*q*q*np.diag(gram)
   rows.append({'subject':idx,'load':load,'max_relative_eigenvalue_error':mismatch,
    'n_modes':len(val),'energy_capture_15':energy[idx,:,j].sum()/total,
    'energy_capture_40':modal[:40].sum()/total,'external_gap_15_16':(val[15]-val[14])/val[14]})
   v.destroy()
  np.savez_compressed(out/f'subject_{idx}_spectrum.npz',eigenvalues=val)
  pd.DataFrame(rows).to_csv(out/'convergence.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
