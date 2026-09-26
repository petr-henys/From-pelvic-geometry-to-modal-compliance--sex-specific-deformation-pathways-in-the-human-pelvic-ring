"""Read-only independent Jacobian check in FE DOF order; outputs audit diagnostics."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,pyvista as pv,ufl
from dolfinx import fem
from analysis.spectral_data import load_reference_space,load_template_and_shapes
from simulation.data_mapper import ShapeMapper
p,V=load_reference_space();mesh=V.mesh
coords=V.tabulate_dof_coordinates(); geom=mesh.geometry.x
print('DOF vs geometry coordinate max difference',np.max(abs(coords-geom)),flush=True)
phi=fem.Function(V);tpl,shapes=load_template_and_shapes()
mapper=ShapeMapper(phi,pv.PolyData(tpl),shapes,{'RBF_DATA_SMOOTHING':50.,'RBF_DATA_NEIGHBORS':10})
W=fem.functionspace(mesh,('DG',0));jfun=fem.Function(W)
expr=fem.Expression(ufl.det(ufl.Identity(3)+ufl.grad(phi)),W.element.interpolation_points)
old=pd.read_csv(ROOT/'analysis_outputs/mode_mechanisms/geometry_audit.csv')
ids=list(dict.fromkeys([0,101,120,206,int(old.n_cells_inverted.idxmax()),int(old.min_J.idxmin())]))
rows=[]
for i in ids:
 mapper.apply_sample(i);jfun.interpolate(expr);J=jfun.x.array.copy()
 row={'subject':i,'old_inverted':int(old.iloc[i].n_cells_inverted),'ufl_inverted':int((J<=0).sum()),'min_J':float(J.min())}
 rows.append(row);print(row,flush=True)
(Path(__file__).parent/'geometry_spotcheck.json').write_text(json.dumps(rows,indent=2))
