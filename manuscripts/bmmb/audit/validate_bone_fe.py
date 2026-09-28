"""Independent UFL scalar-form check of corrected archived-mode derivatives."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,ufl
from scipy.interpolate import RBFInterpolator
from dolfinx import fem
from analysis.spectral_data import load_reference_space,load_template_and_shapes,load_eigenvectors_obj
from analysis.spectral_config import DATA_DIR_FULL
p,V=load_reference_space();mesh=V.mesh;W=fem.functionspace(mesh,('DG',0))
meta=json.loads((ROOT/'results/ref_S1P_fixed_new2/simulation_metadata.json').read_text())['config']
tpl,shapes=load_template_and_shapes();density=np.load(ROOT/meta['DENSITY_DATA_PATH'],mmap_mode='r')[0]
opts=dict(smoothing=float(meta['RBF_DATA_SMOOTHING']),neighbors=int(meta['RBF_DATA_NEIGHBORS']))
phi=fem.Function(V);phi.x.array[:]=RBFInterpolator(tpl,shapes[0]-tpl,**opts)(V.tabulate_dof_coordinates()).ravel();phi.x.scatter_forward()
labels=np.char.lower(np.asarray(p.material_labels,dtype=str));bone=(np.char.find(labels,'bone')>=0)&(np.char.find(labels,'cartilage')<0)&(np.char.find(labels,'symph')<0)
poisson=fem.Function(W)
poisson.x.array[:]=float(meta['BONE_POISSON_RATIO'])
poisson.x.array[np.char.find(labels,'cartilage')>=0]=float(meta['SIJ_CARTILAGE_POISSON_RATIO'])
poisson.x.array[np.char.find(labels,'symph')>=0]=float(meta['SYMPHYSIS_POISSON_RATIO'])
Wdg=np.array([W.dofmap.cell_dofs(i)[0] for i in range(len(labels))])
rho=(np.maximum(density,0)+.09)/1.14
active=rho>float(meta['BONE_MODULUS_THRESHOLD'])
alpha=float(meta['BONE_MODULUS_ALPHA']);beta=float(meta['BONE_MODULUS_BETA'])
da=fem.Function(W);db=fem.Function(W)
for field,source in [(da,np.where(active,rho**beta,0.)),(db,np.where(active,alpha*rho**beta*np.log(rho),0.))]:
    field.x.array[:]=RBFInterpolator(tpl,source,**opts)(W.tabulate_dof_coordinates()).ravel()
    field.x.array[Wdg[~bone]]=0.
u=fem.Function(V)
F=ufl.Identity(3)+ufl.grad(phi);J=ufl.det(F)
eps=ufl.sym(ufl.grad(u)*ufl.inv(F))
g=poisson/((1+poisson)*(1-2*poisson))*ufl.tr(eps)**2+ufl.inner(eps,eps)/(1+poisson)
dx=ufl.Measure('dx',domain=mesh,metadata={'quadrature_degree':int(meta['QUADRATURE_DEGREE'])})
forms=[fem.form(expr*dx) for expr in [J*ufl.inner(u,u),J*da*g,J*db*g]]
eig=load_eigenvectors_obj(DATA_DIR_FULL)
arch=np.load(ROOT/'analysis_outputs/material_uncertainty/corrected_bone_sensitivities.npz')
rows=[]
for mode in [0,8,14]:
    u.x.array[:]=np.asarray(eig[0,mode]).ravel()
    mass,a,b=[float(fem.assemble_scalar(form)) for form in forms]
    row=dict(mode=mode+1,alpha=a/mass,beta=b/mass,
             alpha_vectorized=float(arch['alpha'][0,mode]),beta_vectorized=float(arch['beta'][0,mode]))
    row['alpha_relative_error']=abs(row['alpha']-row['alpha_vectorized'])/max(abs(row['alpha']),1e-20)
    row['beta_relative_error']=abs(row['beta']-row['beta_vectorized'])/max(abs(row['beta']),1e-20)
    rows.append(row)
(Path(__file__).parent/'bone_fe_validation.json').write_text(json.dumps(rows,indent=2))
print(rows,flush=True)
