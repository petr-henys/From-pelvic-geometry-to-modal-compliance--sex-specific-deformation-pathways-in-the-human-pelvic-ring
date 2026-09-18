#!/usr/bin/env python3
"""Scientific rendering of actual reference mesh, load regions and eigenfields."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
from dolfinx import plot
from analysis.spectral_data import load_reference_space,load_inlet_landmarks
from analysis.plos_revision import FIG,style,save


def render(grid,front=True,arrows=(),points=None,scalars=None):
 p=pv.Plotter(off_screen=True,window_size=(650,650));p.set_background('white')
 opts={'color':'#d4d8dc'} if scalars is None else {'scalars':scalars,'cmap':'viridis','clim':[0,1]}
 p.add_mesh(grid,**opts,show_scalar_bar=False,smooth_shading=True)
 for start,vec,color in arrows:
  p.add_arrows(np.asarray(start)[None,:],np.asarray(vec)[None,:],mag=1,color=color)
 if points is not None:
  for pts,c in points:
   p.add_mesh(pv.lines_from_points(pts),color=c,line_width=5)
   p.add_points(pts,color=c,point_size=10,render_points_as_spheres=True)
 center=np.array(grid.center);extent=grid.length
 p.camera_position=(center+np.array([0,-extent*2,0]) if front else center+np.array([0,0,extent*2]),center,(0,0,1) if front else (0,1,0))
 p.enable_parallel_projection();p.camera.zoom(1.1)
 im=p.screenshot(return_img=True);p.close();return im


def main():
 style();FIG.mkdir(parents=True,exist_ok=True)
 parser,V=load_reference_space();topology,types,coords=plot.vtk_mesh(V)
 grid=pv.UnstructuredGrid(topology,types,coords);surface=grid.extract_surface()
 # Parser connectivity is zero-based in FEBio file-node order.
 def centroid(name):
  ids=np.unique(parser.surfaces[name]);return parser.nodes['AllNodes'][ids].mean(axis=0)
 lm=load_inlet_landmarks();fixed=centroid('S1_facet')
 cases=[('B  Two-leg stance',[('right_AC_notch',[0,0,30]),('left_AC_notch',[0,0,30])],True),
 ('C  One-leg stance',[('right_AC_notch',[0,0,45])],True),
 ('D  Ring compression',[('ring_contact_left',[30,0,0]),('ring_contact_right',[-30,0,0])],False),
 ('E  Tuberosity distraction',[('left_ischium_tuber',[30,0,0]),('right_ischium_tuber',[-30,0,0])],True),
 ('F  Outlet distraction',[('pubis_ins',[0,-30,0]),('SCJ',[0,30,0])],False)]
 fig,axs=plt.subplots(3,2,figsize=(6.4,8.0),layout='constrained')
 axs[0,0].imshow(render(surface,front=False,points=[(lm['AP'],'#CC3311'),(lm['ML'],'#0077BB')]))
 axs[0,0].set_title('A  Inlet landmarks (superior)',loc='left',fontweight='bold')
 for ax,(title,loads,front) in zip(list(axs.flat)[1:],cases):
  arrows=[(centroid(name),np.array(v),'#D55E00') for name,v in loads]
  im=render(surface,front=front,arrows=arrows,points=[(np.vstack([fixed+[-8,0,0],fixed+[8,0,0]]),'#000000')])
  ax.imshow(im);ax.set_title(title,loc='left',fontweight='bold')
 for ax in axs.flat:ax.axis('off')
 save(fig,'model_setup')
 # Reference fields only: no undefined median/sign averaging.
 modes=np.load('results/ref_S1P_fixed_new2/reference_solution.npz')['eigenvectors']
 fig,axs=plt.subplots(5,3,figsize=(6.4,9.0),layout='constrained')
 for i,ax in enumerate(axs.flat):
  field=modes[i];amp=np.linalg.norm(field,axis=1);unit=field/amp.max()
  deformed=grid.copy();deformed.points=coords+10*unit;deformed.point_data['amplitude']=amp/amp.max()
  ax.imshow(render(deformed,front=True,scalars='amplitude'));ax.axis('off');ax.set_title(f'Rank {i+1}',loc='left')
 fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='viridis'),ax=axs,location='bottom',fraction=.025,shrink=.7,label='Displacement magnitude / maximum',pad=.01)
 save(fig,'reference_modes')

if __name__=='__main__':main()
