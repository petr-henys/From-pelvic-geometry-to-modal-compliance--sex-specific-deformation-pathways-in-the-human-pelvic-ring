"""Render archived measurement landmarks and exact geometric conventions.
Run from any directory. Does not change cohort measurements or FE fields.
"""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import RBFInterpolator
import zarr
P=Path(__file__).resolve().parents[1]; R=P.parents[1]
sys.path.insert(0,str(R))
from simulation.utils import load_json_points
F=P/'figures'; C=F/'.cache/measurement_atlas'; C.mkdir(parents=True,exist_ok=True)
ITEMS=[('AP','AP','Inlet AP diameter'),('biacetabular','BiacetabularWidth','Biacetabular width'),
 ('biischiadic','BiischiadicWidth','Biischiadic width'),('bituberous','BituberousWidth','Bituberous width'),
 ('iliopectineal_eminence','IliopectinealEminenceWidth','Iliopectineal eminence width'),
 ('PIT','PIT','Pelvic inlet transverse'),('sacral_width','SacralWidth','Sacral width'),
 ('subpubic','SubpubicAngle','Subpubic angle')]
COLORS=['#0072b2','#7b3294','#009e73','#d55e00','#a35b00','#0072b2','#7b3294','#c51b7d']
def points(name):return load_json_points(R/f'anatomy_data/palpace/{name}.mrk.json')
def value(q):
 if len(q)==2:return float(np.linalg.norm(q[1]-q[0]))
 a,b=q[0]-q[1],q[2]-q[1]
 return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
def render(mesh,q,name):
 # Orthographic view containing the measured segment, or the angle's exact plane.
 if len(q)==3:
  normal=np.cross(q[0]-q[1],q[2]-q[1]); focus=q.mean(0); scale=82
 elif name=='AP':normal=np.cross(q[1]-q[0],[1,0,0]);focus=np.array(mesh.center);scale=145
 else:
  e=(q[1]-q[0])/np.linalg.norm(q[1]-q[0]);normal=np.array([0.,-1.,.30]);normal-=e*(normal@e)
  focus=np.array(mesh.center);scale=145
 normal/=np.linalg.norm(normal)
 if normal[1]>0:normal=-normal
 up=np.array([0.,0.,1.]);up-=normal*(normal@up);up/=np.linalg.norm(up)
 pl=pv.Plotter(off_screen=True,window_size=(850,650));pl.set_background('white')
 pl.add_mesh(mesh,color='#dad8d0',smooth_shading=True,ambient=.4,diffuse=.6,specular=.1)
 pl.camera_position=[focus+normal*700,focus,up];pl.enable_parallel_projection();pl.camera.parallel_scale=scale
 pl.render()
 def project(x):
  pl.renderer.SetWorldPoint(*x,1);pl.renderer.WorldToDisplay();u,v,_=pl.renderer.GetDisplayPoint();return [u,650-v]
 xy=np.array([project(x) for x in q])
 arc=None
 if len(q)==3:
  a=q[0]-q[1];b=q[2]-q[1];a/=np.linalg.norm(a);b/=np.linalg.norm(b)
  theta=np.arccos(np.clip(a@b,-1,1));t=np.linspace(0,theta,70)
  tangent=(b-a*np.cos(theta))/np.sin(theta)
  arc=np.array([project(q[1]+20*(a*np.cos(u)+tangent*np.sin(u))) for u in t])
 image=pl.screenshot(return_img=True);pl.close();return image,xy,arc

def validate(mesh):
 X=np.load(R/'data/X.npy',mmap_mode='r'); archive=zarr.open(str(R/'data/dimensions_michal.zarr'),mode='r')
 q=np.concatenate([points(n) for n,_,_ in ITEMS]); report=[]
 for idx in range(len(X)):
  f=RBFInterpolator(mesh.points,X[idx]-mesh.points,neighbors=10,smoothing=100,kernel='thin_plate_spline')
  warped=q+f(q);k=0
  for name,key,_ in ITEMS:
   n=len(points(name));got=value(warped[k:k+n]);expected=float(archive[key][idx]);k+=n
   np.testing.assert_allclose(got,expected,rtol=1e-9,atol=1e-8)
   report.append(dict(subject_index=idx,variable=key,recomputed=got,archived=expected,error=abs(got-expected)))
 (P/'review/measurement_landmark_validation.json').write_text(json.dumps(report,indent=2))

def atlas(mesh):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
 fig,axs=plt.subplots(3,3,figsize=(9,7.4),layout='constrained')
 fig.get_layout_engine().set(w_pad=.035,h_pad=.04,wspace=.015,hspace=.025)
 for j,((name,key,title),color,ax) in enumerate(zip(ITEMS,COLORS,axs.flat)):
  q=points(name);im,xy,arc=render(mesh,q,name)
  # Remove unused render margins without altering projected geometry.
  mask=np.any(im[:,:,:3]<245,axis=2);yy,xx=np.where(mask)
  x0=max(0,int(xx.min())-24);x1=min(im.shape[1],int(xx.max())+25)
  y0=max(0,int(yy.min())-24);y1=min(im.shape[0],int(yy.max())+25)
  im=im[y0:y1,x0:x1];xy=xy-[x0,y0]
  if arc is not None:arc=arc-[x0,y0]
  ax.imshow(im);ax.axis('off')
  title=title.replace('Iliopectineal eminence width','Iliopectineal eminence\nwidth')
  ax.set_title(f'{chr(65+j)}. {title}',loc='left',fontsize=10,fontweight='bold',pad=3)
  ax.plot(xy[:,0],xy[:,1],color=color,lw=2.2,marker='o',ms=5,mec='white',mew=.7)
  for k,(x,y) in enumerate(xy):
   ax.annotate(f'$p_{k+1}$',(x,y),xytext=(7,-12 if k!=1 or len(q)==2 else 13),textcoords='offset points',color=color,fontsize=10,fontweight='bold',bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
  if arc is not None:ax.plot(arc[:,0],arc[:,1],color=color,lw=1.8)
  unit='°' if len(q)==3 else ' mm'
  ax.text(.5,-.015,f'{value(q):.1f}{unit}',transform=ax.transAxes,ha='center',va='top',fontsize=9,color=color)
 ax=axs[2,2];ax.axis('off')
 ax.set_title('I. Measurement key',loc='left',fontsize=10,fontweight='bold',pad=3)
 ax.text(.04,.81,r'$d=\|p_2-p_1\|$',fontsize=14,transform=ax.transAxes)
 ax.text(.04,.60,r'$\alpha_{sub}=\angle(p_1,p_2,p_3)$',fontsize=13,transform=ax.transAxes)
 ax.text(.04,.38,'Points: archived landmarks\nLengths: 3D distances\nAngle vertex: '+r'$p_2$',fontsize=9,linespacing=1.8,transform=ax.transAxes,va='top')
 ax.text(.04,.02,'Numbers refer to the template.',fontsize=9,transform=ax.transAxes)
 for ext in ['pdf','png']:fig.savefig(F/f'Fig_morphometric_landmarks.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)

def conventions(mesh):
 from matplotlib.patches import Arc
 fig,axs=plt.subplots(2,3,figsize=(10,6.8),layout='constrained')
 # Each ordered plane follows the right-handed positive rotation about its normal.
 for ax,title,h,v,angle in zip(axs[0],['A. ML-axis rotation','B. AP-axis rotation','C. CC-axis rotation'],['AP (y)','CC (z)','ML (x)'],['CC (z)','ML (x)','AP (y)'],[r'$\alpha_x$',r'$\beta_y$',r'$\gamma_z$']):
  ax.set_aspect('equal');ax.set_xlim(-.25,1.4);ax.set_ylim(-.2,1.4);ax.axis('off');ax.set_title(title,loc='left',fontsize=11,fontweight='bold')
  ax.annotate('',(1.15,0),(0,0),arrowprops=dict(arrowstyle='->',color='.3'))
  ax.annotate('',(0,1.15),(0,0),arrowprops=dict(arrowstyle='->',color='.3'))
  ax.plot([0,1],[0,0],color='#0072b2',lw=2)
  ax.plot([0,np.cos(.65)],[0,np.sin(.65)],color='#d55e00',lw=2)
  ax.add_patch(Arc((0,0),.9,.9,theta1=0,theta2=np.degrees(.65),color='#d55e00',lw=1.5))
  ax.text(.54,.19,angle,fontsize=12);ax.text(1.1,-.13,h,ha='center',fontsize=9);ax.text(.02,1.17,v,fontsize=9)
  ax.text(.5,-.03,'Isolated positive rotation; schematic',transform=ax.transAxes,ha='center',fontsize=8)
 ax=axs[1,0];ax.axis('off');ax.set_title('D. Translation at a common point',loc='left',fontsize=10,fontweight='bold')
 q0=np.array([.14,.30]);qs=np.array([.40,.35]);qi=np.array([.70,.74])
 for q,label,color in [(q0,r'$p_c$','black'),(qs,r'$T_S(p_c)$','#0072b2'),(qi,r'$T_I(p_c)$','#d55e00')]:
  ax.plot(*q,'o',color=color);ax.text(q[0]+.02,q[1]-.08,label,color=color,fontsize=11)
 for q in [qs,qi]:ax.plot([q0[0],q[0]],[q0[1],q[1]],ls=':',color='.5')
 ax.annotate('',qi,qs,arrowprops=dict(arrowstyle='->',lw=2,color='#009e73'))
 ax.text(.49,.54,r'$\Delta p$',color='#009e73',fontsize=12)
 ax.text(.02,.04,r'$\mathbf{d}=\mathbf{P}^{T}[T_I(p_c)-T_S(p_c)]$',fontsize=10)
 ax.set_xlim(0,1);ax.set_ylim(0,1)
 ax=axs[1,1];ax.axis('off');ax.set_title('E. Bone geometry included',loc='left',fontsize=11,fontweight='bold')
 pl=pv.Plotter(off_screen=True,window_size=(650,650));pl.set_background('white')
 bodies=mesh.split_bodies(); selected=[bodies[i].extract_surface() for i in [0,1,2]]
 for body,color in zip(selected,['#80b1d3','#fdb462','#b3de69']):pl.add_mesh(body,color=color,smooth_shading=True)
 c=np.array(mesh.center);pl.camera_position=[c+[0,-700,200],c,[0,0,1]];pl.enable_parallel_projection();pl.camera.parallel_scale=145
 ax.imshow(pl.screenshot(return_img=True));pl.close()
 ax.text(.5,-.03,'Left and right hip bones + sacrum',ha='center',transform=ax.transAxes,fontsize=8)
 ax=axs[1,2];ax.axis('off');ax.set_title('F. Size and surface measures',loc='left',fontsize=11,fontweight='bold')
 for y,text in [(.85,r'$V=\sum_{b=1}^{3}V_b$'),(.63,r'$A=\sum_{b=1}^{3}A_b$'),(.41,r'$s=(V/V_{\mathrm{template}})^{1/3}$')]:ax.text(.06,y,text,fontsize=13)
 ax.text(.06,.14,'V: enclosed bone volume (cm³)\nA: bone surface area (cm²)\ns: dimensionless scale',fontsize=9,linespacing=1.7)
 for ext in ['pdf','png']:fig.savefig(F/f'SuppFig2_measurement_conventions.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':
 mesh=pv.read(R/'data/pelvic.vtk');validate(mesh)
 bodies=mesh.split_bodies(); pelvic=pv.merge([bodies[i].extract_surface() for i in [0,1,2]])
 atlas(pelvic);conventions(mesh)
 print('Rendered landmark atlas and measurement conventions; 2224 archived values reproduced.')
