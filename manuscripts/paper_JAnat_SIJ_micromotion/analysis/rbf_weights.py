"""Sparse evaluation weights for the repository's degree-1 thin-plate RBF.

The interpolation sites, targets and neighbor count are constant across subjects;
only the field values change. Cache the linear operator instead of solving the
same local interpolation systems for every displacement and shape field.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

def weights(sites,targets,neighbors=10,smoothing=0.,batch=2000):
    sites=np.asarray(sites,float);targets=np.asarray(targets,float)
    _,indices=cKDTree(sites).query(targets,k=neighbors)
    values=np.empty(indices.shape,float)
    def kernel(r):
        return np.where(r>0,r*r*np.log(np.maximum(r,np.finfo(float).tiny)),0.)
    for start in range(0,len(targets),batch):
        end=min(start+batch,len(targets));p=sites[indices[start:end]];q=targets[start:end]
        lo=p.min(axis=1);hi=p.max(axis=1);shift=(hi+lo)/2;scale=(hi-lo)/2;scale[scale==0]=1
        poly=np.concatenate([np.ones((*p.shape[:2],1)),(p-shift[:,None,:])/scale[:,None,:]],axis=2)
        matrix=np.zeros((len(p),neighbors+4,neighbors+4))
        matrix[:,:neighbors,:neighbors]=kernel(np.linalg.norm(p[:,:,None,:]-p[:,None,:,:],axis=3))
        matrix[:,np.arange(neighbors),np.arange(neighbors)]+=smoothing
        matrix[:,:neighbors,neighbors:]=poly;matrix[:,neighbors:,:neighbors]=poly.transpose(0,2,1)
        rhs=np.concatenate([kernel(np.linalg.norm(p-q[:,None,:],axis=2)),np.ones((len(p),1)),(q-shift)/scale],axis=1)
        values[start:end]=np.linalg.solve(matrix,rhs[:,:,None])[:,:neighbors,0]
    return csr_matrix((values.ravel(),indices.ravel(),np.arange(len(targets)+1)*neighbors),shape=(len(targets),len(sites)))
