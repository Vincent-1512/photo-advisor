"""Linear luminance unsharp mask using NumPy only."""
import numpy as np

def apply_sharpen(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Apply Y'=Y+k*(Y-Gaussian(Y,sigma=1,radius=3)); separable reflect-padded Gaussian."""
    if amount == 0: return rgb
    y=np.sum(rgb*np.asarray([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
    x=np.arange(-3,4,dtype=np.float32); k=np.exp(-x*x/2); k/=k.sum()
    p=np.pad(y,((0,0),(3,3),(0,0)),mode='reflect' if y.shape[1]>1 else 'edge')
    b=sum(k[i]*p[:,i:i+y.shape[1],:] for i in range(7))
    p=np.pad(b,((3,3),(0,0),(0,0)),mode='reflect' if y.shape[0]>1 else 'edge')
    b=sum(k[i]*p[i:i+y.shape[0],:,:] for i in range(7))
    yn=np.clip(y+np.float32(amount)*(y-b),0.0,1.0); scale=np.divide(yn,y,out=np.ones_like(y),where=y>0)
    return rgb*scale
