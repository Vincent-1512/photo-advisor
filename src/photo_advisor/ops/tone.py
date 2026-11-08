"""Linear-light shadow and highlight tone curves."""
import numpy as np

def _tone(rgb: np.ndarray, amount: float, shadows: bool) -> np.ndarray:
    """Apply Y'=max(0,Y+0.5*a*4*(1-Y)^2*Y) for shadows, or 4*Y^2*(1-Y) for highlights; scale RGB by Y'/Y."""
    if amount == 0: return rgb
    y=np.sum(rgb*np.asarray([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
    w=4*(1-y)**2*y if shadows else 4*y*y*(1-y)
    yn=np.maximum(0,y+np.float32(.5*amount)*w)
    scale=np.divide(yn,y,out=np.ones_like(y),where=y>0)
    return rgb*scale

def apply_shadows(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Lift shadows with a smooth luminance mask."""
    return _tone(rgb,amount,True)

def apply_highlights(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Adjust highlights with a smooth luminance mask."""
    return _tone(rgb,amount,False)
