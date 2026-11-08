"""Linear-light white balance."""
import numpy as np

def apply_white_balance(rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Apply normalized channel gains: a=2**(.25T+.10t), b=2**(-.05T+.25t), c=2**(-.25T-.10t), normalized by geometric mean."""
    if temperature == 0 and tint == 0: return rgb
    a,b,c=2**(.25*temperature+.10*tint),2**(-.05*temperature+.25*tint),2**(-.25*temperature-.10*tint)
    g=(a*b*c)**(1/3)
    return rgb*np.asarray([a/g,b/g,c/g],dtype=np.float32)
