import numpy as np

def make_diagonal(v):
   v = np.asarray(v, dtype = float)

   if v.ndim != 1:
     raise ValueError("Inputs must be 1D vector")


   return np.diag(v)