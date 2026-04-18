import numpy as np
from collections import Counter

def mean_median_mode(x):
    mean = np.mean(x)
    median = np.median(x)
    mode = Counter(x).most_common(1)[0][0]

    lst = [mean,median,mode]
   

    t = tuple(lst)
    return t