import numpy as np

def auc(fpr, tpr):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    if not np.all(fpr[:-1] <= fpr[1:]):
        raise ValueError("FPR values must be increasing")
    
    auc = 0.0
    for i in range(len(fpr) - 1):
        width = fpr[i+1] - fpr[i]           
        height = 0.5 * (tpr[i] + tpr[i+1]) 
        auc += width * height
    
    return auc

fpr1 = [0, 0, 1]
tpr1 = [0, 1, 1]
print(auc(fpr1, tpr1))  

fpr2 = [0, 1]
tpr2 = [0, 1]
print(auc(fpr2, tpr2))  

fpr3 = [0, 0.1, 0.3, 0.6, 1.0]
tpr3 = [0, 0.25, 0.65, 0.85, 0.95]
print(auc(fpr3, tpr3))  