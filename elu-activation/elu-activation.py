import math

def elu(x, alpha):
    result = []
    
    for val in x:
        if val > 0:
            result.append(val)
        else:
            result.append(alpha * (math.exp(val) - 1))
    
    return result


