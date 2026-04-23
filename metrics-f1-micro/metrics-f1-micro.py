def f1_micro(y_true, y_pred):
    tp = 0
    
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp += 1
    
    return tp / len(y_true)