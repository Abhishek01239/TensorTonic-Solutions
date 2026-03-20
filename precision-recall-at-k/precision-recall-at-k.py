def precision_recall_at_k(recommended, relevant, k):
    relevant_set = set(relevant)
    
    hits = 0
    
    for i in range(k):
        if recommended[i] in relevant_set:
            hits += 1
    
    precision = hits / k
    recall = hits / len(relevant)
    
    return [precision, recall]