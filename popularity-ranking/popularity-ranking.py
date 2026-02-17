def popularity_ranking(items, min_votes, global_mean):
    result = []

    for R, v in items:
        WR = (v / (v + min_votes)) * R + (min_votes / (v + min_votes)) * global_mean
        result.append(float(WR))

    return result
