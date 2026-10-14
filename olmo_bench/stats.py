import statistics


def summarize(values):
    values = [float(v) for v in values]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    else:
        q1 = q3 = values[0]
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "iqr": q3 - q1,
        "mean": mean,
        "cv": std / mean if mean else 0.0,
    }
