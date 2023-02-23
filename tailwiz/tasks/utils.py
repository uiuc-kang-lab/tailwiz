from sklearn import metrics


def metrics_at_recall(recall, probs, labels):
    res = None
    for t in range(10000, -1, -1):
        thresh = t / 10000
        preds = [1 if prob > thresh else 0 for prob in probs]
        r = metrics.recall_score(labels, preds, zero_division=0)
        if r >= recall:
            a = metrics.accuracy_score(labels, preds)
            p = metrics.precision_score(labels, preds, zero_division=0)
            f1 = metrics.f1_score(labels, preds, zero_division=0)
            res = {
                'acc': a,
                'prec': p,
                'rec': r,
                'f1': f1,
            }
            break
    return res
