from sklearn import metrics
import numpy as np


def NMI(y_true,y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def ARI(y_true,y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)

def ACC(y_true,y_pred):
    Y_pred = y_pred
    Y = y_true
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind

if __name__=='__main__':
    


