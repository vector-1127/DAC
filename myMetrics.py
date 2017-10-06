from sklearn import metrics
import numpy as np


def ac(y_true,y_pred):
    sign = np.zeros((np.max(y_true)+1,np.max(y_true)+1))
    for i in range(sign.shape[0]):
        index = np.where(y_pred==i)
        for j in range(sign.shape[1]):
            sign[i,j] = np.sum(y_true[index]==j)
    return sign

def pre(x):
    n = x.shape[0]
    min_r = np.min(x,axis = 1)
    min_r.shape = n,1
    x -= np.repeat(min_r,n,axis = 1)
    
    min_l = np.min(x,axis = 0)
    min_l.shape = 1,n
    x -= np.repeat(min_l,n,axis = 0)
    return x

def select_delete(x):
    sign = 1*(x==0)
    select = np.zeros(x.shape)
    delete = np.zeros(x.shape)
    while np.sum(sign)!=0:
        nb_zeros = np.sum(x==0,axis = 1)
        nb_zeros[np.where(nb_zeros==0)]=100000
        locx = np.argmin(nb_zeros)
        index = np.where(x[locx,:]==0)
        locy = index[0][0]
        select[locx,locy]=1
        delete[locx,np.where(x[locx,:]==0)]=1
        delete[np.where(x[:,locy]==0),locy]=1
        delete[locx,locy]=0
        x[np.where(delete==1)] = -1
        x[np.where(select==1)] = -1
        sign = 1*(x==0)
    return select,delete

def row_line(x,select,delete):
    row = np.zeros(x.shape[0],)
    line = np.zeros(x.shape[0],)
    row[np.where(np.sum(select,axis = 0)==0)]=1
    sign_yt = np.unique(np.where(x[np.where(row==1),:]==0))
    sign_y = sign_yt[0]
    line_temp = np.copy(line)
    line_temp[sign_y]=1
    sign_xt = np.where(x[:,np.where(np.sign(line+line_temp)==1)]==0)
    sign_x = sign_xt[0]
    row_temp = np.copy(row)
    row_temp[sign_x]=1
    while np.sum(line==np.sign(line+line_temp))+np.sum(row==np.sign(row+np.sign(row+row_temp)))!=2*x.shape[0]:
        row = np.sign(row+np.sign(row+row_temp))
        line = np.sign(line+line_temp)
        row[np.where(np.sum(select,axis = 0)==0)]=1
        sign_yt = np.unique(np.where(x[np.where(row==1),:]==0))
        sign_y = sign_yt[0]
        line_temp = np.copy(line)
        line_temp[sign_y]=1
        sign_xt = np.where(x[:,np.where(np.sign(line+line_temp)==1)]==0)
        sign_x = sign_xt[0]
        row_temp = np.copy(row)
        row_temp[sign_x]=1
    return row,line

def adjust(x,row,line):
    res = np.copy(x)
    adjt = x[row==1,:]
    adjt = adjt[:,line==0]
    adj = np.min(adjt)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (row[i]==1) & (line[j]==0):
                res[i,j] -= adj
            if (row[i]==0) & (line[j]==1):
                res[i,j] += adj
    return res

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
    matfn="mnistResult11.mat"
    f = sio.loadmat(matfn)
    acc = f['acc']
    output = f['highOutput']
    y_true = f['y_true']
    y_pred = np.argmax(output[10],axis = 1)
    y_true.shape = 70000
    print(NMI(y_true,y_pred))
    print(ARI(y_true,y_pred))
    print(ACC(y_true,y_pred))
    
    matfn="stlResult1.mat"
    f = sio.loadmat(matfn)
    acc = f['acc']
    output = f['highOutput']
    y_true = f['y_true']
    y_pred = np.argmax(output[2],axis = 1)
    y_true.shape = 13000
    print(NMI(y_true,y_pred))
    print(ARI(y_true,y_pred))
    print(ACC(y_true,y_pred))


