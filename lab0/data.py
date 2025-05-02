import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    def __init__(self, minx=0, maxx=10, miny=0, maxy=10):
        self.mi = np.array([np.random.random_sample() * (maxx - minx) + minx, np.random.random_sample() * (maxy - miny) + miny])
        eigvalx = (np.random.random_sample()*(maxx - minx)/5)**2
        eigvaly = (np.random.random_sample()*(maxy - miny)/5)**2
        d_mat = np.array([
            [eigvalx, 0],
            [0, eigvaly]
        ])
        fi = np.random.random_sample() * np.pi
        r_mat = np.array([
            [np.cos(fi), -np.sin(fi)],
            [np.sin(fi), np.cos(fi)]
        ])
        self.sigma = r_mat.T @ d_mat @ r_mat

    
    def get_sample(self, size):
        return np.random.multivariate_normal(self.mi, self.sigma, size)

def sample_gauss_2d(C, N):
    G = Random2DGaussian()
    X = G.get_sample(N)
    y = np.zeros((N, 1))
    for i in range(1, C):
        G = Random2DGaussian()
        X = np.vstack([X, G.get_sample(N)])
        y = np.vstack([y, np.ones((N, 1)) * i])
    return X, y

def eval_perf_binary(Y,Y_):
    TP = sum(np.logical_and(Y==Y_, Y_==True))
    FN = sum(np.logical_and(Y!=Y_, Y_==True))
    TN = sum(np.logical_and(Y==Y_, Y_==False))
    FP = sum(np.logical_and(Y!=Y_, Y_==False))
    accuracy = (TP + TN) / Y.shape[0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, precision, recall

def eval_AP(Y_):
    precisions = []
    for i in range(len(Y_)):
        if Y_[i, 0] == 0:
            continue
        else:
            Y = np.vstack([np.zeros((i, 1)), np.ones((Y_.shape[0] - i, 1))])
            TP = sum(np.logical_and(Y==Y_, Y_==True))
            FN = sum(np.logical_and(Y!=Y_, Y_==True))
            TN = sum(np.logical_and(Y==Y_, Y_==False))
            FP = sum(np.logical_and(Y!=Y_, Y_==False))
            precision = TP / (TP + FP)
            precisions.append(precision)

    return np.sum(precisions) / np.sum(Y_)

def graph_data(X, Y_, Y):
    X_hit = X[np.squeeze(Y_==Y, 1)]
    X_miss = X[np.squeeze(Y_!=Y, 1)]
    Y_hit = Y_[np.squeeze(Y_==Y, 1)]
    Y_miss = Y_[np.squeeze(Y_!=Y, 1)]
    plt.scatter(X_hit[:, 0], X_hit[:, 1], c=Y_hit, cmap="Greys", marker='o', edgecolors='black')
    plt.scatter(X_miss[:, 0], X_miss[:, 1], c=Y_miss, cmap="Greys", marker='s', edgecolors='black')
    plt.show()

def graph_surface(fun, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width) 
    lsh = np.linspace(rect[0][0], rect[1][0], height)

    xx0, xx1 = np.meshgrid(lsh, lsw)

    grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)
    
    values=fun(grid).reshape((width,height))
    
    delta = offset if offset else 0
    maxval=max(np.max(values)-delta, - (np.min(values)-delta))

    plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval, cmap="gist_ncar")
    
    if offset != None:
        plt.contour(xx0, xx1, values, colors="black", levels=[offset])

def eval_perf_multi(Y, Y_):
    dim = np.unique(Y_).shape[0]

    confusion_matrix = np.zeros((dim, dim))
    for i in range(dim):
        confusion_matrix[i, i] = np.sum(np.logical_and(Y == Y_, Y_==i))
        for j in range(dim):
            if j != i:
                confusion_matrix[i, j] = np.sum(np.logical_and(np.logical_and(Y != Y_, Y_==i), Y==j))

    accuracy = np.sum([confusion_matrix[i, i] for i in range(dim)]) / Y_.shape[0]

    precisions = np.array([confusion_matrix[i,i] for i in range(dim)]) / np.sum(confusion_matrix, axis=0)

    recalls = np.array([confusion_matrix[i,i] for i in range(dim)]) / np.sum(confusion_matrix, axis=1)

    return accuracy, confusion_matrix, precisions, recalls

def sample_gmm_2d(ncomponents, nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs=[]
    Ys=[]
    for i in range(ncomponents):
      Gs.append(Random2DGaussian())
      Ys.append(np.random.randint(nclasses))

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_= np.hstack([[Y]*nsamples for Y in Ys])

    one_hot_Y = np.zeros((Y_.size, Y_.max()+1), dtype=int)
    one_hot_Y[np.arange(Y_.size),Y_] = 1 
    
    return X, one_hot_Y