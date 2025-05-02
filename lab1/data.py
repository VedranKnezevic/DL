import numpy as np
import matplotlib.pyplot as plt
import torch # type: ignore

maxx, minx, maxy, miny = 5, -5, 5, -5
scalecov = 5

def sample_gmm_2d(K, C, N, corr=None):
    Xs = []
    Ys = []
    for i in range(K):
        mi = np.array([np.random.random_sample() * (maxx - minx) + minx, np.random.random_sample() * (maxy - miny) + miny])
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
        sigma = r_mat.T @ d_mat @ r_mat

        Xs.append(np.random.multivariate_normal(mi, sigma, N))
        Ys.append(np.random.randint(C))
        # Ys.append(i)

    X = np.vstack([X for X in Xs])
    Y_ = np.hstack([[Y] * N for Y in Ys])

    return X, Y_


def graph_surface(fun, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width) 
    lsh = np.linspace(rect[0][0], rect[1][0], height)

    xx0, xx1 = np.meshgrid(lsh, lsw)

    grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)
    
    values=fun(grid).reshape((width,height))
    
    delta = offset if offset else 0
    maxval=max(np.max(values)-delta, - (np.min(values)-delta))

    plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval, cmap="gist_ncar", shading="auto")
    
    if offset != None:
        plt.contour(xx0, xx1, values, colors="black", levels=[offset])

# Y_ and Y can't be one_hot encoded, they need to be vectors of integers
def graph_data(X, Y_, Y, special=[]):
    palette=([1,1,1], [0.5,0.5,0.5], [0.2,0.2,0.2], [0,0,0])
    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    for i in range(len(palette)):
      colors[Y_==i] = palette[i]

    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40
    
    hit = (Y_==Y)
    plt.scatter(X[hit,0],X[hit,1], c=colors[hit], s=sizes[hit], marker='o', edgecolors='black')

    miss = (Y_!=Y)
    plt.scatter(X[miss,0],X[miss,1], c=colors[miss], s=sizes[miss], marker='s', edgecolors='black')



def generate_linreg_data(N, corr):
    mi = np.array([np.random.random_sample() * (maxx - minx) + minx, np.random.random_sample() * (maxy - miny) + miny])
    sigma_x = (np.random.random_sample()*(maxx - minx)/5)**2
    sigma_y = (np.random.random_sample()*(maxy - miny)/5)**2
    sigma = np.array([
        [sigma_x, corr*np.sqrt(sigma_x)*np.sqrt(sigma_y)],
        [corr*np.sqrt(sigma_x)*np.sqrt(sigma_y), sigma_y]
    ])

    data = np.random.multivariate_normal(mi, sigma, N)
    return torch.tensor(data[:, 0]), torch.tensor(data[:, 1])


# confusion matrix is rows = predictions, columns = ground truth
def eval_perf_multi(Y, Y_):
    dim = np.unique(Y_).shape[0]

    confusion_matrix = np.zeros((dim, dim))
    for i in range(dim):
        confusion_matrix[i, i] = np.sum(np.logical_and(Y == Y_, Y_==i))
        for j in range(dim):
            if j != i:
                confusion_matrix[i, j] = np.sum(np.logical_and(np.logical_and(Y != Y_, Y_==i), Y==j))

    accuracy = np.sum([confusion_matrix[i, i] for i in range(dim)]) / Y_.shape[0]

    precisions = np.array([confusion_matrix[i,i] for i in range(dim)]) / np.sum(confusion_matrix, axis=1)

    recalls = np.array([confusion_matrix[i,i] for i in range(dim)]) / np.sum(confusion_matrix, axis=0)

    return accuracy, confusion_matrix, precisions, recalls


def graph_svm_data(X, Y_, Y, support):
    palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    for i in range(len(palette)):
        colors[Y_==i] = palette[i]    


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = sample_gmm_2d(4, 2, 30)
    print(Y_.shape)
    plt.scatter(X[:, 0], X[:, 1], c=Y_, cmap="Greys", edgecolors="black")
    plt.show()

