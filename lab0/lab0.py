import numpy as np
import matplotlib.pyplot as plt
import binlogreg
import data
import logreg
    

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return np.reshape(scores, (-1, 1))
    
    


if __name__=="__main__":
    np.random.seed(42)
    # get the training dataset
    n_class = 3
    X,Y_ = data.sample_gmm_2d(3, 3, 30)
    # plt.scatter(X[:,0], X[:,1], c=Y_)
    # plt.show()
    # exit()
    # get the class predictions
    W, b = logreg.logreg_train(X, Y_)
    Y = np.round(logreg.logreg_classify(X, W, b))
    # Y_numbers = np.reshape(np.argmax(Y_, axis=1), (-1, 1))
    # print(np.sum(Y == Y_))
    # print(Y_[:3, :])
    # print(Y_numbers)

    # exit()
    accuraccy, conf_mat, precisions, recalls = data.eval_perf_multi(np.reshape(
                                                    np.argmax(Y, axis=1), (-1, 1)),
                                                    np.reshape(np.argmax(Y_, axis=1), (-1, 1)))
    # print(conf_mat)
    # print(accuraccy)
    # print(precisions)
    # print(recalls)

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(logreg.logerg_decfun(W, b, "class"), bbox)
    data.graph_data(X, np.reshape(np.argmax(Y_, axis=1), (-1, 1)), np.reshape(np.argmax(Y, axis=1), (-1, 1))) 
  
    # show the results
    plt.show()


