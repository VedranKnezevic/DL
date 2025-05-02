import sklearn.svm as svm
import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.clf.fit(X, Y_)
        self.support = self.clf.support_

    def predict(self, X):
        return self.clf.predict(X)
    
    def get_scores(self, X):
        return self.clf.predict_proba(X)
    

def ksvm_decfun(svm):
    return lambda X: svm.get_scores(X)[:, 0]

if __name__=="__main__":
    X, Y_ = data.sample_gmm_2d(6, 2, 30)

    clf = KSVMWrap(X, Y_)

    Y = clf.predict(X)

    support_for_graph = np.array([i in clf.support for i in range(X.shape[0])])

    accuracy, conf_matrix, precisions, recalls = data.eval_perf_multi(np.reshape(Y, (-1, 1)), np.reshape(Y_, (-1, 1)))
    print(f"accuracy: {accuracy}")
    print(f"precisions: {precisions}")
    print(f"recalls: {recalls}")
    AP = average_precision_score(Y_, np.max(clf.get_scores(X), axis=1))
    print(f"AP: {AP}")


    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(ksvm_decfun(clf), bbox, offset=0.5)
    data.graph_data(X, Y_, Y, support_for_graph)
    plt.show()
