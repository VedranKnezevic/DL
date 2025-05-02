import numpy as np

param_iter = 100
param_delta = 0.5
    

def binlogreg_train(X, Y_):
    w = np.random.randn(X.shape[1], 1)
    b = 0
    
    for i in range(param_iter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b * np.ones((X.shape[0], 1))  # N x 1
        
    
        # vjerojatnosti razreda c_1
        probs = 1 / (1 + np.exp(-scores)) # N x 1

        # gubitak
        loss  = np.sum(-Y_ * np.log(probs) - (1 - Y_) * np.log(1 - probs))  # scalar

        # dijagnostički ispis
        # if i % 10 == 0:
        #   print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores =  probs - Y_    # N x 1
        # gradijenti parametara
        grad_w = (1/X.shape[0]) * dL_dscores.T @ X   # D x 1
        grad_b = (1/X.shape[0]) * dL_dscores.T @ np.ones(X.shape[0])   # 1 x 1
        # poboljšani parametri

        w += -param_delta * grad_w.T
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))

def binlogreg_decfun(w,b):
    def classify(X):
      return binlogreg_classify(X, w,b)
    return classify