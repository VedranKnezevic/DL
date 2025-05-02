import numpy as np


param_iter = 1000
param_delta = 0.5

def logreg_train(X, Y_):
    W = np.random.randn(X.shape[1], Y_.shape[1])
    b = np.zeros((Y_.shape[1], 1))

    for i in range(param_iter):
        scores = X @ W + np.ones((X.shape[0], Y_.shape[1])) @ b   # N x C
        expscores = np.exp(scores) # N x C

        # nazivnik sofmaksa
        sumexp = np.reshape(np.sum(expscores, axis=1), (-1, 1))   # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss = np.sum(Y_.T @ -logprobs) # scalar

        # dijagnostički ispis
        if i % 100 == 0:
          print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_   # N x C
        
        # gradijenti parametara
        grad_W = (1/X.shape[0]) * dL_ds.T @ X    # C x D (ili D x C)
        grad_b = (1/X.shape[0]) * dL_ds.T @ np.ones((X.shape[0], 1))    # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W.T
        b += -param_delta * grad_b

    return W, b

# mode can be given as probs -> gives back NxC matrix, max_prob -> gives back Nx1 vector which contains max probability
# for each row, and it can be integer between 0 and C-1, class returns Nx1 of int classes
def logreg_classify(X, W, b, mode="probs"):
    scores = X @ W + np.ones((X.shape[0], b.shape[0])) @ b
    expscores = np.exp(scores)
    sumexp = np.reshape(np.sum(expscores, axis=1), (-1, 1))
    probs = expscores / sumexp 
    if mode == "probs":
        return probs
    elif mode == "max_prob":
        return np.reshape(np.max(probs, axis=1), (-1, 1))
    elif mode == "class":
        return np.reshape(np.argmax(probs, axis=1), (-1, 1))
    elif isinstance(mode, int) and mode <= W.shape[0] and mode >= 0:
        return probs[:, mode]
    
def logerg_decfun(W, b, mode="probs"):
    def classify(X):
        return logreg_classify(X, W, b, mode)
    return classify
