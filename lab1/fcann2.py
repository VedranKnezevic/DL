import numpy as np
import matplotlib.pyplot as plt
import data

max_iter = 40
hidden_layer_size = 3
lr = 1

def ReLu(x):
    return x * (x > 0)

def dReLu(x):
    return 1 * (x > 0)


# the function presumes the Y_ ground truth is already one hot encoded
def fcann2_train(X, Y_):
    W1 = np.random.randn(X.shape[1], hidden_layer_size)
    b1 = np.zeros((hidden_layer_size, 1))
    W2 = np.random.randn(hidden_layer_size, Y_.shape[1])
    b2 = np.zeros((Y_.shape[1], 1))
    for i in range(max_iter):
        s1 = X @ W1 + b1.T
        h1 = ReLu(s1)
        s2 = h1 @ W2 + b2.T
        exps2 = np.exp(s2)
        sumexps = np.reshape(np.sum(exps2, axis=1), (-1, 1))
        probs = exps2 / sumexps
        

        logprobs = np.log(probs)
        loss = (1/X.shape[0]) * np.sum(Y_ * -logprobs) 
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Gs2 = probs - Y_

        grad_W2 = (1/X.shape[0]) * Gs2.T @ h1
        grad_b2 = (1/X.shape[0]) * Gs2.T @ np.ones((X.shape[0], 1))


        Gh1 = Gs2 @ W2.T
        Gs1 = Gh1 * dReLu(s1)

        grad_W1 = (1/X.shape[0]) * Gs1.T @ X
        grad_b1 = (1/X.shape[0]) * Gs1.T @ np.ones((X.shape[0], 1))

        W1 += - lr * grad_W1.T
        b1 += - lr * grad_b1
        W2 += - lr * grad_W2.T
        b2 += - lr * grad_b2
        
    return W1, b1, W2, b2
        
        
def fcann2_decfun(W1, b1, W2, b2, mode="class"):
    def classify(X):
        return fcann2_classify(X, W1, b1, W2, b2, mode)
    return classify

# mode parameter
# "probs" -> N x C matrix of probabilites
# "class" -> N x 1 matrix of class integers
# integer < C -> N x 1 matrix of probabilites for given class
# "max_prob" -> N x 1 matrix of max probability in each row
def fcann2_classify(X, W1, b1, W2, b2, mode="probs"):
    s1 = X @ W1 + b1.T
    h1 = ReLu(s1)
    s2 = h1 @ W2 + b2.T
    exps2 = np.exp(s2)
    sumexps = np.reshape(np.sum(exps2, axis=1), (-1, 1))
    probs = exps2 / sumexps
    if mode == "probs":
        return probs
    elif mode == "max_prob":
        return np.reshape(np.max(probs, axis=1), (-1, 1))
    elif mode == "class":
        return np.reshape(np.argmax(probs, axis=1), (-1, 1))
    elif isinstance(mode, int) and mode < W2.shape[1] and mode >= 0:
        return probs[:, mode]


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 4, 30)

    one_hot_Y_ = np.zeros((Y_.size, Y_.max()+1), dtype=int)
    one_hot_Y_[np.arange(Y_.size), Y_] = 1
    print(np.unique(Y_))
    W1, b1, W2, b2 = fcann2_train(X, one_hot_Y_)
    
    Y = fcann2_classify(X, W1, b1, W2, b2, mode="class")
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(fcann2_decfun(W1, b1, W2, b2), bbox, offset=0.5)
    data.graph_data(X, Y_, np.squeeze(Y))
    plt.show()
    
