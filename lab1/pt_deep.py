import torch 
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


class PTDeep(nn.Module):

    def __init__(self, layers=[2, 3, 3], activation=nn.ReLU()):
        super(PTDeep, self).__init__()
        
        self.activation = activation
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(layers[i], layers[i+1], requires_grad=True)) for i in range(len(layers)-1)])
        self.biasies = nn.ParameterList([nn.Parameter(torch.zeros(layers[i], requires_grad=True)) for i in range(1, len(layers))])

    def forward(self, X):
        for i in range(len(self.weights)-1): # here is len - 2 because we need to apply the softmax manually at the end
            X = self.activation(torch.mm(X, self.weights[i]) + self.biasies[i])
        return torch.softmax(torch.mm(X, self.weights[-1]) + self.biasies[-1], dim=1)
    
    def count_params(self):
        for i in range(len(self.weights)):
            if i == 0:
                print(f"{'X':6s} ... (?, {self.weights[0].shape[0]})")
            print(f"W{i+1:<5} ... ({self.weights[i].shape[0]}, {self.weights[i].shape[1]})")
            print(f"b{i+1:<5} ... (1, {self.biasies[i].shape[0]})")
            if i == len(self.weights) - 1:
                print(f"{'probs':6s} ... (?, {self.weights[i].shape[1]})")


def eval(model, X):
    out = model.forward(torch.Tensor(X))
    return out.detach().numpy()

# Y_ doesn't need to be one hot encoded because I am using nn.CrossEntropyLoss
def train(model, X, Y_, param_niter=1e4, param_delta=0.1, param_lambda=1e-4):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = param_delta, weight_decay=param_lambda)
    for i in range(param_niter):
        Y = model.forward(X)

        loss = loss_fn(Y, Y_)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if i % (param_niter/10) == 0:
            print(f'step: {i}, loss:{loss}')

def pt_deep_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)

if __name__=="__main__":
    np.random.seed(100)
    ptdeep = PTDeep([2, 10, 10, 2], nn.ReLU())
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # ptdeep.count_params()
    # print(ptdeep.weights[-1].shape)
    # exit()

    train(ptdeep, torch.Tensor(X), torch.tensor(Y_, dtype=torch.int64), 10000, 0.1, 1e-4)

    probs = eval(ptdeep, X)
    Y = np.argmax(probs, axis=1)

    accuracy, conf_matrix, precisions, recalls = data.eval_perf_multi(np.reshape(Y, (-1, 1)), np.reshape(Y_, (-1, 1)))
    print(f"accuracy: {accuracy}")
    print(f"precisions: {precisions}")
    print(f"recalls: {recalls}")
    AP = average_precision_score(Y_, np.max(probs,axis=1))
    print(f"AP: {AP}")

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(pt_deep_decfun(ptdeep), bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()
    