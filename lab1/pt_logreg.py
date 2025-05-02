import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import data
import matplotlib.pyplot as plt


class PTLogreg(nn.Module):
    def __init__(self, D, C, param_lambda):
        """Arguments:
        - D: dimensions of each datapoint 
        - C: number of classes
        - lambda: regliarization factor
        """

        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn((D, C), requires_grad=True))
        self.b = nn.Parameter(torch.zeros(C, requires_grad=True))

    def forward(self, X):
        scores = torch.mm(X, self.W) + self.b
        return torch.softmax(scores, dim=1)

    def get_loss(self, X, Yoh_):
        Y = self.forward(X)
        regularization = torch.linalg.matrix_norm(self.W)
        return -torch.sum(Yoh_ * torch.log(Y + 1e-20), dim=1).mean() + regularization


def train(model, X, Yoh_, param_niter, param_delta):
    optimizer = torch.optim.SGD(model.parameters(), lr = param_delta)
    for i in range(param_niter):
        Y = model.forward(X)
        loss = model.get_loss(X, Yoh_)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f'step: {i}, loss:{loss}')


def eval(model, X):
    """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
    """
    out = model.forward(torch.Tensor(X))
    return out.detach().numpy()


def pt_logreg_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Yoh_ = data.sample_gmm_2d(6, 3, 20)
    # print(np.unique(np.reshape(Yoh_, (-1, 1))))
    # exit()
    one_hot_Y_ = np.zeros((Yoh_.size, Yoh_.max()+1), dtype=int)
    one_hot_Y_[np.arange(Yoh_.size), Yoh_] = 1
    # definiraj model:
    ptlr = PTLogreg(X.shape[1], one_hot_Y_.shape[1], 1)
    # print(Yoh_)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, torch.Tensor(X), torch.Tensor(one_hot_Y_), 1000, 1)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    # print(probs)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)

    accuracy, conf_matrix, precisions, recalls = data.eval_perf_multi(np.reshape(Y, (-1, 1)), np.reshape(Yoh_, (-1, 1)))

    print(conf_matrix)
    print(f"precisions: {precisions}")
    print(f"recalls: {recalls}")
    # iscrtaj rezultate, decizijsku plohu

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(pt_logreg_decfun(ptlr), bbox, offset=0.5)
    print(Yoh_)
    data.graph_data(X, Yoh_, Y)
    plt.show()

