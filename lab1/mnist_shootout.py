import torch
import torch.utils
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import data
from tqdm import tqdm
import sklearn.svm as svm


def make_batches(X, Y, batch_size=32):
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    out = []
    batch = []
    for i in index:
        batch.append(i)
        if len(batch) == batch_size:
            out.append([X[batch], Y[batch]])
            batch = []
    if len(batch) != 0:
        out.append([X[batch], Y[batch]])
    return out


def train_mb(model, X_train, X_val, Y_train, Y_val, param_niter=10000, param_delta=0.1, param_lambda=1e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr = param_delta, weight_decay=param_lambda)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    train_loss = []
    val_loss = []
    min_val_loss = None
    early_stop_model = None
    with tqdm(range(param_niter), unit=" epoch") as t:
        for i in t: # epochs 
            batches = make_batches(X_train, Y_train)
            batches_loss = []
            for X_batch, Y_batch in batches:
                Y = model(X_batch)
                loss = F.cross_entropy(Y, Y_batch)
                batches_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_loss.append(np.mean(batches_loss))
            scheduler.step()
            Y = model(X_val)
            loss = F.cross_entropy(Y, Y_val)
            val_loss.append(loss.item())
            t.set_postfix(loss=loss.item())
            if min_val_loss is None or min_val_loss < loss:
                min_val_loss = loss
                early_stop_model = copy.deepcopy(model)

    return early_stop_model, train_loss, val_loss
        
def eval(model, X_test, Y_test):
    Y_ = model(X_test)
    loss = F.cross_entropy(Y_, Y_test)
    
    acc, c_mat, pr, recall = data.eval_perf_multi(np.argmax(Y_.detach().numpy(), axis=1),
                                                   Y_test.detach().numpy())
    return loss.item(), acc, c_mat, pr, recall


def FC_2d_Model(layers=[784, 10], activation=nn.ReLU()):
    return nn.Sequential(
        nn.Flatten(),
        *[nn.Sequential(nn.Linear(layers[i], layers[i+1]), activation) for i in range(len(layers)-2)],
        nn.Linear(layers[-2], layers[-1]),
        nn.Softmax(dim=1)
    )

def singe_loss(model, X_test, Y_test):
    return [F.cross_entropy(model(X_test[i]), Y_test[i]) for i in range(len(X_test))]


if __name__=="__main__":
    dataset_root = '/tmp/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    n = len(mnist_train)
    mnist_list = torch.utils.data.random_split(mnist_train, [int(n * 0.8), n - int(n * 0.8)])
    mnist_train = mnist_list[0]
    mnist_val = mnist_list[1]

    indices_0 = mnist_list[0].indices
    indices_1 = mnist_list[1].indices


    x_train, y_train = mnist_train.dataset.data[indices_0], mnist_train.dataset.targets[indices_0]
    x_val, y_val = mnist_val.dataset.data[indices_1], mnist_val.dataset.targets[indices_1]
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_val, x_test = x_train.float().div_(255.0), x_val.float().div_(255.0), x_test.float().div_(255.0)

    N = x_train.shape[0]
    D = x_train.shape[1] * x_train.shape[2]
    C = y_train.max().add_(1).item()

    
    model = FC_2d_Model([D,  C], nn.ReLU())

    best_model, train_loss, val_loss = train_mb(model, x_train, x_val, y_train, y_val, param_niter=15, param_delta=0.5)

    loss1, acc1, _, prec1, recall1 = eval(best_model, x_test, y_test)

    # fig, ax = plt.subplots(2,5, figsize=(10,10))
    # ax = ax.ravel()
    # weights = best_model[1].weight.detach().numpy()
    # for i in range(10):
    #     ax[i].imshow(weights[i].reshape(28, 28))
    #     ax[i].set_title(f"weights[{i}]")
        
    # plt.show()


    # plt.plot(np.arange(len(train_loss)), train_loss, c='blue', label="train loss", alpha=0.3)
    # plt.plot(np.arange(len(val_loss)), val_loss, c='yellow', label="val loss")
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend(loc="best")
    # plt.show()


    model_bigger = FC_2d_Model([D, 100, C], nn.ReLU())

    best_model_bigger, train_loss, val_loss = train_mb(model_bigger, x_train, x_val, y_train, y_val, param_niter=40)

    # plt.plot(np.arange(len(train_loss)), train_loss, c='blue', label="train loss", alpha=0.3)
    # plt.plot(np.arange(len(val_loss)), val_loss, c='yellow', label="val loss")
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend(loc="best")
    # plt.show()

    loss1, acc1, cmat1, prec1, recall1 = eval(best_model, x_test, y_test)

    loss2, acc2, cmat2, prec2, recall2 = eval(best_model_bigger, x_test, y_test)

    if acc2 > acc1:
        model = best_model_bigger
    else:
        model = best_model

    # loss_per_example = F.cross_entropy(model(x_test), y_test, reduction='none').detach().numpy()
    # index_of_top = np.argsort(loss_per_example)[-10:]


    # fig, ax = plt.subplots(2,5, figsize=(10,10))
    # ax = ax.ravel()
    # for i, ind in enumerate(index_of_top):
    #     ax[i].imshow(x_test[ind].reshape(28, 28), cmap=plt.get_cmap('Greys'))
    #     ax[i].set_title(f"loss={loss_per_example[ind]:2.4f}")
        
    # plt.show()


    svc_x_train = np.reshape(x_train.detach().numpy(), (x_train.shape[0], -1))
    svc_y_train = y_train.detach().numpy()
    svc_x_test = np.reshape(x_test.detach().numpy(), (x_test.shape[0], -1))
    svc_y_test = y_test.detach().numpy()

    clf = svm.SVC(kernel='linear')
    clf.fit(svc_x_train, svc_y_train)
    Y = clf.predict(svc_x_test)

    acc_lin, _, prec_lin, rec_lin = data.eval_perf_multi(np.reshape(Y, (-1, 1)), np.reshape(svc_y_test, (-1, 1)))

    clf = svm.SVC(kernel='rbf')
    clf.fit(svc_x_train, svc_y_train)
    Y = clf.predict(svc_x_test)

    acc_rbf, _, prec_rbf, rec_rbf = data.eval_perf_multi(np.reshape(Y, (-1, 1)), np.reshape(svc_y_test, (-1, 1)))


    print(f"{'':8s} {'model_small':11s} {'model_big':11s} {'svm_lin':11s} {'svm_rbf':11s}")
    print(f"{'loss':8s} {loss1:^11.5f} {loss2:^11.5f} {'':^11s} {'':^11s}")
    print(f"{'accuracy':8s} {acc1:^11.5f} {acc2:^11.5f} {acc_lin:^11.5f} {acc_rbf:^11.5f}")
    # macro averaged precision
    print(f"{'precision':8s} {np.mean(prec1):^11.5f} {np.mean(prec2):^11.5f} {np.mean(prec_lin):^11.5f} {np.mean(prec_rbf):^11.5f}")
    # macro averaged recall
    print(f"{'recall':8s} {np.mean(recall1):^11.5f} {np.mean(recall2):^11.5f} {np.mean(rec_lin):^11.5f} {np.mean(rec_rbf):^11.5f}")
