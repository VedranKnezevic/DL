import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np
import data
import matplotlib.pyplot as plt



def pt_linreg(X, Y, lr, niter):
    ## Definicija računskog grafa
    # podaci i parametri, inicijalizacija parametara
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)


    # optimizacijski postupak: gradijentni spust
    optimizer = optim.SGD([a, b], lr=lr)

    for i in range(100):
        # afin regresijski model
        Y_ = a*X + b

        diff = (Y-Y_)

        # kvadratni gubitak
        loss = torch.mean(diff ** 2)

        # računanje gradijenata
        loss.backward()
        
        grad_a = torch.mean(-2 * diff * X)
        grad_b = torch.mean(-2 * diff)
        if i == 0 :
            print(f"analytical grad_a: {grad_a}, grad_b: {grad_b}")
            print(f"pytorch grad_a: {a.grad}, grad_b: {b.grad}")

        # korak optimizacije
        optimizer.step()

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()

        # if i % 10 == 0 :
        #     print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')

    return a.detach().numpy(), b.detach().numpy()

if __name__=="__main__":
    # np.random.seed(4)
    X, Y = data.generate_linreg_data(100, 0.9,)
    plt.scatter(X.detach().numpy(), Y.detach().numpy())
    a, b = pt_linreg(X, Y, lr=0.1, niter=100)
    plt.plot(X.detach().numpy(), X.detach().numpy()* a + b, c="r")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()