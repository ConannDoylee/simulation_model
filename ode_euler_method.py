import numpy as np
from matplotlib import pyplot as plt
import copy

class Euler(object):

    def __init__(self):
        return
    
    def __init__(self,f,X0,H,N=1):
        self.f = f
        self.X = X0
        self.H = H
        self.N = N
        return

    def update(self,x,u):
        dH = self.H / self.N
        tmp = x
        for i in np.arange(self.N):
            self.X += dH * self.f(self.X,u,tmp)
            tmp += dH
        return

    def state(self):
        return copy.copy(self.X)


def test_func(X,u,x):
    # dy/dx = exp(x)
    return np.exp(x)

def main():
    H = 0.1
    cycle = 100
    euler_method = Euler(test_func,1,H,1)
    
    x_list = [0]
    y_list = [1]
    y_true_list = [np.exp(x_list[-1])]
    for i in np.arange(cycle):
        euler_method.update(x_list[-1],0)
        y = euler_method.state()
        y_list.append(y)
        x = x_list[-1] + H
        x_list.append(x)

        # y_true
        y_true = np.exp(x)
        y_true_list.append(y_true)

    plt.plot(x_list,y_list,label='Euler Numerical solution')
    plt.plot(x_list,y_true_list,label='Analytical solution')
    plt.legend()
    plt.show()



# if __name__ == '__main__':
    # main()
