import numpy as np
import copy
from matplotlib import pyplot as plt

# Model: dot_x(t) = f(x(t),u(t),t)

class RungeKutta(object):

    def __init__(self,f,X0,H,N=1):
        self.f = f
        self.X = X0
        self.H = H
        self.N = N
        return

    def update(self,u,x):
        dH = self.H / self.N
        for i in np.arange(self.N):
            K1 = dH*self.f(self.X,u,x)
            K2 = dH*self.f(self.X+K1/2,u,x+dH/2)
            K3 = dH*self.f(self.X+K2/2,u,x+dH/2)
            K4 = dH*self.f(self.X+K3,u,x+dH)
            self.X += 1/6.0*(K1+2.0*K2+2.0*K3+K4)
        return

    def state(self):
        return copy.copy(self.X)

def test_func(X,u,x):
    # dy/dx = exp(x)
    return np.exp(x)

def main():
    H = 0.1
    cycle = 100
    rk4_method = RungeKutta(test_func,1,H,1)
    
    x_list = [0]
    y_list = [1]
    y_true_list = [np.exp(x_list[-1])]
    for i in np.arange(cycle):
        rk4_method.update(0,x_list[-1])
        y = rk4_method.state()
        y_list.append(y)
        x = x_list[-1] + H
        x_list.append(x)

        # y_true
        y_true = np.exp(x)
        y_true_list.append(y_true)

    plt.plot(x_list,y_list,label='RK4 Numerical solution')
    plt.plot(x_list,y_true_list,label='Analytical solution')
    plt.legend()
    plt.show()



# if __name__ == '__main__':
    # main()