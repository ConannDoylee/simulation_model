import numpy as np
from matplotlib import pyplot as plt

from ode_euler_method import Euler
from ode_runge_kutta_method import RungeKutta

def test_func(X,u,x):
    # dy/dx = exp(x)
    return np.exp(x)

H = 0.1
cycle = 100
rk4_method = RungeKutta(test_func,1,H,1)
euler_method = Euler(test_func,1,H,1)

x_list = [0]
y_true_list = [np.exp(x_list[-1])]
y_euler_list = [1]
y_rk4_list = [1]

for i in np.arange(cycle):
    # rk4
    rk4_method.update(0,x_list[-1])
    y = rk4_method.state()
    y_rk4_list.append(y)
    # euler
    euler_method.update(x_list[-1],0)
    y = euler_method.state()
    y_euler_list.append(y)

    x = x_list[-1] + H
    x_list.append(x)
    # y_true
    y_true = np.exp(x)
    y_true_list.append(y_true)

rt4_error_list = [rk4-t for rk4,t in zip(y_rk4_list,y_true_list)]
eu_error_list = [eu-t for eu,t in zip(y_euler_list,y_true_list)]

plt.figure()
plt.subplot(2,1,1)
plt.title('dy/dx=exp(x)')
plt.plot(x_list,y_euler_list,label='Euler Numerical solution')
plt.plot(x_list,y_rk4_list,label='RK4 Numerical solution')
plt.plot(x_list,y_true_list,label='Analytical solution')
plt.legend()
plt.subplot(2,1,2)
plt.plot(x_list,eu_error_list,label='euler error')
plt.plot(x_list,rt4_error_list,label='kr4 error')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()