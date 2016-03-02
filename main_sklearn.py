import numpy as np
import matplotlib.pyplot as plt
import random,math
from sklearn import linear_model

def generate_values(xs, func_true, sigma):
    ys = []

    for x in xs:
        y = func_true(x) + np.random.normal(0, sigma)
        ys.append(y)

    return ys

def estimate_by_lsm(xs, ys, dim):
    return np.polyfit(xs, ys, dim)

def plot_dots(xs, ys):
    plt.plot(xs,ys,'*')

def plot_curve(func, label, x_min, x_max, delta):
    xs = np.arange(x_min, x_max, delta)
    ys = map(lambda x: func(x), xs)
    plt.plot(xs, ys, '-', label=label)

def estimate_by_lasso(xs, ys, dim, penalty_order=-6):
    Y = np.array([ys], dtype=np.float64).transpose()
    a = []

    for x in xs:
        row = map(lambda d: math.pow(x,d), range(dim, -1, -1))
        a.append(row)

    A = np.asfortranarray(a, dtype=np.float64)
    estimator = linear_model.Lasso(
            alpha=math.pow(10, penalty_order)
            ,max_iter=1000000
            #,tol=math.pow(10,-5)
            #,selection='random'
            )
    W = estimator.fit(A,Y)
    #print "iter num:" + str(W.n_iter_)
    #print "coef:" + str(W.coef_)
    return W.coef_

def calc_and_plot(title, func_true, x_min, x_max, x_delta, noise_sigma,
        lsm_dims=[2,10], lasso_dim=20, ylim=None, legend_loc='upper left',
        subplot_param=None):
    if subplot_param:
        plt.subplot(*subplot_param)

    if ylim:
        plt.ylim(*ylim)

    plt.title(title)
    xs = np.arange(x_min, x_max, x_delta)
    ys = generate_values(xs, func_true, noise_sigma)

    plot_dots(xs,ys)
    plot_curve(func_true, 'True', x_min, x_max, 0.01)

    for d in lsm_dims:
        coef_lsm   = estimate_by_lsm(xs, ys, d)
        func_lsm   = lambda x: np.poly1d(coef_lsm)(x)
        plot_curve(func_lsm, "LSM(d=%d)" % (d), x_min, x_max, 0.01)

    coef_lasso = estimate_by_lasso(xs,ys,lasso_dim)
    func_lasso = lambda x: np.poly1d(coef_lasso)(x)
    plot_curve(func_lasso, "Lasso(d=%d)" % (lasso_dim), x_min, x_max, 0.01)

    plt.legend(loc=legend_loc)


plt.figure(num=None, figsize=(8,12), dpi=80)

func_true  = lambda x: -math.pow(x,10)+x
func_true2 = lambda x: math.sin(x)

np.random.seed(0)
calc_and_plot(r"$-x^{10}+x$", func_true, 0., 1., 0.05, 0.05, subplot_param=(2,1,1), lasso_dim=20)
calc_and_plot(r"$\sin x$", func_true2, 0., math.pi*2, 0.2, 0.001, subplot_param=(2,1,2), ylim=(-1.3, 1.3), legend_loc='lower left', lsm_dims=[4,6], lasso_dim=20)

plt.savefig('image.png')
