import numpy as np
import matplotlib.pyplot as plt
import random,math

algorithm = 'spams_lars'

if algorithm in ['sklearn_cd']:
    from sklearn import linear_model
elif algorithm in ['spams_lars', 'spams_fista']:
    import spams
elif algorithm in ['my_fista']:
    pass
else:
    raise "Specify Algorithm"

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

def lasso_sklearn_cd(Y, A, lambda1):
    estimator = linear_model.Lasso(
            alpha=lambda1
            ,max_iter=1000
            )
    estimator.fit(A,Y)
    return estimator.coef_

def lasso_spams_lars(Y, A, lambda1):
    Y = Y.reshape((Y.shape[0], 1), order='F')
    A = np.asfortranarray(A,dtype=np.float64)
    n = Y.shape[0]
    W = spams.lasso(Y, D=A, return_reg_path=False, lambda1=n*lambda1)
    r = []
    for i in range(W.shape[0]):
        r.append(W[i,0].item())
    return r

def lasso_spams_fista(Y, A, lambda1):
    Y = Y.reshape((Y.shape[0], 1), order='F')
    A = np.asfortranarray(A,dtype=np.float64)
    n = Y.shape[0]
    params = {
            'loss': 'square',
            'regul': 'l1',
            'verbose': False,
            'lambda1': n*lambda1,
            'it0': 10,
            'max_it': 10000,
            #'L0': 1e-1,
            #'tol': 1e-3,
            #'intercept': False,
            #'pos' : False
            }
    W0 = np.zeros((A.shape[1], Y.shape[1]), dtype=np.float64, order='F')
    W = spams.fistaFlat(Y, A, W0, False, **params)
    r = []
    for i in range(W.shape[0]):
        r.append(W[i,0].item())
    return r

def lasso_my_fista(Y, A, lambda1):
    def l1_prox(lambda1, v):
        r = np.zeros(v.shape, dtype=np.float64)

        for i in xrange(v.shape[0]):
            vi = v[i]
            vi_abs = abs(vi)
            if vi_abs > lambda1:
                sign = vi / vi_abs
                r[i] = (vi_abs-lambda1) * sign

        return r

    def lasso_fista(y,A,w0,lambda1,max_iter=None, eps=1e-5):
        e = np.full(y.shape[0], eps)
        w_cur = w0
        z_cur = w0
        s_cur = 0.
        # Lipschits constant
        L = np.linalg.norm(np.dot(A.transpose(), A)) / lambda1
        i = 0

        while True:
            if max_iter and i >= max_iter:
                break

            v = z_cur + (1. / (L * lambda1)) * np.dot(A.transpose(), y - np.dot(A, z_cur))
            w_succ = l1_prox((1. / L), v)
            s_succ = 0.5 * (1. + math.sqrt(1. + 4. * s_cur * s_cur))
            z_succ = w_succ + ((s_cur-1)/s_succ) * (w_succ - w_cur)

            if i % 2 == 0 and np.all(np.absolute(w_succ - w_cur) < e):
                w_cur = w_succ
                break

            w_cur = w_succ
            z_cur = z_succ
            i += 1

        print "end at {} th iter".format(i)
        return w_cur

    n = Y.shape[0]
    Y = Y.reshape((Y.shape[0], 1))
    W0 = np.zeros((A.shape[1], Y.shape[1]), dtype=np.float64)
    W = lasso_fista(Y, A, W0, n*lambda1, max_iter=10000)
    r = []
    for i in range(W.shape[0]):
        r.append(W[i,0].item())
    return r

def estimate_by_lasso(xs, ys, dim, lambda1):
    def make_input(xs,ys,dim):
        Y = np.array(ys, dtype=np.float64)
        a = []
        for x in xs:
            row = map(lambda d: math.pow(x,d), xrange(dim, -1, -1))
            a.append(row)
        A = np.asfortranarray(a, dtype=np.float64)
        return Y,A

    Y,A = make_input(xs, ys, dim)

    if algorithm == 'sklearn_cd':
        return lasso_sklearn_cd(Y, A, lambda1)
    elif algorithm == 'spams_lars':
        return lasso_spams_lars(Y, A, lambda1)
    elif algorithm == 'spams_fista':
        return lasso_spams_fista(Y, A, lambda1)
    elif algorithm == 'my_fista':
        return lasso_my_fista(Y, A, lambda1)
    else:
        raise "Specify Algorithm"

def calc_and_plot(title, func_true, x_min, x_max, x_delta, noise_sigma,
        lsm_dims=[2,10], lasso_dim=20, lasso_lambda=0.1, ylim=None, legend_loc='upper left',
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

    coef_lasso = estimate_by_lasso(xs,ys,lasso_dim,lasso_lambda)
    func_lasso = lambda x: np.poly1d(coef_lasso)(x)
    plot_curve(func_lasso, "Lasso(d=%d)" % (lasso_dim), x_min, x_max, 0.01)

    plt.legend(loc=legend_loc)


plt.figure(num=None, figsize=(8,12), dpi=80)

func_true  = lambda x: -math.pow(x,10)+x
func_true2 = lambda x: math.sin(x)

np.random.seed(1)
lasso_lambda = 1e-30
calc_and_plot(r"$-x^{10}+x$", func_true, 0., 1., 0.05, 0.05, subplot_param=(2,1,1), ylim=(0.0, 1.0), lasso_dim=20, lasso_lambda=lasso_lambda)
calc_and_plot(r"$\sin x$", func_true2, 0., math.pi*2, 0.2, 0.05, subplot_param=(2,1,2), ylim=(-1.3, 1.3), legend_loc='lower left', lsm_dims=[4,6], lasso_dim=20, lasso_lambda=lasso_lambda)

plt.savefig('image.png')
