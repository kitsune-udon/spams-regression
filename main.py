import numpy as np
import matplotlib.pyplot as plt
import random,math

algorithm = 'my_cd'

if algorithm in ['sklearn_cd']:
    from sklearn import linear_model
elif algorithm in ['spams_lars', 'spams_fista']:
    import spams
elif algorithm in ['my_fista', 'my_cd']:
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
            ,max_iter=10000
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

        #print "end at {} th iter".format(i)
        return w_cur

    n = Y.shape[0]
    Y = Y.reshape((Y.shape[0], 1))
    W0 = np.zeros((A.shape[1], Y.shape[1]), dtype=np.float64)
    W = lasso_fista(Y, A, W0, n*lambda1, max_iter=10000)
    r = []
    for i in range(W.shape[0]):
        r.append(W[i,0].item())
    return r

def lasso_my_cd(Y,A,lambda1):
    def lasso_cd(y,X,b0,lambda1,max_iter=1000, eps=1e-5):
        def st_op(lambda1,x):
            x_abs = abs(x)
            sign = x / x_abs
            return sign * max((x_abs - lambda1), 0.)

        e = np.full(X.shape[1], eps)
        b_cur = b0
        i = 0
        while True:
            if max_iter and i >= max_iter:
                break

            b_succ = np.array(b_cur)

            for j in xrange(X.shape[1]):
                r_j = y - np.dot(X, b_succ) + X[:,j].reshape((X.shape[0],1)) * b_succ[j,0]
                x_j = X[:,j]
                st_arg = np.dot(x_j.transpose(), r_j) / np.dot(x_j.transpose(), x_j)
                b_j = st_op(lambda1, st_arg) # (1,1) matrix
                b_succ[j,0] = b_j

            if np.all(np.absolute(b_succ - b_cur) < e):
                b_cur = b_succ
                break

            b_cur = b_succ
            i += 1

        #print "end at {} th iter".format(i)
        return b_cur

    n = A.shape[0]
    Y = Y.reshape((A.shape[0], 1))
    W0 = np.zeros((A.shape[1], 1), dtype=np.float64)
    W = lasso_cd(Y, A, W0, n*lambda1, max_iter=10000, eps=1e-3)
    return W.ravel().tolist()

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
    elif algorithm == 'my_cd':
        return lasso_my_cd(Y, A, lambda1)
    else:
        raise "Specify Algorithm"

def calc_and_plot(plot_param):
    if plot_param["subplot_param"]:
        plt.subplot(*plot_param["subplot_param"])

    if plot_param["ylim"]:
        plt.ylim(*plot_param["ylim"])

    plt.title(plot_param["title"])
    x_min, x_max = plot_param["xlim"]
    xs = plot_param["x_gen"](x_min, x_max)
    ys = generate_values(xs, plot_param["func_true"], plot_param["noise_sigma"])

    plot_dots(xs,ys)
    plot_curve(plot_param["func_true"], 'True', x_min, x_max, 0.01)

    for d in plot_param["lsm_dims"]:
        coef_lsm   = estimate_by_lsm(xs, ys, d)
        func_lsm   = lambda x: np.poly1d(coef_lsm)(x)
        plot_curve(func_lsm, "LSM(d={})".format(d), x_min, x_max, 0.01)

    coef_lasso = estimate_by_lasso(xs,ys,plot_param["lasso_dim"],plot_param["lasso_lambda"])
    func_lasso = lambda x: np.poly1d(coef_lasso)(x)
    plot_curve(func_lasso, "Lasso(d={})".format(plot_param["lasso_dim"]), x_min, x_max, 0.01)

    plt.legend(loc=plot_param["legend_loc"])


plt.figure(num=None, figsize=(15,5), dpi=80)

func_true  = lambda x: -math.pow(x,10)+x
func_true2 = lambda x: math.sin(x)

np.random.seed(1)
lasso_lambda = 1e-8
plot_param = {
        "title" : r"$-x^{10}+x$",
        "xlim" : (0., 1.),
        "x_gen" : lambda x_min, x_max: np.arange(x_min, x_max, 0.05),
        "noise_sigma" : 0.05,
        "subplot_param" : (1,2,1),
        "ylim" : (0., 1.),
        "func_true" : func_true,
        "lasso_dim" : 2,
        "lasso_lambda" : lasso_lambda,
        "legend_loc" : "upper left",
        "lsm_dims" : [4,6],
        "lasso_dim" : 20,
        }
plot_param2 = {
        "title" : r"$\sin x$",
        "xlim" : (- math.pi,  math.pi),
        "x_gen" : lambda x_min, x_max: np.arange(x_min, x_max, 0.5),
        "noise_sigma" : 0.05,
        "subplot_param" : (1,2,2),
        "ylim" : (-1.3, 1.3),
        "func_true" : func_true2,
        "lasso_dim" : 2,
        "lasso_lambda" : lasso_lambda,
        "legend_loc" : "upper left",
        "lsm_dims" : [4,6],
        "lasso_dim" : 20,
        }
calc_and_plot(plot_param)
calc_and_plot(plot_param2)

#plt.savefig('image.png')
plt.savefig('image_'+algorithm+'.png')
