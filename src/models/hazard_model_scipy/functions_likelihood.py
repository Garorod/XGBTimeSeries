def log_likelihood(params, x_names, data_uncens, data_all, verbose=False):
    import numpy as np
    if verbose:
        print('params')
        print(params)

    p, gamma = params[:2]
    beta = params[2:]
    n_uncens = len(data_uncens)
    log_like = n_uncens*np.log(p*gamma) \
               + ((p-1)*np.log(gamma*data_uncens['period_end'])).sum() \
               - (np.log(1+(gamma*data_uncens['period_end'])**p)).sum() \
               + (data_uncens[x_names].dot(beta)).sum()

    log_like += -((np.exp(data_all[x_names].dot(beta)))
                  * (np.log(1+(gamma*data_all['period_end'])**p)
                     - np.log(1+(gamma*data_all['period_begin'])**p))).sum()

    if verbose:
        print('loglike')
        print(log_like)

    return log_like


def get_optimal_params(x_names, data_uncens, data_all):
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    import numpy as np

    f = lambda params: -log_likelihood(params, x_names, data_uncens, data_all)
    res = minimize(fun=f, x0=np.random.normal(size=len(x_names)+2),
                   method='trust-constr', options={'disp': True},  # , 'maxiter': 10},
                   bounds=np.array([[0, np.inf], [0, np.inf]] + [[-np.inf, np.inf] for i in range(len(x_names))]))
    params = res.x
    success = res.success

    return params, success
