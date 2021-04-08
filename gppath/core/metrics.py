import numpy as np
from gpsearch import recommend, custom_KDE, funmin


def mll(m_list, inputs, pts=None, y_list=None, t_list=None):
    """Mean log loss as defined in (23) of Merchant and Ramos, ICRA 2014.

    Parameters
    ----------
    m_list : list
        A list of GPy models generated by `OptimalDesign`.
    inputs : instance of `Inputs`
        The input space.
    pts : array_like
        Sampled points used for RMSE computation.
    y_list : list
        Output values of the true map at `pts` and various time instants
        corresponding to `record_time`.
       
    Returns
    -------
    res : list
        A list containing the values of the MLL for each model 
        in `m_list`. 

    """
    res = np.zeros(len(m_list))
    for ii, model in enumerate(m_list):
        time = t_list[ii]
        aug_pts = np.hstack((pts, time*np.ones((pts.shape[0],1)))) 
        mu, var = model.predict(aug_pts)
        mu, var = mu.flatten(), var.flatten()
        yy = y_list[ii].flatten()
        res[ii] = 0.5 * np.mean( np.log(2*np.pi*var) + (mu-yy)**2/var )
    return res


def rmse(m_list, inputs, pts=None, y_list=None, t_list=None):
    """Root-mean-square error between GP model and objective function.

    Parameters
    ----------
    m_list : list
        A list of GPy models generated by `OptimalDesign`.
    inputs : instance of `Inputs`
        The input space.
    pts : array_like
        Sampled points used for RMSE computation.
    y_list : list
        Output values of the true map at `pts` and various time instants
        corresponding to `record_time`.
       
    Returns
    -------
    res : list
        A list containing the values of the RMSE for each model 
        in `m_list`. 

    """
    res = np.zeros(len(m_list))
    for ii, model in enumerate(m_list):
        time = t_list[ii]
        aug_pts = np.hstack((pts, time*np.ones((pts.shape[0],1)))) 
        mu = model.predict(aug_pts)[0]
        diff = mu.flatten() - y_list[ii].flatten()
        res[ii] = np.sqrt(np.mean(np.square(diff)))
    return res
    

def log_pdf(m_list, inputs, pts=None, pt_list=None, clip=True, t_list=None):
    """Log-error between estimated pdf and true pdf.

    Parameters
    ----------
    m_list : list
        A list of GPy models generated by `OptimalDesign`.
    inputs : instance of `Inputs`
        The input space.
    pts : array_like
        Randomly sampled points used for KDE of the GP model.
    pt_list : list of instances of `FFTKDE` 
        The true pdf for time instants corresponding to `record_time`.
    clip : boolean, optional
        Whether or not to clip the pdf values below machine-precision.
       
    Returns
    -------
    res : list
        A list containing the values of the log-error for each model 
        in `m_list`. The log-error is defined as
            e = \int | log(pdf_{GP}) - log(pdf_{true}) | dy 

    """

    res = np.zeros(len(m_list))

    for ii, model in enumerate(m_list):

        time = t_list[ii]
        aug_pts = np.hstack((pts, time*np.ones((pts.shape[0],1)))) 

        mu = model.predict(aug_pts)[0].flatten()
        ww = inputs.pdf(pts)
        pb = custom_KDE(mu, weights=ww)
        pt = pt_list[ii]

        x_min = min( pb.data.min(), pt.data.min() )
        x_max = max( pb.data.max(), pt.data.max() )
        rang = x_max-x_min
        x_eva = np.linspace(x_min - 0.01*rang,
                            x_max + 0.01*rang, 1024)

        yb, yt = pb.evaluate(x_eva), pt.evaluate(x_eva)
        log_yb, log_yt = np.log(yb), np.log(yt)

        if clip: # Clip to machine-precision
            np.clip(log_yb, -14, None, out=log_yb)
            np.clip(log_yt, -14, None, out=log_yt)

        log_diff = np.abs(log_yb-log_yt)
        noInf = np.isfinite(log_diff)
        res[ii] = np.trapz(log_diff[noInf], x_eva[noInf])

    return res


def regret_tmap(m_list, inputs, true_ymin_list=None, tmap=None, t_list=None):
    """Immediate regret using objective function.

    Parameters
    ----------
    m_list : list
        A list of GPy models generated by `OptimalDesign`.
    inputs : instance of `Inputs`
        The input space.
    true_ymin : list
        The minimum values of the objective function arranged
        in a list for each time instant in `record_time`.
    tmap : instance of `BlackBox`
        The black box.
       
    Returns
    -------
    res : list
        A list containing the values of the immediate regret for each 
        model in `m_list` using the black-box objective function:
            $r(n) = f(x_n) - y_{true}$
        where f is the black box, x_n the algorithm recommendation at 
        iteration n, and y_{true} the minimum of the objective function.

    """
    res = np.zeros(len(m_list))
    for ii, model in enumerate(m_list):
        time = t_list[ii]
        x_min = recommend(model, inputs, time)
        tmap.kwargs["time"] = time
        y_min = tmap.evaluate(x_min, include_noise=False)
        res[ii] = y_min - true_ymin_list[ii]
    return res


def distmin_model(m_list, inputs, true_xmin_list=None, t_list=None):
    """Distance to minimum using surrogate GP model.
    
    Parameters
    ----------
    m_list : list
        A list of GPy models generated by `OptimalDesign`.
    inputs : instance of `Inputs`
        The input space.
    true_xmin : list
        The locations of the minima of the objective function arranged
        in a list for each time instant in `record_time`.
       
    Returns
    -------
    res : list
        A list containing the values of the distance to minimum for each 
        model in `m_list` using the surrogate GP model:
            $\ell(n) = \Vert x_n - x_{true} \Vert^2$
        where x_n is the algorithm recommendation at iteration n, and 
        x_{true} the location of the minimum of the objective function.
        When more than one global minimum exists, we compute the 
        distance to each minimum and report the smallest value.

    """
    res = np.zeros(len(m_list))
    for ii, model in enumerate(m_list):
        time = t_list[ii]
        x_min = recommend(model, inputs, time)
        l2_dist = [ np.linalg.norm(x_min - true) 
                    for true in true_xmin_list[ii] ]
        res[ii] = min(l2_dist)
    return res


def recommend(model, inputs, time, num_restarts=10, parallel_restarts=False):
    """Compute recommendation for where minimum is located.
    
    Parameters
    ----------
    model : instance of `GPRegression`
        A GPy model.
    inputs : instance of `Inputs`
        The input space.
    num_restarts : int, optional
        Number of restarts for the optimizer. 
    parallel_restarts : boolean, optional
        Whether or not to solve the optimization problems in parallel.
       
    Returns
    -------
    x_min : array
        The recommendation for where the GP model believes the global 
        minimum is located.

    """
    if parallel_restarts:
        set_worker_env()
    x_min = funmin(compute_mean,
                   compute_mean_jac,
                   inputs,
                   args=(model, time),
                   num_restarts=num_restarts,
                   parallel_restarts=parallel_restarts,
                   init_method="sample_fun")
    return x_min


def compute_mean(x, model, time):
    x = np.atleast_2d(x)
    x = np.hstack((x, time * np.ones((x.shape[0],1))))
    mu, _ = model.predict(x)
    return mu.flatten()


def compute_mean_jac(x, model, time):
    x = np.atleast_2d(x)
    x = np.hstack((x, time * np.ones((x.shape[0],1))))
    mu_jac, _ = model.predictive_gradients(x)
    return mu_jac[:,0:2,0]



