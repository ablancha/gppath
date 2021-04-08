import numpy as np
from gpsearch import (Branin, Ackley, GaussianInputs,
                      Himmelblau, BraninModified, RosenbrockModified, 
                      UrsemWaves, Michalewicz, Bukin, Bird, custom_KDE)
from gppath import PathPlanner, OptimalPath, BenchmarkerPath
from gppath.core.metrics import *


def augment(cls):
    """Add `time` to kwargs list."""
    class New(cls):
        @staticmethod
        def _myfun(x, *args, time=0, **kwargs):
            return super(New,New)._myfun(x)
    return New


def run(function, prefix):

    noise_var = 1e-3

    b = augment(function)(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

    # Use Gaussian prior
    domain = inputs.domain
    mean = np.zeros(inputs.input_dim) + 0.5
    cov = np.ones(inputs.input_dim)*0.01
    inputs = GaussianInputs(domain, mean, cov)

    record_time = np.linspace(0,15,226)
    n_trials = 50
    n_jobs = 20

    pts = inputs.draw_samples(n_samples=int(1e5), sample_method="uni")
    yy = my_map.evaluate(pts, parallel=True, include_noise=False)
    y_list = [ yy ] * len(record_time)
    pt_list = [ custom_KDE(yy, weights=inputs.pdf(pts)) ] * len(record_time)
    true_ymin_list = [ true_ymin ] * len(record_time)
    true_xmin_list = [ true_xmin ] * len(record_time)

    metric = [ ( rmse,          dict(pts=pts, y_list=y_list, t_list=record_time) ),
               ( mll,           dict(pts=pts, y_list=y_list, t_list=record_time) ),
               ( log_pdf,       dict(pts=pts, pt_list=pt_list, t_list=record_time) ),
               ( distmin_model, dict(true_xmin_list=true_xmin_list, t_list=record_time) ),
               ( regret_tmap,   dict(true_ymin_list=true_ymin_list, tmap=my_map, t_list=record_time) ) ]

    X_pose = (0, 0, np.pi/4)
    planner = PathPlanner(inputs.domain, 
                          look_ahead=0.2, 
                          turning_radius=0.02)

    acq_list = ["US_IW", "US_LW", "IVR_IW", "IVR_LW"]

    for acq in acq_list:
        print("Benchmarking " + acq)
        b = BenchmarkerPath(my_map, acq, planner, X_pose, record_time, inputs, metric)
        result = b.run_benchmark(n_trials, n_jobs=n_jobs, filename=prefix+acq)


if __name__ == "__main__":
    run(Ackley, "ackley_"); 
    run(Bird, "bird_")
    run(Bukin, "bukin_")
    run(Michalewicz, "micha_")
    run(RosenbrockModified, "rosenmod_")

