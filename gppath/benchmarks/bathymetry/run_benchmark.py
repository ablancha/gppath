import numpy as np
from netCDF4 import Dataset
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from gpsearch import BlackBox, UniformInputs, GaussianInputs, custom_KDE
from gppath import PathPlanner, OptimalPath, BenchmarkerPath
from gppath.core.metrics import *


def map_def(theta, interpolant, time=0):
    return interpolant(theta[0], theta[1])


def run(casename):

    fh = Dataset("../data/" + casename + ".nc", mode="r")
    lons = fh.variables["lon"][:]
    lats = fh.variables["lat"][:]
    depths = fh.variables["elevation"][:]

# Rescale data and flip depths to have anomalies as minima

    lons = MinMaxScaler().fit_transform(lons.reshape(-1, 1))
    lats = MinMaxScaler().fit_transform(lats.reshape(-1, 1))
    tmp = StandardScaler().fit_transform(depths.reshape(-1, 1))
    depths = -tmp.reshape(depths.shape)
    if casename in ["cuba", "challenger", "izu", "izu2", "izu3"]: depths = -depths

    domain = [ [0, 1], [0, 1] ]
    inputs = UniformInputs(domain)
    inputs = GaussianInputs(domain, [0.5,0.5], [0.01,0.01])
    noise_var = 0.0

    interpolant = interpolate.interp2d(lons, lats, depths, kind="cubic")
    my_map = BlackBox(map_def, args=(interpolant,), noise_var=noise_var)
    idx_max = np.unravel_index(depths.argmin(), depths.shape)
    true_xmin = [ lons[idx_max[1]], lats[idx_max[0]] ]
    true_ymin = np.min(depths)
   
    record_time = np.linspace(0,15,226)
    n_trials = 50
    n_jobs = 40

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
        result = b.run_benchmark(n_trials, n_jobs=n_jobs, filename=casename+"_"+acq)


if __name__ == "__main__":
    run("izu3")

