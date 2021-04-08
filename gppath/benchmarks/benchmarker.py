import numpy as np
from joblib import Parallel, delayed
from ..core.optimalpath import OptimalPath
from gpsearch.core.utils import set_worker_env
from gpsearch import Benchmarker


class BenchmarkerPath(Benchmarker):
    """A class for benchmark of path planning algorithm.

    Parameters
    ----------
    tmap, acquisition, inputs, metric : see parent class `Benchmarker`
    path_planner : instance of `PathPlanner`
        Path parametrization.
    X_pose : array_like
        The robot's starting pose. Must be in the form 
        (x_pos, y_pos, angle)
    record_time : array_like
        Time vector for when measurements are made.

    Attributes
    ----------
    tmap, acquisition, inputs, metric, path_planner, X_pose, 
        record_time : see Parameters

    """


    def __init__(self, tmap, acquisition, path_planner, X_pose,
                 record_time, inputs, metric):
        super().__init__(tmap, acquisition, 0, 0, inputs, metric)
        self.path_planner = path_planner
        self.X_pose = X_pose
        self.record_time = record_time

    def _run_benchmark(self, n_trials, n_jobs=20):
        if n_jobs > 1:
            set_worker_env()
        x = Parallel(n_jobs=n_jobs, backend="loky", verbose=10) \
                    ( delayed(self.optimization_loop)(self.tmap,
                                                      self.acquisition,
                                                      self.path_planner,
                                                      self.X_pose,
                                                      self.record_time,
                                                      self.inputs,
                                                      self.metric,
                                                      ii) \
                      for ii in range(n_trials) )
        return np.array(x)

    @staticmethod
    def optimization_loop(tmap, acquisition, path_planner, X_pose,
                          record_time, inputs, metric, seed=None):
        np.random.seed(seed)
        tmap.kwargs["time"] = 0
        Y_pose = tmap.evaluate(X_pose[0:2])
        np.random.seed(seed)
        o = OptimalPath(X_pose, Y_pose, tmap, inputs,
                        fix_noise=False,
                        noise_var=None,
                        normalize_Y=True)
        s_list = o.optimize(record_time, acquisition, path_planner,
                            callback=False, save_iter=False)[2]
        result = [ met(s_list, inputs, **kwa) for (met, kwa) in metric ]
        return result

