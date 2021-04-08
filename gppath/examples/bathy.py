import numpy as np
from matplotlib import pyplot as plt
from gpsearch import BlackBox,GaussianInputs
from gppath import PathPlanner, OptimalPath, OptimalPathStatic, AugmentedInputs
from gppath.core.metrics import *
from netCDF4 import Dataset
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def map_def(theta, interpolant, time=0):
    return interpolant(theta[0], theta[1])


def main():

    np.random.seed(2)
    casename = "izu3"

    fh = Dataset("../benchmarks/bathymetry/data/" + casename + ".nc", mode="r")
    lons = fh.variables["lon"][:]
    lats = fh.variables["lat"][:]
    depths = fh.variables["elevation"][:]

# Rescale data and flip depths to have anomalies as minima

    lons = MinMaxScaler().fit_transform(lons.reshape(-1, 1))
    lats = MinMaxScaler().fit_transform(lats.reshape(-1, 1))
    tmp = StandardScaler().fit_transform(depths.reshape(-1, 1))
    depths = -tmp.reshape(depths.shape)
    if casename in ["cuba", "challenger", "izu", "izu2"]: depths = -depths

    domain = [ [0, 1], [0, 1] ]
   #inputs = UniformInputs(domain)
    inputs = GaussianInputs(domain, [0.5,0.5], [0.01,0.01])
    noise_var = 0.0

    interpolant = interpolate.interp2d(lons, lats, depths, kind="cubic")
    my_map = BlackBox(map_def, args=(interpolant,), noise_var=noise_var)

# Set up path-planner

    record_time = np.linspace(0,2,31)
   #record_time = np.linspace(0,4,61)
   #record_time = np.linspace(0,10,151)
    my_map.kwargs["time"] = record_time[0]
    X_pose = (0,0,np.pi/4); X = X_pose[0:2]
    Y_pose = my_map.evaluate(X); print(Y_pose); 

    p = PathPlanner(inputs.domain, 
                    look_ahead=0.2, 
                    turning_radius=0.02)

# Run mission

  # o = OptimalPathStatic(X_pose, Y_pose, my_map, inputs)
    o = OptimalPath(X_pose, Y_pose, my_map, inputs)#, static=True)
    m_list, p_list, s_list = o.optimize(record_time,
                                        path_planner=p, 
                                        acquisition="US")

# Plot stuff

    ngrid = 50
    pts = inputs.draw_samples(n_samples=ngrid, sample_method="grd")
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    X, Y = grd[0], grd[1]
    yy = my_map.evaluate(pts, include_noise=False)
    ZZt = yy.reshape( (ngrid,)*ndim ).T 

    r_list = [ np.array(p.make_itinerary(path,1000)[0]) for path in p_list ]
    for ii in range(0, len(m_list)):
        model = m_list[ii]
        stamp = model.X[-1,-1]*np.ones(len(pts))
        yy = model.predict( np.hstack((pts, stamp[:,None])) )[0]
        ZZ = yy.reshape( (ngrid,)*ndim ).T

        fig = plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.contourf(X, Y, ZZ);
        for rr in r_list[0:ii+1]:
            plt.plot(rr[:,0], rr[:,1], 'r-')
        plt.plot(model.X[:,0],model.X[:,1],'ro')
        plt.xlim(0,1); plt.ylim(0,1)
        plt.axis('equal');

        plt.subplot(122)
        plt.contourf(X, Y, ZZt);
        plt.xlim(0,1); plt.ylim(0,1)
        plt.axis('equal');
        plt.show(); 


if __name__ == "__main__":
    main()



