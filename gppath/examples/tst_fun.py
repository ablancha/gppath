import numpy as np
from matplotlib import pyplot as plt
from gpsearch import Bird, Michalewicz, RosenbrockModified, GaussianInputs
from gppath import PathPlanner, OptimalPath, OptimalPathStatic, AugmentedInputs
from gppath.core.metrics import *


def augment(cls):
    """Add `time` to kwargs list."""
    class New(cls):
        @staticmethod
        def _myfun(x, *args, time=0, **kwargs):
            return super(New,New)._myfun(x) 
    return New


def main():

    np.random.seed(2)

    noise_var = 1e-3
    b = augment(Bird)(noise_var=noise_var, rescale_X=True)
    my_map, inputs, true_ymin, true_xmin = b.my_map, b.inputs, b.ymin, b.xmin

# Use Gaussian prior instead

#   domain = inputs.domain
#   mean = np.zeros(inputs.input_dim) + 0.5
#   cov = np.ones(inputs.input_dim)*0.01
#   inputs = GaussianInputs(domain, mean, cov)

# Set up path-planner

    record_time = np.linspace(0,2,31)
   #record_time = np.linspace(0,4,61)
   #record_time = np.linspace(0,10,151)
    my_map.kwargs["time"] = record_time[0]
    X_pose = (0,0,np.pi/4); X = X_pose[0:2]
    Y_pose = my_map.evaluate(X)
    p = PathPlanner(inputs.domain, 
                    look_ahead=0.2, 
                    turning_radius=0.02)

# Run mission

    o = OptimalPath(X_pose, Y_pose, my_map, inputs)
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
    ZZ = yy.reshape( (ngrid,)*ndim ).T 
    plt.contourf(X, Y, ZZ); 

    r_list = [ np.array(p.make_itinerary(path,1000)[0]) for path in p_list ]
    for ii in range(0, len(m_list)):
        model = m_list[ii]
       #print(model.X)
        stamp = model.X[-1,-1]*np.ones(len(pts))
        yy = model.predict( np.hstack((pts, stamp[:,None])) )[0]
        ZZ = yy.reshape( (ngrid,)*ndim ).T
        plt.contourf(X, Y, ZZ);
        for rr in r_list[0:ii+1]:
            plt.plot(rr[:,0], rr[:,1], 'r-')
        plt.plot(model.X[:,0],model.X[:,1],'ro')
        plt.xlim(0,1); plt.ylim(0,1)
        plt.axis('equal');
        plt.show(); 


if __name__ == "__main__":
    main()



