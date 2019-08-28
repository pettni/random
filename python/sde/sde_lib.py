import numpy as np
from fipy import Grid1D, CellVariable, DiffusionTerm, UpwindConvectionTerm, TransientTerm

# Solve PDE problem
def solve_pde(xmin,   # domain min
              xmax,    # domain max
              Tmax,    # time max
              theta=1,   # dynamics drift
              mu=0,      # dynamics stable level
              sigma=1,   # dynamics noise
              dx=0.01,   # space discretization
              dt=0.01,   # time discretization
              gbm=False):# state-dependent drift  

    mesh = Grid1D(dx=dx, nx=(xmax-xmin)/dx) + xmin
    Tsteps = int(Tmax/dt)+1
    
    x_face = mesh.faceCenters
    x_cell = mesh.cellCenters[0]  
    V = CellVariable(name="V", mesh=mesh, value=0.)

    # PDE
    if gbm:
        eq = TransientTerm(var=V) == (DiffusionTerm(coeff=float(sigma)*sigma/2, var=V) 
                                      -UpwindConvectionTerm(coeff=float(theta)*(x_face-float(mu)), var=V) 
                                      + V*(float(theta) - x_cell) )
    else:
        eq = TransientTerm(var=V) == (DiffusionTerm(coeff=float(sigma)*sigma/2, var=V) 
                                      -UpwindConvectionTerm(coeff=float(theta)*(x_face-float(mu)), var=V) 
                                      + V*float(theta) )

    
    # Boundary conditions
    V.constrain(1., mesh.facesRight)
    V.faceGrad.constrain([0.], mesh.facesLeft)

    # Solve by stepping in time
    sol = np.zeros((Tsteps, mesh.nx))
    for step in range(Tsteps):
        eq.solve(var=V, dt=dt)
        sol[step] = V.value
        
    X = mesh.cellCenters.value[0]
    T = dt * np.arange(Tsteps)
    return T, X, sol

def sim_sde_euler(mu, sigma, x0, t):
    n = len(t)

    if type(x0) is np.ndarray:
        x = np.zeros([n, len(x0)])
    else: 
        x = np.zeros([n, 1])

    x[0, :] = x0

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        dx = mu(t[i], x[i, :]) * dt + sigma(t[i], x[i, :]) * np.sqrt(dt) * np.random.randn()
        x[i + 1, :] = x[i, :] + dx
    return x

def sim_sde_milstein(mu, sigma, sds, x0, t):
    # Use milsetin scheme
    # sds(t,x) = sigma(t,x) * dsigma(t,x)

    if sds is None:
        raise Exception("Milstein scheme requires sds to be specified")
    if type(x0) is np.ndarray:
         if len(x0) > 1:
            raise Exception("Milstein scheme does not support multi-dimensional SDE")

    n = len(t)
    x = np.zeros(n)
    x[0] = x0

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        Z = np.random.randn()
        dx = mu(t[i], x[i]) * dt \
             + sigma(t[i], x[i]) * np.sqrt(dt) * Z \
             + 0.5*sds(t[i], x[i])*dt*(Z**2-1)
        x[i + 1] = x[i] + dx
    return x

def sim_mc(mu, sigma, x0, t, num_sim, sds=None, scheme='euler'):
    if scheme == 'euler':
        return [sim_sde_euler(mu, sigma, x0, t) for i in range(num_sim)]
    if scheme == 'milstein':
        return [sim_sde_milstein(mu, sigma, sds, x0, t) for i in range(num_sim)]
    raise Exception("Unknown scheme {}".format(scheme))
        
def sup(trajectory):
    r = np.zeros(len(trajectory))
    r[0] = trajectory[0]
    for i in range(1, len(trajectory)):
        r[i] = max(r[i-1], trajectory[i])
    return r

def sup_statistic(trajectories, l):
    '''calculate P(sup x >= l)'''

    ret = np.zeros(len(trajectories[0]))

    for sup_traj in map(sup, trajectories):
        fidx = np.argmax(sup_traj >= l)
        if fidx > 0:
            ret[fidx:] += 1

    ret /= len(trajectories)

    return ret