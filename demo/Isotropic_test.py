"""
Homogeneous turbulence. See [1] for initialization and [2] for a section
on forcing the lowest wavenumbers to maintain a constant turbulent
kinetic energy.

[1] R. S. Rogallo, "Numerical experiments in homogeneous turbulence,"
NASA TM 81315 (1981)

[2] A. G. Lamorgese and D. A. Caughey and S. B. Pope, "Direct numerical simulation
of homogeneous turbulence with hyperviscosity", Physics of Fluids, 17, 1, 015106,
2005, (https://doi.org/10.1063/1.1833415)

"""
from __future__ import print_function
import warnings
import numpy as np
from numpy import pi, zeros, sum
from shenfun import Function
from shenfun.fourier import energy_fourier
from spectralDNS import config, get_solver, solve

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def get_turbulence_params(Re, uEta = 1.0):
    C = 3.0 # np.sqrt(196.0/20.0) #2.87657077
    # K = 2/3.0 * C * np.sqrt(15)
    eps = (Re/np.sqrt(60))**(3./2.)
    # eps = np.power(uEta*uEta * Re / K, 3.0/2.0)
    nu = np.power(uEta, 4) / eps

    L_k = (nu**3/eps)**0.25
    T_k = np.sqrt(nu/eps)
    U_k = (nu*eps)**0.25

    return eps, nu, L_k, T_k, U_k

def init_from_file(filename, solver, context):
    f = h5py.File(filename, driver="mpio", comm=solver.comm)
    assert "0" in f["U/3D"]
    U_hat = context.U_hat
    s = context.T.local_slice(True)

    U_hat[:] = f["U/3D/0"][:, s[0], s[1], s[2]]
    if solver.rank == 0:
        U_hat[:, 0, 0, 0] = 0.0

    if 'VV' in config.params.solver:
        context.W_hat = solver.cross2(context.W_hat, context.K, context.U_hat)

    context.target_energy = energy_fourier(U_hat, context.T)

    f.close()

def initialize(solver, context):
    c = context
    # Create mask with ones where |k| < Kf2 and zeros elsewhere
    kf = config.params.Kf2
    c.k2_mask = np.where(c.K2 <= kf**2, 1, 0)
    np.random.seed(solver.rank)
    k = np.sqrt(c.K2)
    k = np.where(k == 0, 1, k)
    kk = c.K2.copy()
    kk = np.where(kk == 0, 1, kk)
    k1, k2, k3 = c.K[0], c.K[1], c.K[2]
    ksq = np.sqrt(k1**2+k2**2)
    ksq = np.where(ksq == 0, 1, ksq)

    E0 = np.sqrt(9./11./kf*c.K2/kf**2)*c.k2_mask
    E1 = np.sqrt(9./11./kf*(k/kf)**(-5./3.))*(1-c.k2_mask)
    Ek = E0 + E1
    # theta1, theta2, phi, alpha and beta from [1]
    theta1, theta2, phi = np.random.sample(c.U_hat.shape)*2j*np.pi
    alpha = np.sqrt(Ek/4./np.pi/kk)*np.exp(1j*theta1)*np.cos(phi)
    beta = np.sqrt(Ek/4./np.pi/kk)*np.exp(1j*theta2)*np.sin(phi)
    c.U_hat[0] = (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    c.U_hat[1] = (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    c.U_hat[2] = beta*ksq/k
    c.mask = c.T.get_mask_nyquist()
    c.T.mask_nyquist(c.U_hat, c.mask)

    solver.get_velocity(**c)
    U_hat = solver.set_velocity(**c)

    K = c.K
    # project to zero divergence
    U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2])*c.K_over_K2

    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0.0

    # Scale to get correct kinetic energy. Target from [2]
    energy = 0.5*energy_fourier(c.U_hat, c.T)
    # target = config.params.Re_lam*(config.params.nu*config.params.kd)**2/np.sqrt(20./3.)
    target = config.params.eps_forcing**(2/3.0)*3.0
    c.U_hat *= np.sqrt(target/energy)

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    config.params.t = 0.0
    config.params.tstep = 0
    c.target_energy = energy_fourier(c.U_hat, c.T)

    print(target,c.target_energy)

def L2_norm(comm, u):
    r"""Compute the L2-norm of real array a

    Computing \int abs(u)**2 dx

    """
    N = config.params.N
    result = comm.allreduce(np.sum(u**2))
    return result/np.prod(N)

def spectrum(solver, context):
    c = context
    uiui = np.zeros(c.U_hat[0].shape)
    uiui[..., 1:-1] = 2*np.sum((c.U_hat[..., 1:-1]*np.conj(c.U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((c.U_hat[..., 0]*np.conj(c.U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((c.U_hat[..., -1]*np.conj(c.U_hat[..., -1])).real, axis=0)
    uiui *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/3))
    bins = np.array(range(0, Nb))+0.5
    z = np.digitize(np.sqrt(context.K2), bins, right=True)

    # Sample
    Ek = np.zeros(Nb)
    ll = np.zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = np.where((z > k0) & (z <= k))
        ll[i] = len(ii[0])
        Ek[i] = (k**3 - k0**3)*np.sum(uiui[ii])
        # Ek[i] = np.sum(uiui[ii])

    Ek = solver.comm.allreduce(Ek)
    ll = solver.comm.allreduce(ll)
    for i in range(Nb):
        if not ll[i] == 0:
            Ek[i] = Ek[i] / ll[i]

    E0 = uiui.mean(axis=(1, 2))
    E1 = uiui.mean(axis=(0, 2))
    E2 = uiui.mean(axis=(0, 1))

    return Ek, bins, E0, E1, E2

k = []
w = []
kold = zeros(1)
im1 = None
energy_new = None
def update(context):
    global k, w, im1, energy_new
    c = context
    params = config.params
    solver = config.solver
    curl_hat = Function(c.VT, buffer=c.work[(c.U_hat, 2, True)])

    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0

    if params.solver == 'VV':
        c.U_hat = solver.cross2(c.U_hat, c.K_over_K2, c.W_hat)

    energy_new = energy_fourier(c.U_hat, c.T) # Sum of squares, no 1/2 factor
    energy_lower = energy_fourier(c.U_hat*c.k2_mask, c.T)
    energy_old = energy_new

    # Constant energy forcing
    if params.forcing_mode == 'constant_E':
        energy_upper = energy_new - energy_lower
        alpha2 = (c.target_energy - energy_upper) /energy_lower
        alpha = np.sqrt(alpha2)
    else: # Constant rate forcint
        alpha = (1 + params.eps_forcing/energy_lower)*params.dt

    c.U_hat *= (alpha*c.k2_mask + (1-c.k2_mask))

    energy_new = energy_fourier(c.U_hat, c.T)

    if params.forcing_mode == 'constant_E':
        assert np.sqrt((energy_new-c.target_energy)**2) < 1e-7, np.sqrt((energy_new-c.target_energy)**2)

    if params.solver == 'VV':
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    if (params.tstep % params.compute_energy == 0 or
            params.tstep % params.plot_step == 0 and params.plot_step > 0):
        solver.get_velocity(**c)
        solver.get_curl(**c)
        if 'NS' in params.solver:
            solver.get_pressure(**c)

    K = c.K # local wave numbers
    if plt is not None:
        if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
            #div_u = solver.get_divergence(**c)

            if not plt.fignum_exists(1):
                plt.figure(1)
                #im1 = plt.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
                im1 = plt.contourf(c.X[1][..., 0], c.X[0][..., 0], c.U[0, ..., 10], 100)
                plt.colorbar(im1)
                plt.draw()
            else:
                im1.ax.clear()
                #im1.ax.contourf(c.X[1][:,:,0], c.X[0][:,:,0], div_u[:,:,10], 100)
                im1.ax.contourf(c.X[1][..., 0], c.X[0][..., 0], c.U[0, ..., 10], 100)
                im1.autoscale()
            plt.pause(1e-6)

    if params.tstep % params.compute_spectrum == 0:
        Ek, _, _, _, _ = spectrum(solver, context)
        f = h5py.File(context.spectrumname, driver='mpio', comm=solver.comm)
        f['Turbulence/Ek'].create_dataset(str(params.tstep), data=Ek)
        f.close()

    if params.tstep % params.compute_energy == 0:
        dx, L = params.dx, params.L

        # Estimate of the dissipation using the norm of the Jacobian
        duidxj = np.zeros(((3, 3)+c.U[0].shape), dtype=c.float) # Jacobian?
        for i in range(3):
            for j in range(3):
                duidxj[i, j] = c.T.backward(1j*K[j]*c.U_hat[i], duidxj[i, j]) # Derivative in Fourier space
        eps_l2J = L2_norm(solver.comm, duidxj)*params.nu # Forebius norm of the jacobian is the entstrophy nu

        # Estimate of the dissipation using the derivative of the RHS
        ddU = np.zeros(((3,)+c.U[0].shape), dtype=c.float)
        dU = solver.ComputeRHS(c.dU, c.U_hat, solver, **c)
        for i in range(3):
            ddU[i] = c.T.backward(dU[i], ddU[i])
        eps_rhs = solver.comm.allreduce(sum(ddU*c.U))/np.prod(params.N)

        # Compute dissipation from rate of change of energy
        e_current = 0.5*L2_norm(solver.comm, c.U)
        eps_dEdt = (energy_new-energy_old)/2/params.dt

        # Estimate of the dissipation using the norm of the vorticity
        curl_hat = solver.cross2(curl_hat, K, c.U_hat)
        dissipation = energy_fourier(curl_hat, c.T) # Entstrophy
        eps_l2vort = dissipation*params.nu

        # Compute Re number from dissipation and forcing
        eps_forcing = params.eps_forcing
        kk = 0.5*energy_new
        Re_lam_eps_dissipation = np.sqrt(20*kk**2/(3*params.nu*eps_l2vort))
        Re_lam_eps_forcing = np.sqrt(20*kk**2/(3*params.nu*eps_forcing))


        if solver.rank == 0:
            k.append(energy_new)
            w.append(dissipation)

            if params.tstep % (params.compute_energy*10) == 0:
                print(' Tstep Time   Energy       eps_forcing  eps_l2vort   eps_l2J      eps_rhs      eps_dEdt     Re_dissip    Re_forcing')            
            print('{tstep:6d} {t:.4f} {e_current:.6e} {eps_forcing:.6e} {eps_l2vort:.6e} {eps_l2J:.6e} {eps_rhs:.6e} {eps_dEdt:.6e} {Re_lam_eps_dissipation:.6e} {Re_lam_eps_forcing:.6e}'.format(
                    tstep=params.tstep,t=params.t, e_current=e_current, eps_forcing=eps_forcing, eps_l2vort=eps_l2vort, 
                    eps_l2J=eps_l2J, eps_rhs=eps_rhs, eps_dEdt=eps_dEdt, 
                    Re_lam_eps_dissipation=Re_lam_eps_dissipation, Re_lam_eps_forcing=Re_lam_eps_forcing),flush=True)

            turb_qty = {'E':e_current,'eps_forcing':eps_forcing,'eps_l2vort':eps_l2vort,
                        'eps_l2J':eps_l2J,'eps_rhs':eps_rhs,'eps_dEdt':eps_dEdt,
                        'Re_dissip':Re_lam_eps_dissipation,
                        'Re_forcing':Re_lam_eps_forcing
                        }

            f = h5py.File(context.spectrumname)
            f['Turbulence/TurbQty'].create_dataset(str(params.tstep), data=str(turb_qty))
            f.close()

if __name__ == "__main__":
    import h5py
    
    # Parameters
    config.update(
        {'nu': 0.005428,              # Viscosity (not used, see below)
         'dt': 0.002,                 # Time step
         'T': 5,                      # End time
         'L': [2.*pi, 2.*pi, 2.*pi],
         'checkpoint': 100,
         'write_result': 1e8,
         'dealias': '3/2-rule',
        }, "triplyperiodic"
    )
    config.triplyperiodic.add_argument("--N", default=[64, 64, 64], nargs=3,
                                       help="Mesh size. Trumps M.")
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=10)
    config.triplyperiodic.add_argument("--compute_spectrum", type=int, default=10)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=1000)
    config.triplyperiodic.add_argument("--Kf2", type=int, default=3)
    config.triplyperiodic.add_argument("--kd", type=float, default=50.)
    config.triplyperiodic.add_argument("--Re_lam", type=float, default=100.)

    # Define solver
    solver = get_solver(update=update, mesh="triplyperiodic")
    context = solver.get_context()

    # Turbulence parameters
    eps, nu, L_k, T_k, U_k = get_turbulence_params(config.params.Re_lam)
    config.params.nu = nu
    config.params.eps_forcing = eps
    config.params.L_k = L_k
    config.params.T_k = T_k
    config.params.U_k = U_k

    config.params.dt = T_k/config.params.N[0] # Set time step to 1/N the kolmogorov time step
    config.params.compute_energy = config.params.N[0]   # Compute energy every komlmogorv time scale
    config.params.compute_spectrum = config.params.N[0] # Compute spectrum every komlmogorv time scale

    # Initialize turbulence
    initialize(solver, context)
    context.hdf5file.filename = "NS_isotropic_{}_{}_{}".format(*config.params.N)

    # Get initial power spectrum
    Ek, bins, E0, E1, E2 = spectrum(solver, context)
    E = context.target_energy

    L_I = 3*np.pi/(4*E)*np.trapz(Ek/bins,x=bins)
    T_I = L_I/np.sqrt(2*E/3)
    config.params.T = 30*T_I

    # Save initial power spectrum
    context.spectrumname = context.hdf5file.filename+".h5"
    f = h5py.File(context.spectrumname, mode='w', driver='mpio', comm=solver.comm)
    f.create_group("Turbulence")
    f["Turbulence"].create_group("Ek")
    f["Turbulence"].create_group("TurbQty")

    bins = np.array(bins)
    f["Turbulence"].create_dataset("bins", data=bins)
    f.close()

    # Advance simulation
    if solver.rank == 0:

        print('Running Homogenou Isotropic Turbulence N={} \n'.format(config.params.N[0]))
        print('Turbulence quantities:')
        print('Re_tau {}, resulting eps {:.5f}, nu {:.5f}'.format(config.params.Re_lam,config.params.eps_forcing,config.params.nu))
        print('Kolmogorov time scale {:.5f}, dt = L_k/N {:.5f}'.format(config.params.T_k,config.params.dt))
        print('Kolmogorov length scale {:.5f} \n'.format(config.params.L_k))
        
        print('From initialization: E={:.5f}'.format(E))
        print('Integral length scale {:.5f} time scale {:.5f}'.format(L_I,T_I),flush=True)
        print('Total simulation time = 30*L_I = {:.5f}, total time steps {}'.format(config.params.T,config.params.T/config.params.dt),flush=True)
        print('Scalings:')
        print('L/eta {:.5f} Re^3/4 {:.5f}'.format(L_I/config.params.L_k,config.params.Re_lam**(3./4)))
        print('T_I/T_k {} Re^1/2 {:.5f} \n'.format(T_I/config.params.T_k,config.params.Re_lam**(1./2)))

        print('Running simulations:')

        print(' Tstep Time   Energy       eps_forcing  eps_l2vort   eps_l2J      eps_rhs       eps_dEdt     Re_dissip    Re_forcing')            

    solve(solver, context)

    # Save simulation
    from mpi4py_fft import generate_xdmf
    if solver.rank == 0:
        generate_xdmf(context.hdf5file.filename+"_w.h5")
