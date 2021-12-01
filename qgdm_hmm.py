import numpy as np
import sys

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from scipy.ndimage.interpolation import shift
from dedalus.tools import post
from scipy import integrate


import logging
import subprocess
import pathlib
import time
logger = logging.getLogger(__name__)


# Parameters:
Ekm = 1e-4      # the Ekman number
Ra  = 80        # the reduced Rayleigh number
Pm  = 0.7       # the reduced magnetic Prandtl number
Pr  = 1.0       # the Prandtl number
ik  = 1.3048    # the wavenumber k


# Bases and domain:
L  = 1.0
Nd = 128
z_basis = de.Chebyshev('z', Nd, interval=(0, L))
domain = de.Domain([z_basis], grid_dtype=np.float64)
z = domain.grid(0)

# Auxiliary fields:
Psi_W   = domain.new_field()
Bx_s    = domain.new_field()
By_s    = domain.new_field()
Bx_sproj = domain.new_field()
By_sproj = domain.new_field()


# Run setup:
dt_mic = 5e-4                       # micro step
numic  = 50                         # number of micro steps in micro-solver
dt_mac = numic*dt_mic*2             # macro step
dt_jmp = dt_mac - numic*dt_mic      # projection step


sltime  = 3.0                       # time to reach settled solution
Epsilon = Ekm**(1.0/3.0)            # small parameter
T_final = 40.0                      # final time


Restart = False
ipath   = "/nobackup/QGDM"



# 1. SETTING UP MACRO-SOLVER (solution of equations on the slow scale)
prob_slow = de.IVP(domain, variables = ['Bx', 'By', 'Bx_z', 'By_z'])
prob_slow.parameters['eps']     = Epsilon
prob_slow.parameters['Pm']      = Pm
prob_slow.parameters['Psi_W']   = Psi_W

# Equations:
prob_slow.add_equation("eps**(-3.0/2.0)*dt(Bx) - dz(Bx_z)/Pm = - Pm*dz(Psi_W*By)")
prob_slow.add_equation("eps**(-3.0/2.0)*dt(By) - dz(By_z)/Pm =   Pm*dz(Psi_W*Bx)")
prob_slow.add_equation("Bx_z - dz(Bx) = 0")
prob_slow.add_equation("By_z - dz(By) = 0")

# Boundary conditions:
prob_slow.add_bc("left(Bx) = 0.0")
prob_slow.add_bc("right(Bx) = 0.0")
prob_slow.add_bc("left(By) = 0.0")
prob_slow.add_bc("right(By) = 0.0")

# Build solver:
solverSlow = prob_slow.build_solver(de.timesteppers.RK443)
solverSlow.stop_wall_time = np.inf
solverSlow.stop_iteration = np.inf
solverSlow.stop_sim_time  = np.inf

# Initial conditions:
Bx = solverSlow.state['Bx']
By = solverSlow.state['By']

if Restart:
    write, dt = solverSlow.load_state(ipath + "/analysisSL/analysisSL.h5", -1)
else:
    Bx['g'] = np.sin(np.pi*z)
    By['g'] = np.sin(np.pi*z)





# 2. SETTING UP MICRO-SOLVER (solution of equations on the fast scale)
prob_fast = de.IVP(domain, variables=['Psi', 'W', 'The', 'tet', 'tet_z'])
prob_fast.parameters['Pi']    = np.pi
prob_fast.parameters['eps']   = Epsilon
prob_fast.parameters['Pr']    = Pr
prob_fast.parameters['k']     = ik
prob_fast.parameters['Pm']    = Pm
prob_fast.parameters['Ra']    = Ra
prob_fast.parameters['iBx']   = Bx_s
prob_fast.parameters['iBy']   = By_s

# Equations:
prob_fast.add_equation("dt(Psi) + dz(W)/k**2 + k**2*Psi = - Pm*(iBx**2 + iBy**2)*Psi/2.0")
prob_fast.add_equation("dt(W)   + dz(Psi)   - Ra*The/Pr + k**2*W = - Pm*(iBx**2 + iBy**2)*W/2.0 ")
prob_fast.add_equation("dt(The) + k**2*The/Pr = - W*dz(tet)")
prob_fast.add_equation("dz(tet_z)/Pr = dz(W*The)")
prob_fast.add_equation("tet_z - dz(tet) = 0")

# Boundary conditions:
prob_fast.add_bc("left(tet) = 1.0")
prob_fast.add_bc("right(tet) = 0.0")
prob_fast.add_bc("left(W) = 0.0")
prob_fast.add_bc("right(W) = 0.0")

# Build solver:
solverFast = prob_fast.build_solver(de.timesteppers.RK443)
solverFast.stop_wall_time = np.inf
solverFast.stop_iteration = np.inf
solverFast.stop_sim_time = T_final

# Initial conditions:
Psi = solverFast.state['Psi']
W   = solverFast.state['W']
The = solverFast.state['The']
tet = solverFast.state['tet']

if Restart:
    write, dt = solverFast.load_state(ipath + "/analysisFA/analysisFA.h5", -1)
else:
    Psi['g'] =-np.pi/ik**4*np.cos(np.pi*z)
    The['g'] = 1.0/ik**2*np.sin(np.pi*z)
    tet['g'] = 1-z
    W['g']   = np.sin(np.pi*z)

# Set initial B-fields:
Bx_s['g'] = Bx['g']
By_s['g'] = By['g']




# 3. SETTING UP PROJECTOR SOLVER (replicates the equations from micro-solver)
prob_fastHelp = de.IVP(domain, variables=['Psi', 'W', 'The', 'tet', 'tet_z'])
prob_fastHelp.parameters['eps']   = Epsilon
prob_fastHelp.parameters['Pr']    = Pr
prob_fastHelp.parameters['k']     = ik
prob_fastHelp.parameters['Pm']    = Pm
prob_fastHelp.parameters['Ra']    = Ra
prob_fastHelp.parameters['iBx']   = Bx_sproj
prob_fastHelp.parameters['iBy']   = By_sproj

# Equations:
prob_fastHelp.add_equation("dt(Psi) + dz(W)/k**2 + k**2*Psi = - Pm*(iBx**2 + iBy**2)*Psi/2.0")
prob_fastHelp.add_equation("dt(W)   + dz(Psi)   - Ra*The/Pr + k**2*W = - Pm*(iBx**2 + iBy**2)*W/2.0")
prob_fastHelp.add_equation("dt(The) + k**2*The/Pr = - W*dz(tet)")
prob_fastHelp.add_equation("dz(tet_z)/Pr = dz(W*The)")
prob_fastHelp.add_equation("tet_z - dz(tet) = 0")

# Boundary conditions:
prob_fastHelp.add_bc("left(tet) = 1.0")
prob_fastHelp.add_bc("right(tet) = 0.0")
prob_fastHelp.add_bc("left(W) = 0.0")
prob_fastHelp.add_bc("right(W) = 0.0")

# Build solver:
solverHelper = prob_fastHelp.build_solver(de.timesteppers.SBDF1)
solverHelper.stop_wall_time = np.inf
solverHelper.stop_iteration = np.inf
solverHelper.stop_sim_time = np.inf

# Initial conditions:
Psi_proj = solverHelper.state['Psi']
W_proj   = solverHelper.state['W']
The_proj = solverHelper.state['The']
tet_proj = solverHelper.state['tet']




# Additional parameters:
icount  = 0
Field   = np.zeros([numic, Nd])
timer   = np.zeros([numic])
F_mean  = np.zeros(Nd)
xk = np.linspace(-1.0,1.0,num=numic)


# Kernel definition:
kernel    = np.zeros(numic)
kernel[:] = 15.0/16.0*(1.0 - xk[:]**2)**2
iker = integrate.simps(kernel,x = None,dx = dt_mic)

# Analysis For Macro-solver:
analysisSL = solverSlow.evaluator.add_file_handler('analysisSL', iter=1)
analysisSL.add_task("0.5*integ(Bx**2 + By**2, 'z')", layout='g', name='EM')
analysisSL.add_task("Psi_W", layout='g', name='MeanForce')

# Analysis For Micro-solver:
analysisFA = solverFast.evaluator.add_file_handler('analysisFA', iter=10)
analysisFA.add_task("1 + integ(W*The, 'z')", layout='g', name='NU')
analysisFA.add_task("Psi*W", layout='g', name='Psi_W')

# Additional Analysis:
evolutionSL = solverSlow.evaluator.add_file_handler('evolutionSL', iter=1)
evolutionSL.add_system(solverSlow.state, layout='g')
evolutionFA = solverFast.evaluator.add_file_handler('evolutionFA', iter=10)
evolutionFA.add_system(solverFast.state, layout='g')




# Main loop:
start_time = time.time()
while solverFast.ok:

    # Get settled solution:
    if solverFast.sim_time <= (sltime - dt_mic):

        # Make Fast step:
        solverFast.step(dt_mic)
        Psi_W['g']  = np.copy(Psi['g']*W['g'])

        # Make Slow step:
        solverSlow.step(dt_mic)
        
        # Update B-fields:
        Bx_s['g'] = solverSlow.state['Bx']['g']
        By_s['g'] = solverSlow.state['By']['g']
    
    # HMM-like strategy for solving the simplified multiscale QGDM problem:    
    else:        
        
        solverFast.step(dt_mic)
        Psi_W['g']  = np.copy(Psi['g']*W['g'])

        Field[icount,:] = Psi['g']*W['g']
        timer[icount]   = solverFast.sim_time
        icount +=1

        
        if icount % numic == 0:
            
            icount = 0
            # get mean field on the micro scale:
            for j in range(Nd):
                F_mean[j] = integrate.simps(Field[:, j]*kernel[:], x=None, dx=dt_mic)/iker

            # make slow step:
            Psi_W['g']  = np.copy(F_mean)
            solverSlow.step(dt_mac)
            
            # update slow fields:
            Bx_s['g'] = solverSlow.state['Bx']['g']
            By_s['g'] = solverSlow.state['By']['g']

            # init projection step:
            Psi_proj = solverHelper.state['Psi']
            W_proj   = solverHelper.state['W']
            The_proj = solverHelper.state['The']
            tet_proj = solverHelper.state['tet']
            Psi_proj['g'] = np.copy(Psi['g'])
            W_proj['g']   = np.copy(W['g'])
            The_proj['g'] = np.copy(The['g'])
            tet_proj['g'] = np.copy(tet['g'])
            Bx_sproj['g'] = np.copy(Bx_s['g'])
            By_sproj['g'] = np.copy(By_s['g'])

            # make projection step:
            solverHelper.step(dt_jmp)
            solverFast.sim_time = solverFast.sim_time + dt_jmp
            
            # init micro-solver:
            Psi = solverFast.state['Psi']
            W   = solverFast.state['W']
            The = solverFast.state['The']
            tet = solverFast.state['tet']        
            Psi['g'] = np.copy(Psi_proj['g'])
            W['g']   = np.copy(W_proj['g']) 
            The['g'] = np.copy(The_proj['g'])
            tet['g'] = np.copy(tet_proj['g'])
            Psi_W['g']  = np.copy(Psi_proj['g']*W_proj['g'])

                

    if solverFast.iteration % 200 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solverFast.iteration, solverFast.sim_time, dt_mic))
        

end_time = time.time()
print('Runtime:', end_time-start_time)


# Write runtime:
f = open("runtime_hmm","w")
f.write('%.12e' % (end_time-start_time))
f.close()


# Post analysis:
post.merge_process_files("analysisSL", cleanup=True)
set_paths = list(pathlib.Path("analysisSL").glob("analysisSL_s*.h5"))
post.merge_sets("analysisSL/analysisSL.h5", set_paths, cleanup=True)

post.merge_process_files("evolutionSL", cleanup=True)
set_paths = list(pathlib.Path("evolutionSL").glob("evolutionSL_s*.h5"))
post.merge_sets("evolutionSL/evolutionSL.h5", set_paths, cleanup=True)

post.merge_process_files("analysisFA", cleanup=True)
set_paths = list(pathlib.Path("analysisFA").glob("analysisFA_s*.h5"))
post.merge_sets("analysisFA/analysisFA.h5", set_paths, cleanup=True)

post.merge_process_files("evolutionFA", cleanup=True)
set_paths = list(pathlib.Path("evolutionFA").glob("evolutionFA_s*.h5"))
post.merge_sets("evolutionFA/evolutionFA.h5", set_paths, cleanup=True)