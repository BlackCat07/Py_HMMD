import numpy as np

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from dedalus.tools import post

import subprocess
import logging
import pathlib
import time
logger = logging.getLogger(__name__)


# Parameters:
Ekm = 1e-4      # the Ekman number
Ra  = 80        # the reduced Rayleigh number
Pm  = 0.7       # the reduced magnetic Prandtl number
Pr  = 1.0       # the Prandtl number
ik  = 1.3048    # wavenumber

T_final = 40.0  # final time


# Bases and domain:
z_basis = de.Chebyshev('z', 128, interval=(0, 1.0))
domain = de.Domain([z_basis], grid_dtype=np.float64)

# Problem:
problem = de.IVP(domain, variables=['Psi', 'W', 'The', 'tet', 'Bx', 'By', 'Bx_z', 'By_z', 'tet_z'])
problem.parameters['eps']   = Ekm**(1.0/3.0)
problem.parameters['Pr']    = Pr
problem.parameters['k']     = ik
problem.parameters['Pm']    = Pm
problem.parameters['Ra']    = Ra

# Main equations and boundary conditions:
problem.add_equation("dt(Psi) + dz(W)/k**2 + k**2*Psi = - Pm*(Bx**2 + By**2)*Psi/2.0")
problem.add_equation("dt(W) + dz(Psi) - Ra*The/Pr + k**2*W = - Pm*(Bx**2 + By**2)*W/2.0 ")

problem.add_equation("dt(The)  + k**2*The/Pr = - W*dz(tet)")
problem.add_equation("dz(tet_z)/Pr = dz(W*The)")

problem.add_equation("eps**(-3.0/2.0)*dt(Bx) - dz(Bx_z)/Pm = - Pm*dz(Psi*W*By)")
problem.add_equation("eps**(-3.0/2.0)*dt(By) - dz(By_z)/Pm =   Pm*dz(Psi*W*Bx)")

problem.add_equation("tet_z - dz(tet) = 0")
problem.add_equation("Bx_z - dz(Bx) = 0")
problem.add_equation("By_z - dz(By) = 0")

problem.add_bc("left(W) = 0.0")
problem.add_bc("right(W) = 0.0")
problem.add_bc("left(tet) = 1.0")
problem.add_bc("right(tet) = 0.0")
problem.add_bc("left(Bx) = 0.0")
problem.add_bc("right(Bx) = 0.0")
problem.add_bc("left(By) = 0.0")
problem.add_bc("right(By) = 0.0")


# Build solver:
solver = problem.build_solver(de.timesteppers.RK443)
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
solver.stop_sim_time = T_final


# Initial conditions:
z = domain.grid(0)
Restart = False

if Restart:
    write, dt = solver.load_state('/analysis/analysis.h5', -1)
    solver.sim_time = 0.0

else:
    Psi = solver.state['Psi']
    W = solver.state['W']
    The = solver.state['The']
    tet = solver.state['tet']
    Bx = solver.state['Bx']
    By = solver.state['By']

    Psi['g']    =-np.pi/ik**4*np.cos(np.pi*z)
    The['g']    = 1.0/ik**2*np.sin(np.pi*z)
    tet['g']    = 1-z
    W['g']      = np.sin(np.pi*z)
    Bx['g']     = np.sin(np.pi*z)
    By['g']     = np.sin(np.pi*z)


# Analysis:
analysis = solver.evaluator.add_file_handler('ref_analysis', sim_dt = 0.05, max_writes=500)
analysis.add_task("integ(Bx**2, 'z')**0.5", layout='g', name='BX2')
analysis.add_task("1 + integ(W*The, 'z')", layout='g', name='NU')
analysis.add_task("0.5*integ(Bx**2 + By**2, 'z')", layout='g', name='EM')

evolution = solver.evaluator.add_file_handler('ref_evolution', sim_dt = 0.05, max_writes=500)
evolution.add_system(solver.state, layout='g')



# Main loop:
dt = 2e-4
start_time = time.time()
while solver.ok:
    solver.step(dt)
    if solver.iteration % 500 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()
print('Runtime:', end_time-start_time)


# Write runtime:
f = open("runtime_reference","w")
f.write('%.12e' % (end_time-start_time))
f.close()


# Post analysis:
post.merge_process_files("ref_analysis", cleanup=True)
set_paths = list(pathlib.Path("ref_analysis").glob("ref_analysis_s*.h5"))
post.merge_sets("ref_analysis/ref_analysis.h5", set_paths, cleanup=True)
print(subprocess.check_output("find ref_analysis", shell=True).decode())

post.merge_process_files("ref_evolution", cleanup=True)
set_paths = list(pathlib.Path("ref_evolution").glob("ref_evolution_s*.h5"))
post.merge_sets("ref_evolution/ref_evolution.h5", set_paths, cleanup=True)
print(subprocess.check_output("find ref_evolution", shell=True).decode())
