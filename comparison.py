import h5py
import numpy as np
import matplotlib.pyplot as plt



z_slice = 32    # z_basis = de.Chebyshev('z', 128, interval=(0, 1.0)) So, z=0.5 means z_slice = 64


# evolution:
with h5py.File("ref_evolution/ref_evolution.h5", mode='r') as file:
    bx_ref = np.array(file['tasks']['Bx'])
    te_ref = np.array(file['scales']['sim_time'])


with h5py.File("evolutionSL/evolutionSL.h5", mode='r') as file:
    bx_hmm = np.array(file['tasks']['Bx'])
    te_hmm = np.array(file['scales']['sim_time'])



# analysis:
with h5py.File("ref_analysis/ref_analysis.h5", mode='r') as file:
    em_ref = np.array(file['tasks']['EM'])
    nu_ref = np.array(file['tasks']['NU'])
    ta_ref = np.array(file['scales']['sim_time'])

with h5py.File("analysisSL/analysisSL.h5", mode='r') as file:
    em_hmm = np.array(file['tasks']['EM'])
    tas_hmm = np.array(file['scales']['sim_time'])

with h5py.File("analysisFA/analysisFA.h5", mode='r') as file:
    nu_hmm = np.array(file['tasks']['NU'])
    taf_hmm = np.array(file['scales']['sim_time'])    



plt.figure(figsize=(10, 6))    
plt.plot(te_ref, bx_ref[:,z_slice], 'b', label="Ref")
plt.plot(te_hmm, bx_hmm[:,z_slice], 'r--', label="HMM")
plt.legend(loc='lower right', fontsize=14)
plt.xlabel('Time')
plt.title('Vertical slices of the Bx component at z = 0.25')


plt.figure(figsize=(10, 6))    
plt.plot(ta_ref, em_ref, 'b', label="Ref")
plt.plot(tas_hmm, em_hmm, 'r--', label="HMM")
plt.legend(loc='lower right', fontsize=14)
plt.xlabel('Time')
plt.title('The magnetic energy')
    

plt.figure(figsize=(10, 6))    
plt.plot(ta_ref, nu_ref, 'b', label="Ref")
plt.plot(taf_hmm, nu_hmm, 'r--', label="HMM")
plt.legend(loc='lower right', fontsize=14)
plt.xlabel('Time')
plt.title('The Nusselt number')

plt.show()