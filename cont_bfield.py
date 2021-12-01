import h5py
import numpy as np
import matplotlib.pyplot as plt


FS = 16
plt.figure(figsize=(10, 6))
with h5py.File("ref_evolution/ref_evolution.h5", mode='r') as file:
    bx = np.array(file['tasks']['Bx'])
    t = np.array(file['scales']['sim_time'])
    z = np.array(file['scales']['z']['1.0'])
    c = plt.contourf(t, z, np.transpose(bx), levels=np.linspace(np.min(bx),np.max(bx),30), cmap='seismic')
    cbr = plt.colorbar(c,shrink=0.9)
    cbr.ax.tick_params(labelsize=FS) 


plt.xlabel('Time',fontsize=FS)
plt.ylabel('z',fontsize=FS)
plt.tick_params(axis="x", labelsize=FS)
plt.tick_params(axis="y", labelsize=FS)
plt.title("Contour plots of the mean magnetic field Bx component",fontsize=FS)
plt.tight_layout()
plt.show()
