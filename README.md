# Py_HMMD
This repository contains example codes for the publication: 

**Efficiency gains of a multi-scale integration method applied to a scale-separated model for rapidly rotating dynamos**, *Computer Physics Communications Journal*, 2021

**Abstract** Numerical geodynamo simulations with parameters close to an Earth-like regime would be of great interest for understanding the dynamics of the Earth's liquid outer core and the associated geomagnetic field. Such simulations are far too computationally demanding owing to the large range in spatiotemporal scales. This paper explores the application of a multi-scale timestepping method to an asymptotic model for the generation of magnetic field in the fluid outer core of the Earth. The method is based on the heterogeneous multiscale modelling (HMM) strategy, which incorporates scale separation and utilizes several integrating models for the fast and slow fields. Quantitative comparisons between the multi-scale simulations and direct solution of the asymptotic model in the limit of rapid rotation and low Ekman number are performed. The multi-scale method accurately captures the varying temporal and spatial dynamics of the mean magnetic field at lower computational costs compared to the direct solution of the asymptotic model.


The following scripts require **installed** DEDALUS framework https://dedalus-project.org/

Simulations are carried out in a 1D domain with size L = 1.0. The vertical spatial dimension is discretized using Chebyshev polynomials and we keep constant number of Chebyshev points in all simulations Nz=128. 

**```qgdm_ref.py```** script computes the reference solutions for the quasi-geostrophic dynamo model (QGDM). The system is solved using Runge-Kutta method. Both slow and fast variables processed at the same time. Simulation may take some time: ~ 1661 sec on Intel Core i5-7200U.

**```qgdm_hmm.py```** implementation of the HMM-like strategy for solving the simplified multiscale QGDM problem. The approximation scheme is constructed on the basis of available time-stepping methods that are accurate up to 3rd order with respect to small time step inside the open source DEDALUS framework and we utilize flexible built-in tools to analyse the results.

**```comparison.py```** provides graphical visualization of computed solutions using HMM-like strategy and the direct computation of the QGDM.

**```cont_bfield.py```** shows Bx magnetic field component profiles in space and time.
