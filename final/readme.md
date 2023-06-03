# Skills, Education, and Family Background: Solving a Dynamic Life Cycle Model of Education Choice

Mathias Koawlczyk Hansen and Annasofie Marckstrøm Olesen

Term Paper in Dynamic Programming, spring 2023

## Dependancies
Apart from the standard Anaconda package, the following installations are required to run the code:

* pip install line_profiler

## Folder contents
Results for the project are produced in the notebooks in this folder. 
* solution.ipynb: Contains code producing model solution and simulation results as well as verifications of the solution. Plots of policy functions, simulated results etc. are produced here.
* Monte_Carlo.ipynb: Performs Monte Carlo experiments and produces plots of the criterion functions.

Code for solving, simulating and estimating is stored in the folder /project_code:
* model.py: Model class, parameters and general solution method.
* DC_EGM.py: Solution of the studying stage using DC-EGM.
* EGM.py: Solution of the working stage using EGM.
* Simulation.py: Simulation of model.
* Estimation.py: Estimation of p_high, p_low, and of p_high,p_low and phi_high using SMM.
* auxiliary_funcs.py: Auxiliary functions. Currently only contains the wage function.
* tools.py: Various numerical tools used in solution and simulation, including linear interpolation and Gauss-Hermite Quadrature. Originally from the consav (https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks) package.

Plots produced is stored in /figs. The folder /pickles contains estimation results exported using the pickle module.
