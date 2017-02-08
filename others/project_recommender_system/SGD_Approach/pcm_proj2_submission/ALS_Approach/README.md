##Structure of the directory

Start reading the code with notebook. It will give the understanding of the program control flow and help to understand the structure of the library.
In order to reproduce the results of ALS method fast use run.py file.
In case of the need to take a deeper look of the implementation use main.py file.
To see the cross-validation checkout grid_search.py file.

###Files:
	1. IPYTHON NOTEBOOK  
		* Alternating_Least_Squares_model.ipynb - notebook containing the code that can produce prediction for single set of parameters  
	2. COMPUTATIONS  
		* run.py - program in this file produces output files for two different methods (later averaged for the purpose of submission) (estimated time 15 minutes)  
		* grid_search.py - program in this file performs a grid search of parameters (estimated time 19.05 hours)  
	3. LIBRARIES  
		* preprocessing.py - function definitions associated with preprocessing  
		* prediction.py - main function to create predicion for a single method  
		* ALS.py - implementation of the Alternating Least Squares algorithm  
		* helpers.py - definitions of some tool functions  
		* plots.py - definitions of functions used in notebook to plot some graphs

###External libraries
	We use numpy and scipy. For plotting we use matplotlib.