#TODO list

###Everybody
1. [x] Choose one of the three possible topics. Related Kaggle competitions:  
	* https://inclass.kaggle.com/c/epfml-rec-sys
	* https://inclass.kaggle.com/c/epfml-text
	* https://inclass.kaggle.com/c/epfml-segmentation
--> We have decided to take recommender system.  
2. [ ] Register the team (we're still waiting for instructions)  
VERY ROUGH DESCRIPTION  
3. [x] data_set is given in specific form-make it be matrix NxN where on i,j position is rating for i-th movie and j-th user  
4. [x] decide which libraries to use, scikit and panda are first choices, but think about it  
5. [x] make basic model which would predict missing rating using matrix factorisation  
6. make cross-validation to check basic accuracy of prediction  
7. [x] upload first submission-make .csv file  
deadline: in a few days-this first points should not be so difficult  
8. [ ] add data preprocessing-it is not clear to me what should be x here for recommendation systems, but this is surely big point  
9. [ ] see other ways to make better model except matrix factorisation  
10. [ ] determine if model overfitts or underfits and set parameters and do regularisation according to that  
11. [ ] do some kind of visualisation of model, this could be x as any step, just to see how model works  
12. [ ] Move implementation from notebook to external library  
13. [ ] Comment the code (both documentation and markdown)  
14. [ ] Check code style using PEP8  
15. [ ] Collect all the needed files (see: instruction)  
16. [ ] Write script to pack it  
17. [ ] Write a report  
18. [ ] SUBMIT FINAL WORKING SOLUTION  

--------------------------------------------------------------------------------

###Optimisations
Things to work on during optimisation:  
1. Try different initialization of the matrices W and Z  
  * RANDOM: Assign small random numbers to all the matrix entries in both matrices
  * FIRST-COLUMN-MEAN: Initialize the item matrix W by assigning the average rating for that movie as the first column, and small random numbers for the remaining entries
  * GLOBAL-MEAN: Initialize all elements of W and Z equal to square root of global_mean/num_features
  * SVD: Use SVD algorithm for first approximation of the product of W and Z^T  
2. Try trick with removing bias and adding it back in the end  
  * Remove bias of item and users (try this order and inverted)
  * Remove average of biases  
3. Find the optimal value for parameters for SGD  
  * step_size
  * number_of_iterations
  * number_of_latent_features
  * lambda_item
  * lambda_user
  * lambda_bias_item
  * lambda_bias_user  
4. Find the optimal value for parameters for ALS  
  * number_of_iterations
  * number_of_latent_features
  * lambda_item
  * lambda_user  
5. Trim values lower then 1 and higher then 5  
6. Try to see if discretising float values by assigning the closest matching improves the score  
7. Average output of both SGD and ALS classifiers.

--------------------------------------------------------------------------------

####Paweł
* [x] Implement ALS
* [ ] Find best parameters for the model and determine best combination of possible optimisations
* [ ] Merge submissions generated for ALS and SGD (in separate notebook)

--------------------------------------------------------------------------------

####Ana
* [x] I did 3rd dot and i coded 5.dot but it is not tested if it works
* [ ] Report

--------------------------------------------------------------------------------

####Akhilesh
* [x] Implement SGD
* [ ] Try some cross-validation
* [ ] Find the best parameters for the model and try different optimisations
