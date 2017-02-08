import numpy as np
import datetime
from helpers import *

def compute_ALS(train_set, train_nonzero_indices, test_set, test_nonzero_indices, num_epochs, cutoff, max_iter_threshold, num_features, lambda_item, lambda_user, item_features, user_features, test_mode):
	'''
		parameters
		train_set: test matrix
		train_nonzero_indices: indices of non_zero entries of test matrix
		test_set: hidden test set
		test_nonzero_indices: indices of non_zero entries of hidden test set
		num_epochs: number of iterations (stop-condition)
		cutoff: if True use max_iter_treshold as a stop_condition
		max_iter_threshold: value of the treshold of differences between two consecutive iterations for RMSE
		num_features: number of latent features
		lambda_item: value of lambda regularization parameter for items
		lambda_user: value of lambda regularization parameter for users
		item_features: preinitialized matrix W
		user_features: preinitialized matrix Z
		test_mode: if True cross-validation is performed
		
		returns
		prediction_matrix: result of algorithm
		train_rmse: RMSE of the training set
		test_rmse: RMSE of the test set
		it: number of performed iterations
	'''
    # initialize matrices used to compute RMSE
    train_label = np.zeros(len(train_nonzero_indices))
    test_label = np.zeros(len(test_nonzero_indices))
    train_prediction_label = np.zeros(len(train_nonzero_indices))
    test_prediction_label = np.zeros(len(test_nonzero_indices))
    
    # initialize accumulators for RMSE of every iteration
    train_rmse = np.zeros(num_epochs)
    test_rmse = np.zeros(num_epochs)
    if test_mode == True:
        for i in range(len(test_nonzero_indices)):
            test_label[i] = test_set[test_nonzero_indices[i][0], test_nonzero_indices[i][1]]
    
    # initialize lambda matrices
    lambda_user_diag = np.identity(num_features)
    np.fill_diagonal(lambda_user_diag, lambda_user)
    lambda_item_diag = np.identity(num_features)
    np.fill_diagonal(lambda_item_diag, lambda_user)
    
    # initialize accumulator for RMSE error calculation for stop condition
    last_train_rmse = 0
    
    for it in range(num_epochs):
        begin = datetime.datetime.now() # start time measurement
        
        print("Epoch:", it)

        # perform one iteteration of the algorithm
        
        # first fix item features: Z^T = (W^T*W + (lambda_z*I_K)^(-1)*W^T*X)
        user_features = (np.linalg.inv(item_features.T.dot(item_features) + lambda_user_diag).dot(item_features.T.dot(train_set))).T
        # then fix user features: W^T = (Z^T*Z + (lambda_w*I_K)^(-1)*Z^T*X^T)
        item_features = (np.linalg.inv(user_features.T.dot(user_features) + lambda_item_diag).dot(user_features.T.dot(train_set.T))).T

        # calculate training RMSE
        for i in range(len(train_nonzero_indices)):
            train_label[i] = train_set[train_nonzero_indices[i][0], train_nonzero_indices[i][1]]
            train_prediction_label[i] = item_features[train_nonzero_indices[i][0], :].dot(user_features.T[:, train_nonzero_indices[i][1]])
        
        # store train RMSE of current iteration
        train_rmse[it] = calculate_mse(train_label, train_prediction_label)
        
        print("RMSE on training set:", train_rmse[it])
        
        if test_mode == True:
            # calculate test RMSE
            for i in range(len(test_nonzero_indices)):
                test_prediction_label[i] = item_features[test_nonzero_indices[i][0], :].dot(user_features.T[:, test_nonzero_indices[i][1]])

            # store test RMSE of current iteration
            test_rmse[it] = calculate_mse(test_label, test_prediction_label)

            print("RMSE on test set:", test_rmse[it])

        end = datetime.datetime.now() # stop time measurement
        
        # compute the time of the iteration
        execution_time = (end - begin).total_seconds()
        print("Execution time:", execution_time)

        print("*" * 50)
        
        if np.fabs(last_train_rmse - train_rmse[it]) < max_iter_threshold:
            print("NO SIGNIFICANT CHANGES")
            if cutoff == True:
                break
        else:
            last_train_rmse = train_rmse[it]
    
    return item_features.dot(user_features.T), train_rmse, test_rmse, it
