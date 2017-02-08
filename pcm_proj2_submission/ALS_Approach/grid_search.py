#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

from prediction import *

# define parameters
num_epochs = 50 # number of iterations of ALS
cutoff = True # setting for usage of max_iter_threshold stop condition
max_iter_threshold = 0.00005 # stop condition for ALS algorithm, no visible improvement
split_ratio = 0.9 # ratio between size of training and test set

for method in range(4):
	init_method_num = method # number of matrices initialization method
	for features in list([10, 25, 50, 100]):
		num_features = features # number of latent features in matrix factorization
		for lambda1 in list([10, 20, 35, 50, 60]):
			lambda_item = lambda1 # regularization parameter for item features
			for lambda2 in list([10, 20, 35, 50, 60]):
				lambda_user = lambda2 # regularization parameter for user features
				
				print("Method:", method)
				print("Number of features:", features)
				print("Lambda item:", lambda1)
				print("Lambda user:", lambda2)
				
				if init_method_num == 1:
					cutoff = False
				else:
					cutoff = True
				
				create_prediction(init_method_num, num_epochs, cutoff, max_iter_threshold, num_features, lambda_item, lambda_user, split_ratio)
				
				print("#" * 50)
				print("#" * 50)
				print("#" * 50)
