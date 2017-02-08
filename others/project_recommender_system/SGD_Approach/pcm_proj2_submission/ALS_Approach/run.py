#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

from prediction import *

# define parameters sets to generate predictions using two different methods
# (init_method_num, num_epochs, cutoff, max_iter_threshold, num_features, lambda_item, lambda_user, split_ratio)
set1 = (1, 100, True, 0.00005, 50, 60, 10, 1.0) # first_column_mean method
set2 = (2, 1, False, 0.00005, 100, 60, 10, 1.0) # SVD method

for param_set in list([set1, set2]):
	create_prediction(*param_set)

