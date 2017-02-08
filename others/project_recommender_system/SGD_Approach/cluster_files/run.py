
# coding: utf-8

# Latent_Factor_with_biases_with_preproc_and_code_optimization

# In[1]:

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import datetime
import time
from helpers import *


# In[2]:

path_dataset = "data_train.csv"
ratings_data = load_data(path_dataset)
print('Shape of ratings matrix:',ratings_data.shape)


# In[128]:

np.random.seed(77)


# In[17]:

# Chose submatrix from the 10000x1000 matrix
ratings = scipy.sparse.lil_matrix.todense(ratings_data)
#sub_rows = np.random.randint(0, ratings.shape[0], 2000)
#sub_cols = np.random.randint(0, ratings.shape[1], 200)
#ratings = ratings[sub_rows,:]
#ratings = ratings[:,sub_cols]

# In[18]:

print('Shape of ratings',ratings.shape)
print('No of non-zero entries',len(ratings.nonzero()[0]))


# In[19]:

def remove_empty_rows_and_cols(ratings):

    del_rows = ([])
    for row in range(ratings.shape[0]):
        if len(ratings[row,:].nonzero()[0]) == 0:
            del_rows.append(row)

    ratings = np.delete(ratings,del_rows,axis=0)
    
    del_cols = ([])
    for col in range(ratings.shape[1]):
        if len(ratings[:,col].nonzero()[0]) == 0:
            del_cols.append(col)
            

    ratings = np.delete(ratings,del_cols,axis=1)
    return ratings


# In[20]:

def preprocess(ratings):
    
    def find_mean_vectors(ratings):
        
        # calculating the user_mean and item_mean vectors:

        # user_mean vector:

        user_mean = np.zeros((ratings.shape[1],1))

        for user_no in range(ratings.shape[1]):

            a = ratings[:,user_no].sum()
            b = np.shape(ratings[:,user_no].nonzero())[1]
            user_mean[user_no,0] = (a/b)

        # item_mean vector:

        item_mean = np.zeros((ratings.shape[0],1))

        for item_no in range(ratings.shape[0]):

            a = ratings[item_no,:].sum()
            b = np.shape(ratings[item_no,:].nonzero())[1]
            item_mean[item_no,0] = (a/b)
        print('user_mean and item_mean computed!')   
        return user_mean, item_mean

    user_mean, item_mean = find_mean_vectors(ratings)
    
    mask = ratings.copy()
    mask[mask>0] = 1
#     mask = scipy.sparse.lil_matrix.todense(mask)
    
    A = ((user_mean@np.ones((1,ratings.shape[0]))).T)
    B = ((item_mean@np.ones((1,ratings.shape[1]))))
    preproc_layer = ( np.multiply(mask,A)
                     + np.multiply(mask,B) ) / 2
    
#     ratings_dense = scipy.sparse.lil_matrix.todense(ratings)
    ratings_dense = ratings
    ratings_preproc = (ratings_dense - (preproc_layer))
    retrieve_layer = (A + B)/2
    return ratings_preproc, preproc_layer, retrieve_layer


# In[21]:

def find_global_mean(ratings):
    global_mean = np.sum(ratings)/len(ratings.nonzero()[0])
    print(global_mean,'global_mu')
    return global_mean


# In[22]:

def find_bias_vectors(ratings):
    
    # calculating the user_bias and item_bias vectors:

    global_mean = np.sum(ratings)/len(ratings.nonzero()[0])
    print(global_mean,'global_mu')

    # user_bias vector:
    user_bias = np.zeros((ratings.shape[1],1))

    for user_no in range(ratings.shape[1]):

        a = ratings[:,user_no].sum()
        b = np.shape(ratings[:,user_no].nonzero())[1]
        user_bias[user_no,0] = global_mean - (a/b)

    # item_bias vector:
    item_bias = np.zeros((ratings.shape[0],1))

    for item_no in range(ratings.shape[0]):

        a = ratings[item_no,:].sum()
        b = np.shape(ratings[item_no,:].nonzero())[1]
        item_bias[item_no,0] = global_mean - (a/b)
    
    print('user_bias and item_bias computed!')   
    return global_mean, user_bias, item_bias


# In[23]:

ratings = remove_empty_rows_and_cols(ratings)
ratings_preproc, preproc_layer, retrieve_layer = preprocess(ratings)
global_mean, user_bias_stored, item_bias_stored = find_bias_vectors(ratings_preproc)


# In[24]:

def SGD_train(gamma,num_features,lambda_,num_epochs,ratings,change_step):
    
    
    global_mean = find_global_mean(ratings)
    # define parameters
    gamma =  gamma   #0.00008 for no lambdas and w/0 preproc, 0.003 for the cluster code
    num_features = num_features   # K in the lecture notes
    split_ratio = 0.8   # ratio between size of training and test set
    lambda_ = lambda_
    [lambda_user, lambda_item, lambda_user_bias, lambda_item_bias] = [lambda_, lambda_, lambda_, lambda_]
    num_epochs = num_epochs     # number of full passes through the train set


    test_mode = True
    if split_ratio == 1.0:
        test_mode = False

    def nonzero_indices(matrix):
        nz_row, nz_col = matrix.nonzero()
        return list(zip(nz_row, nz_col))

    nonzero_indices = nonzero_indices(ratings)
    split_point = int(np.floor(len(nonzero_indices) * split_ratio))
    train_nonzero_indices = nonzero_indices[:split_point]
    test_nonzero_indices = nonzero_indices[split_point:]

    train_set = np.zeros(ratings.shape)
    test_set = np.zeros(ratings.shape)

    for i, j in train_nonzero_indices:
        train_set[i, j] = ratings[i, j]

    for i, j in test_nonzero_indices:
        test_set[i, j] = ratings[i, j]

    # find the non-zero ratings indices 
    nz_row, nz_col = train_set.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test_set.nonzero()
    nz_test = list(zip(nz_row,nz_col))


    # init matrix
    item_features = 0.0001*np.random.random((train_set.shape[0],num_features))
    user_features = 0.0001*np.random.random((train_set.shape[1],num_features))

    user_bias = 0.0001* np.random.random((train_set.shape[1]))
    item_bias = 0.0001* np.random.random((train_set.shape[0]))

    real_train_label = np.zeros(len(nz_train))
    prediction_train = np.zeros(len(nz_train))
    rmse_train = np.zeros(num_epochs)


    # Printing training rmse before any update loop

    mat_pred = ( global_mean*np.ones((train_set.shape)) +
                (user_bias.reshape((train_set.shape[1],1)).dot(np.ones((1,train_set.shape[0])))).T +
                (item_bias.reshape((train_set.shape[0],1)).dot(np.ones((1,train_set.shape[1])))) +
                np.dot(item_features,user_features.T) )

    mat_pred_for_mse = (mat_pred + retrieve_layer)

    for i in range(len(nz_train)):
        real_train_label[i] = train_set[nz_train[i][0],nz_train[i][1]] + retrieve_layer[nz_train[i][0],nz_train[i][1]]
        prediction_train[i] = mat_pred_for_mse[nz_train[i][0],nz_train[i][1]]

    rmse = calculate_mse(real_train_label, prediction_train)
    print('Train rmse with initialization: ',rmse)   
    print('gamma = ',gamma)

    if test_mode == True:

        real_test_label  = np.zeros(len(nz_test))
        prediction_test = np.zeros(len(nz_test))
        rmse_test  = np.zeros(num_epochs)

        # Printing test rmse before any update loop

        for i in range(len(nz_test)):
            real_test_label[i] = test_set[nz_test[i][0],nz_test[i][1]] + retrieve_layer[nz_test[i][0],nz_test[i][1]]
            prediction_test[i] = mat_pred_for_mse[nz_test[i][0],nz_test[i][1]]

        rmse = calculate_mse(real_test_label, prediction_test)
        print('Test rmse with initialization: ',rmse)  


    for it in range(num_epochs): 
        if change_step == True:
            if it>11:
                gamma = 0.00002

        print('Iteration No',it+1)
        
        # decrease step size
        # gamma /= 1.2

        begin = datetime.datetime.now()
        count = 0
        for d,n in nz_train:
#             count += 1
#             if count%10000 == 0:
#                 print(count)
            difference = train_set[d,n] - mat_pred[d,n]

            # Updating the W
            gradient1 = -1* (difference) * user_features[n,:]
            item_features[d,:] = item_features[d,:]*(1 - gamma*lambda_item) - gamma * gradient1


            # Updating the Z
            gradient2 = -1* (difference) * item_features[d,:]
            user_features[n,:] = user_features[n,:]*(1 - gamma*lambda_user) - gamma * gradient2

            # Updating the user_bias vector
            gradient3 = -1* (difference) 
            user_bias[n] = user_bias[n]*(1 - gamma*lambda_user_bias) - gamma * gradient3

            # Updating the item_bias vector
            gradient4 = -1* (difference)
            item_bias[d] = item_bias[d]*(1 - gamma*lambda_item_bias) - gamma * gradient4


            mat_pred[d,:] = (np.dot(user_features,item_features[d,:])
                             + user_bias
                             + item_bias[d]*np.ones((ratings.shape[1])) 
                             + global_mean*np.ones((ratings.shape[1])))

            mat_pred[:,n] = (np.dot(item_features,user_features[n,:])
                             + item_bias
                             + user_bias[n]*np.ones((ratings.shape[0]))
                             + global_mean*np.ones((ratings.shape[0])))

        mat_pred_for_mse = (mat_pred + retrieve_layer)

        #Calculating training rmse
        for i in range(len(nz_train)):
            real_train_label[i] = train_set[nz_train[i][0],nz_train[i][1]] + retrieve_layer[nz_train[i][0],nz_train[i][1]]
            prediction_train[i] = mat_pred_for_mse[nz_train[i][0],nz_train[i][1]]

        rmse = calculate_mse(real_train_label, prediction_train) 
        rmse_train[it] = rmse

        if test_mode == True:
            
            for i in range(len(nz_test)):
                real_test_label[i] = test_set[nz_test[i][0],nz_test[i][1]] + retrieve_layer[nz_test[i][0],nz_test[i][1]]
                prediction_test[i] = mat_pred_for_mse[nz_test[i][0],nz_test[i][1]]

            rmse_t = calculate_mse(real_test_label, prediction_test)
            rmse_test[it] = rmse_t

        print("iter: {}, RMSE on training set: {}.".format(it+1, rmse))
        if test_mode == True:
            print("iter: {}, RMSE on testing set: {}.".format(it+1, rmse_t))
        end = datetime.datetime.now()
        execution_time = (end - begin).total_seconds()

        np.save('mat_pred_for_mse.npy',mat_pred_for_mse)
        np.save('user_bias.npy', user_bias)
        np.save('item_bias.npy', item_bias)
        np.save('user_features.npy', user_features)
        np.save('item_features.npy', item_features)
        np.save('rmse_train.npy',rmse_train)
        np.save('rmse_test.npy',rmse_train)


        print('Iteration runtime: ',execution_time)
            
    return mat_pred, retrieve_layer, user_bias, item_bias, user_features, item_features, rmse_test, rmse_train


# In[26]:

gamma =   0.006   #0.00008 for no lambdas and w/0 preproc, 0.003 for the cluster code
num_features = 50  #20   # K in the lecture notes
lambda_ = 0 #5 #0.7
num_epochs = 25

print('Trying gamma (stepsize) = ',gamma)
[mat_pred, retrieve_layer, user_bias, item_bias, user_features, item_features, rmse_test, rmse_train] = (
    SGD_train(gamma,num_features,lambda_,num_epochs,ratings_preproc,change_step=False))
# plt.plot(np.arange(num_epochs),rmse_train,label='Train rmse')
# plt.plot(np.arange(num_epochs),rmse_test,label='Test rmse')
# plt.legend()
# plt.show()

# In[ ]:

def create_csv_submission(prediction, submission_file_path = "submission.csv"):
    """
        Creates an output file in csv format for submission to kaggle.

        Arguments:
            prediction: matrix W * Z^T
            submission_file_path: string name of .csv output file to be created
    """

    dataset_file_path = "sampleSubmission.csv" # file path to the dataset of the entries to be predicted
    sample_ratings = load_data(dataset_file_path)
    
    # find the non-zero ratings indices 
    nz_row_sr, nz_col_sr = sample_ratings.nonzero()
    nz_sr = list(zip(nz_row_sr, nz_col_sr))
    
    def trim_values(x):
        if x < 1:
            return 1
        if x > 5:
            return 5
        return x
    
    submission_file_path = time.strftime("%Y%m%d_%H%M%S") + " " + submission_file_path
    with open(submission_file_path, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for i, j in nz_sr:
            writer.writerow({'Id' : 'r' + str(i + 1) + '_' + 'c' + str(j + 1),
                             'Prediction' : str(trim_values(prediction[i, j]))})


# In[ ]:

# mat_pred = ( global_mean*np.ones((10000,1000)) +
#            (user_bias.reshape((1000,1)).dot(np.ones((1,10000)))).T +
#            (item_bias.reshape((10000,1)).dot(np.ones((1,1000)))) +
#            np.dot(item_features,user_features.T) )

mat_pred_for_mse = (mat_pred + retrieve_layer)
prediction = mat_pred_for_mse
create_csv_submission(prediction)

np.save('mat_pred_for_mse.npy',mat_pred_for_mse)
np.save('user_bias.npy', user_bias)
np.save('item_bias.npy', item_bias)
np.save('user_features.npy', user_features)
np.save('item_features.npy', item_features)
np.save('rmse_train.npy',rmse_train)
np.save('rmse_test.npy',rmse_train)
