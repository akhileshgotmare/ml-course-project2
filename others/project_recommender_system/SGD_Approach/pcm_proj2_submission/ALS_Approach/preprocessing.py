import numpy as np

def initialize_matrices_random(train, num_features):
    """
        Initialize randomly matrices W and Z of matrix factorization.

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """
    
    # W matrix initialization
    item_features = np.random.random((train.shape[0], num_features))
    # Z matrix initialization
    user_features = np.random.random((train.shape[1], num_features))
    
    return item_features, user_features

def initialize_matrices_first_column_mean(train, num_features):
    """
        Initialize randomly matrices W and Z of matrix factorization.
        In matrix W first column is assigned to average rating for that movie.
        In matrix Z first column is assigned to average rating for that user.

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """

    # W matrix initialization
    item_features = np.random.random((train.shape[0], num_features))
    item_features[:, 0] = train.mean(axis=1).reshape(item_features.shape[0])
    # Z matrix initialization
    user_features = np.random.random((train.shape[1], num_features))
    user_features[:, 0] = train.mean(axis=0).reshape(user_features.shape[0])

    return item_features, user_features

def initialize_matrices_SVD(train, num_features):
    """
        Initialize matrices W and Z of matrix factorization using SVD
        decomposition of original matrix X.

        Arguments:
            train: training set (matrix X)
            num_features: number of latent variables in the W*Z^T decomposition

        Returned value(s):
            item_features: matrix W of shape = num_features, num_item
            user_features: matrix Z of shape = num_features, num_user
    """
    
    U, s, V = np.linalg.svd(train, full_matrices=False)
    
    S = np.diag(s)

    U_1 = U[:, 0:num_features]
    S_1 = S[0:num_features, 0:num_features]
    V_1 = V[0:num_features, :]
    
    # W matrix initialization
    item_features = U_1
    # Z matrix initialization
    user_features = (S_1.dot(V_1)).T
    
    return item_features, user_features
