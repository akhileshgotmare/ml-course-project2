import csv
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import time

def read_txt(path):
    """Read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def preprocess_data(data):
    """Preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    
    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating    
    return ratings

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle
    competition."""
    return preprocess_data(read_txt(path_dataset)[1:])

def non_zero_indices(matrix):
	"""Returns list of indices of nonzero entries of the matrix."""
    nz_row, nz_col = matrix.nonzero()
    return list(zip(nz_row, nz_col))

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return np.sqrt(1.0 * t.dot(t.T) / len(t))

def create_csv_submission(prediction, submission_file_path = "submission.csv"):
    """
        Creates an output file in csv format for submission to kaggle.

        Arguments:
            prediction: matrix W * Z^T
            submission_file_path: string name of .csv output file to be created
    """

	# file path to the dataset of the entries to be predicted
    dataset_file_path = "sampleSubmission.csv"
    sample_ratings = load_data(dataset_file_path)
    
    # find the non-zero ratings indices 
    nz_row_sr, nz_col_sr = sample_ratings.nonzero()
    nz_sr = list(zip(nz_row_sr, nz_col_sr))
    
    # helper function to trim marginal values, predictions should be in interval
    # of [1, 5]
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
