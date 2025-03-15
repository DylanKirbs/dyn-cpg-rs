# Based on https://medium.com/we-talk-data/explaining-sparse-datasets-with-practical-examples-dead60c2c3b7

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

# Tombstones upon deletion (mutating the graph)


def print_matrix(matrix):

    # if it walks like a duck and quacks like a duck...
    if "toarray" in dir(matrix):
        matrix = matrix.toarray()

    with np.printoptions(precision=3):
        print(matrix)


# Define a simple 5x5 matrix
data = np.array([3, 4, 5])
rows = np.array([0, 1, 3])
cols = np.array([0, 2, 4])  # Create a COO sparse matrix
coo = coo_matrix((data, (rows, cols)), shape=(5, 5))
print("COO format:\n", coo)  # Convert to CSR format
csr = coo.tocsr()
print("\nCSR format:\n", csr)  # Convert to CSC format
csc = coo.tocsc()
print("\nCSC format:\n", csc)


# Create a dense matrix (just for illustration)
dense_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 2, 0]])  # Convert to a sparse matrix
sparse_matrix = csr_matrix(dense_matrix)
print("Sparse Matrix (CSR):\n", sparse_matrix)

# Create two sparse matrices
sparse_matrix_1 = csr_matrix([[0, 0, 3], [1, 0, 0], [0, 2, 0]])
sparse_matrix_2 = csr_matrix([[0, 1, 0], [2, 0, 0], [0, 0, 4]])
# Perform matrix multiplication
result = sparse_matrix_1.dot(sparse_matrix_2)
print("Result of multiplication:")
print_matrix(result)

# Perform element-wise addition
sum_matrix = sparse_matrix_1 + sparse_matrix_2
print("Result of addition:")
print_matrix(sum_matrix)
