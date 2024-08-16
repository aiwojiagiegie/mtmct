import numpy as np


# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# Function to perform KNN
def knn(data, target, k):
    # Calculate distances between target and all points in data
    distances = [euclidean_distance(d, target) for d in data]
    # Combine distances with data indices
    distances = np.array(list(zip(distances, range(len(data)))))
    # Sort by distance
    sorted_distances = distances[distances[:, 0].argsort()]
    # Get the top k closest indices
    closest_k_indices = sorted_distances[:k, 1].astype(int)
    # Return the top k closest vectors
    return data[closest_k_indices]
# Test case
def test_knn():
    # Define a dataset
    data = np.array([
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0]
    ])

    # Define a target vector
    target = np.array([2, 2])

    # Choose k
    k = 3

    # Expected result: the three closest vectors to the target
    expected_result = np.array([
        [1, 2],
        [4, 2],
        [1, 4]
    ])

    # Perform KNN
    result = knn(data, target, k)

    # Check if the result matches the expected result
    assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"
    print("Test passed!")
if __name__ == '__main__':
    matrix = np.loadtxt('test.txt', delimiter=',')
    data=matrix[1]
    result = knn(matrix[1], matrix, 10)
    print(result)