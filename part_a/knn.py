from sklearn.impute import KNNImputer
from utils import *

import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    K = (1, 6, 11, 16, 21, 26)
    # User-based
    print('User-based collaborative filtering')
    accuracy = []
    for k in K:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracy.append(acc)

    plt.plot(K, accuracy, 'o-')
    plt.title('User-based Filtering kNN Accuracy')
    plt.xlabel('k')
    plt.ylabel('Validation Data Accuracy')
    plt.show()

    best_k = K[accuracy.index(max(accuracy))]
    print('The value of k with the highest performance on the validation data is ', best_k)
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print('The test accuracy on k =', best_k, 'is', test_acc)

    # Item-based
    print('Item-based collaborative filtering')
    accuracy = []
    for k in K:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracy.append(acc)

    plt.plot(K, accuracy, 'o-')
    plt.title('Item-based Filtering kNN Accuracy')
    plt.xlabel('k')
    plt.ylabel('Validation Data Accuracy')
    plt.show()

    best_k = K[accuracy.index(max(accuracy))]
    print('The value of k with the highest performance on the validation data is ', best_k)
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print('The test accuracy on k =', best_k, 'is', test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
