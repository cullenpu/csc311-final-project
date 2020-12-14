from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    n, m = u.shape[0], z.shape[0]

    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    correct = train_data["is_correct"][i]
    user = train_data["user_id"][i]
    question = train_data["question_id"][i]

    inner = correct - np.transpose(u[user]) @ z[question]
    u_update = lr * inner * z[question]
    u[user] = u[user] + u_update

    z_update = lr * inner * u[user]
    z[question] = z[question] + z_update

    return u, z


def als(train_data, valid_data, k, lr, num_iteration, calculate_losses=True):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix, list of squared error every 1000 iterations
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    train_losses = []
    val_losses = []
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)

        if calculate_losses:
            if i % 10000 == 0:
                print("num iter: " + str(i))
                sel = squared_error_loss(train_data, u, z)
                train_losses.append(sel)
                print("train loss: " + str(sel))
                sel = squared_error_loss(valid_data, u, z)
                val_losses.append(sel)
                print("valid loss: " + str(sel))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    mat = u @ np.transpose(z)
    return mat, train_losses, val_losses


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # k_vals = [1, 5, 10, 20, 50, 100]
    # results = []
    # for k in k_vals:
    #     result_matrix = svd_reconstruct(train_matrix, k)
    #     results.append(sparse_matrix_evaluate(val_data, result_matrix))
    # print(results)
    #
    # result_matrix = svd_reconstruct(train_matrix, 5)
    # test_result = (sparse_matrix_evaluate(test_data, result_matrix))
    # print(test_result)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # num_iterations = [10000, 50000, 100000, 200000, 500000, 1000000, 1250000, 1500000]
    # for num_iters in num_iterations:
    #     print("num iterations: " + str(num_iters))
    #     result = als(train_data, 5, 0.01, num_iters)
    #     test_result = (sparse_matrix_evaluate(val_data, result))
    #     print(test_result)

    # learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    # for lr in learning_rates:
    #     print("learning rate: " + str(lr))
    #     result = als(train_data, 5, lr, 500000)
    #     test_result = (sparse_matrix_evaluate(val_data, result))
    #     print(test_result)

    # choose k=20
    # k_vals = [1, 5, 10, 20, 50, 100]
    # for k in k_vals:
    #     print("k: " + str(k))
    #     result = als(train_data, k, 0.01, 500000)
    #     test_result = (sparse_matrix_evaluate(val_data, result))
    #     print(test_result)

    # final runs
    predictions, train_loss, val_loss = als(train_data, val_data, 20, 0.01, 500000)
    iters_range = np.arange(0, 500000, 10000)
    plt.plot(iters_range, train_loss, color='red', label="training")
    plt.plot(iters_range, val_loss, color='blue', label="validation")
    plt.xlabel("num iterations")
    plt.ylabel("squared error loss")
    plt.title("Squared Error Loss vs. Iterations")
    plt.legend()
    plt.show()

    train_acc = sparse_matrix_evaluate(train_data, predictions)
    val_acc = sparse_matrix_evaluate(val_data, predictions)
    test_acc = sparse_matrix_evaluate(test_data, predictions)

    print("training accuracy: " + str(train_acc))
    print("validation accuracy: " + str(val_acc))
    print("test accuracy: " + str(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
