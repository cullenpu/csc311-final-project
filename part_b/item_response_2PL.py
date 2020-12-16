from utils import *
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, a, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector representing the ability of student i
    :param a: Vector representing the discrimination of problem j
    :param beta: Vector representing the difficulty of problem j
    :return: float
    """
    log_like = 0
    for i, c in enumerate(data['is_correct']):
        curr_theta = theta[data['user_id'][i]]
        curr_a = a[data['question_id'][i]]
        curr_beta = beta[data['question_id'][i]]

        diff = curr_theta - curr_beta
        log_like += (c * curr_a * diff) - np.log(1 + np.exp(curr_a * diff))
    return -log_like


def update_theta_beta(data, lr, theta, a, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        a <- new_a
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param a: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    theta_deriv = np.zeros(theta.shape[0])
    a_deriv = np.zeros(a.shape[0])
    beta_deriv = np.zeros(beta.shape[0])

    for i, correct in enumerate(data['is_correct']):
        curr_theta = theta[data['user_id'][i]]
        curr_a = a[data['question_id'][i]]
        curr_beta = beta[data['question_id'][i]]

        diff = curr_theta - curr_beta
        theta_deriv[data['user_id'][i]] += (correct * curr_a - curr_a * sigmoid(curr_a * diff))
        a_deriv[data['question_id'][i]] += correct * diff - diff * sigmoid(curr_a * diff)
        beta_deriv[data['question_id'][i]] += (curr_a * sigmoid(curr_a * diff) - correct * curr_a)
    theta += (lr * theta_deriv)
    a += (lr * a_deriv)
    beta += (lr * beta_deriv)
    return theta, a, beta


def irt(data, val_data, lr=0.01, iterations=10):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.random.rand(542)
    a = np.random.rand(1774)
    beta = np.random.rand(1774)

    val_acc_lst = []
    train_acc_lst = []
    train_neg_lld_lst = []
    val_neg_lld_lst = []

    val_score = 0

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, a=a, beta=beta)
        train_neg_lld_lst.append(neg_lld)
        neg_lld = neg_log_likelihood(val_data, theta=theta, a=a, beta=beta)
        val_neg_lld_lst.append(neg_lld)

        train_score = evaluate(data=data, theta=theta, a=a, beta=beta)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, a=a, beta=beta)
        val_acc_lst.append(val_score)

        print("NLLK: {} \t Score: {}".format(neg_lld, val_score))
        theta, a, beta = update_theta_beta(data, lr, theta, a, beta)

    return theta, a, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst, val_score


def irt_predict(data, theta, a, beta):
    pred = []
    pred_vals = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (a[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
        pred_vals.append(p_a)
    return pred_vals, pred


def evaluate(data, theta, a, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (a[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def plot(num_iterations, training, validation, ylabel, title):
    iters_range = np.arange(0, num_iterations)
    plt.plot(iters_range, training, color='red', label="training")
    plt.plot(iters_range, validation, color='blue', label="validation")
    plt.xlabel("num iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # TEST DIFFERENT NUMBER OF ITERATIONS
    num_iterations = [5, 10, 20, 30, 50]
    iters_val_results = []
    for num in num_iterations:
        theta, a, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst, val_score = irt(train_data, val_data, iterations=num)
        iters_val_results.append(val_score)

    # TEST DIFFERENT LEARNING RATES
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    lr_val_results = []
    for lr in learning_rates:
        theta, a, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst, val_score = irt(train_data, val_data, lr=lr)
        lr_val_results.append(val_score)

    best_iter = num_iterations[iters_val_results.index(max(iters_val_results))]
    print("Best iterations: ", best_iter)
    best_lr = learning_rates[lr_val_results.index(max(lr_val_results))]
    print("Best learning rate: ", best_lr)

    theta, a, beta, train_acc, val_acc, train_log_likes, val_log_likes, final = \
        irt(train_data, val_data, best_lr, best_iter)
    print("Validation Accuracy: ", final)

    plot(best_iter, train_acc, val_acc, "accuracy", "Accuracies vs Num Iterations")
    plot(best_iter, train_log_likes, val_log_likes, "log likelihood", "Log Likelihoods vs Num Iterations")

    # theta, a, beta, train_acc, val_acc, train_log_likes, val_log_likes, final = \
    #     irt(train_data, test_data, best_lr, best_iter)
    # print("Test Accuracy: ", final)


if __name__ == "__main__":
    main()
