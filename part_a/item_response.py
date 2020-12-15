from utils import *
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector representing the ability of student i
    :param beta: Vector representing the difficulty of problem j
    :return: float
    """
    log_like = 0
    for i, c in enumerate(data['is_correct']):
        curr_theta = theta[data['user_id'][i]]
        curr_beta = beta[data['question_id'][i]]

        diff = curr_theta - curr_beta
        log_like += (c * diff) - np.log(1 + np.exp(diff))
    return -log_like


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    theta_deriv = np.zeros(theta.shape[0])
    beta_deriv = np.zeros(beta.shape[0])

    for i, correct in enumerate(data['is_correct']):
        curr_theta = theta[data['user_id'][i]]
        curr_beta = beta[data['question_id'][i]]

        theta_deriv[data['user_id'][i]] += (correct - sigmoid(curr_theta - curr_beta))
        beta_deriv[data['question_id'][i]] += (sigmoid(curr_theta - curr_beta) - correct)

    theta += (lr * theta_deriv)
    beta += (lr * beta_deriv)
    return theta, beta


def irt(data, val_data, lr, iterations):
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
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    train_acc_lst = []
    train_neg_lld_lst = []
    val_neg_lld_lst = []

    val_score = 0

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_neg_lld_lst.append(neg_lld)
        neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_neg_lld_lst.append(neg_lld)

        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        print("NLLK: {} \t Score: {}".format(neg_lld, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst, val_score


def evaluate(data, theta, beta):
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
        x = (theta[u] - beta[q]).sum()
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


def plot_probabilities(beta, data):
    thetas = np.arange(-5, 6)
    questions = [20, 40, 60, 80, 100]
    colors = ['green', 'blue', 'red', 'cyan', 'magenta']
    color_index = 0
    for q in questions:
        question = data["question_id"][q]
        curr = []
        for theta in thetas:
            prob = sigmoid(theta - beta[question])
            curr.append(prob)
        plt.plot(thetas, curr, color=colors[color_index], label="Question #" + str(q))
        color_index += 1
    plt.xlabel("theta")
    plt.ylabel("probability of correct response")
    plt.title("Probability of Correct Response vs Theta")
    plt.legend()
    plt.show()



def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # TEST DIFFERENT NUMBER OF ITERATIONS
    # num_iterations = [5, 10, 20, 50, 100]
    # for num in num_iterations:
    #     print("Number of iterations " + str(num))
    #     irt(train_data, val_data, 0.01, num)

    # TEST DIFFERENT LEARNING RATES
    # learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    # for lr in learning_rates:
    #     print("Learning rate:" + str(lr))
    #     irt(train_data, val_data, lr, 30)

    num_iterations = 30
    learning_rate = 0.005
    theta, beta, train_acc, val_acc, train_log_likes, val_log_likes, final = \
        irt(train_data, val_data, learning_rate, num_iterations)
    print("Validation Accuracy: ", final)
    plot_probabilities(beta, val_data)

    plot(num_iterations, train_acc, val_acc, "accuracy", "Accuracies vs Num Iterations")
    plot(num_iterations, train_log_likes, val_log_likes, "log likelihood", "Log Likelihoods vs Num Iterations")

    theta, beta, train_acc, val_acc, train_log_likes, val_log_likes, final = \
        irt(train_data, test_data, learning_rate, num_iterations)
    print("Test Accuracy: ", final)


if __name__ == "__main__":
    main()
