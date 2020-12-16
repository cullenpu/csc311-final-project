import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from funk_svd import SVD
from utils import *


np.random.seed(311)


def svd(data, svd_data, lr=0.01, reg=0.1, k=10, iters=500):
    train_data, val_data = data['train_data'], data['val_data']
    train_svd, val_svd = svd_data['train_svd'], svd_data['val_svd']

    svd = SVD(learning_rate=lr, regularization=reg, n_epochs=iters, n_factors=k, min_rating=0, max_rating=1)
    svd.fit(X=pd.DataFrame(train_svd), X_val=pd.DataFrame(val_svd), early_stopping=False, shuffle=False)

    # Train Accuracy
    pred = svd.predict(train_svd)
    train_acc = evaluate(train_data, pred)

    # Validate Accuracy
    pred = svd.predict(val_svd)
    val_acc = evaluate(val_data, pred)

    return train_acc, val_acc


def plot(x_vals, train_y, valid_y, label):
    plt.plot(x_vals, train_y, '--ok', color='red', label="training")
    plt.plot(x_vals, valid_y, '--ok', color='blue', label="validation")
    plt.xlabel(label)
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. " + label)
    plt.legend()
    plt.show()


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    train_svd = {'u_id': train_data['user_id'], 'i_id': train_data['question_id'], 'rating': train_data['is_correct']}
    val_svd = {'u_id': val_data['user_id'], 'i_id': val_data['question_id'], 'rating': val_data['is_correct']}
    test_svd = {'u_id': test_data['user_id'], 'i_id': test_data['question_id'], 'rating': test_data['is_correct']}

    data = {"train_data": train_data, "val_data": val_data}
    svd_data = {"train_svd": train_svd, "val_svd": val_svd}

    lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    regs = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    ks = [1, 5, 10, 20, 50, 100]
    iters = [10, 50, 100, 500, 1000, 2000]

    lr_train_results = []
    lr_val_results = []
    reg_train_results = []
    reg_val_results = []
    ks_train_results = []
    ks_val_results = []
    iters_train_results = []
    iters_val_results = []

    for lr in lrs:
        train_result, val_result = svd(data, svd_data, lr=lr)
        lr_train_results.append(train_result)
        lr_val_results.append(val_result)

    for reg in regs:
        train_result, val_result = svd(data, svd_data, reg=reg)
        reg_train_results.append(train_result)
        reg_val_results.append(val_result)

    for k in ks:
        train_result, val_result = svd(data, svd_data, k=k)
        ks_train_results.append(train_result)
        ks_val_results.append(val_result)

    for iter in iters:
        train_result, val_result = svd(data, svd_data, iters=iter)
        iters_train_results.append(train_result)
        iters_val_results.append(val_result)

    best_lr = lrs[lr_val_results.index(max(lr_val_results))]
    print("Best learning rate: ", best_lr)
    best_reg = regs[reg_val_results.index(max(reg_val_results))]
    print("Best regularization value: ", best_reg)
    best_k = ks[ks_val_results.index(max(ks_val_results))]
    print("Best k: ", best_k)
    best_iter = iters[iters_val_results.index(max(iters_val_results))]
    print("Best iterations: ", best_iter)

    plot(lrs, lr_train_results, lr_val_results, "Learning Rates")
    plot(regs, reg_train_results, reg_val_results, "Regularized Rates")
    plot(ks, ks_train_results, ks_val_results, "K-Values")
    plot(iters, iters_train_results, iters_val_results, "Iterations")

    final_svd = SVD(learning_rate=best_lr, regularization=best_reg, n_epochs=best_iter, n_factors=best_k, min_rating=0, max_rating=1)
    final_svd.fit(X=pd.DataFrame(train_svd), X_val=pd.DataFrame(val_svd), early_stopping=False, shuffle=False)

    # Train Accuracy
    pred = final_svd.predict(train_svd)
    train_acc = evaluate(train_data, pred)
    print("Final Train Accuracy: ", train_acc)

    # Validate Accuracy
    pred = final_svd.predict(val_svd)
    val_acc = evaluate(val_data, pred)
    print("Final Validation Accuracy: ", val_acc)

    # Test Accuracy
    pred = final_svd.predict(test_svd)
    test_acc = evaluate(test_data, pred)
    print("Final Test Accuracy: ", test_acc)


if __name__ == '__main__':
    main()
