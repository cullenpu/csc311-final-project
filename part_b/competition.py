from part_b.item_response_2PL import irt, irt_predict
from utils import *
import numpy as np
import pandas as pd
from funk_svd import SVD

np.random.seed(311)


def generate_resamples(data, m):
    n = len(data["user_id"])
    resamples = []
    for i in range(m):
        indices = np.random.choice(n, n, replace=True)
        resample = {"user_id": (np.array(data["user_id"])[indices]).tolist(),
                    "question_id": (np.array(data["question_id"])[indices]).tolist(),
                    "is_correct": (np.array(data["is_correct"])[indices]).tolist()}
        resamples.append(resample)
    return resamples


def main():
    # Hyperparameters
    m = 1  # Number of bootstrap resamples

    irt_iters = 1
    irt_lr = 0.05

    svd_lr = 0.01
    svd_reg = 0.10
    svd_iters = 500
    svd_k = 5

    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_private_test_csv('../data')

    val_svd = {'u_id': val_data['user_id'], 'i_id': val_data['question_id'], 'rating': val_data['is_correct']}
    test_svd = {'u_id': test_data['user_id'], 'i_id': test_data['question_id'], 'rating': test_data['is_correct']}

    svd_train_resamples = generate_resamples(train_data, m)
    irt_train_resamples = generate_resamples(train_data, m)

    svd_test_pred, irt_test_pred = [], []
    for i in range(m):
        curr_irt, curr_svd = irt_train_resamples[i], svd_train_resamples[i]

        # Train 2-PL IRT
        theta, a, beta, train_acc, val_acc, train_log_likes, val_log_likes, final = \
            irt(curr_irt, val_data, irt_lr, irt_iters)

        irt_test_pred.append(irt_predict(test_data, theta, a, beta)[0])

        # Train Funk SVD
        curr_svd = {'u_id': curr_svd['user_id'], 'i_id': curr_svd['question_id'],
                     'rating': curr_svd['is_correct']}

        svd = SVD(learning_rate=svd_lr, regularization=svd_reg, n_epochs=svd_iters, n_factors=svd_k, min_rating=0, max_rating=1)
        svd.fit(X=pd.DataFrame(curr_svd), X_val=pd.DataFrame(val_svd), early_stopping=False, shuffle=False)

        svd_test_pred.append(svd.predict(test_svd))

    test_avg = np.sum(irt_test_pred + svd_test_pred, axis=0) / (2 * m)

    binary_pred = [0 if x < 0.5 else 1 for x in test_avg]
    test_data['is_correct'] = binary_pred
    save_private_test_csv(test_data)


if __name__ == "__main__":
    main()
