from part_b.item_response_2PL import irt, irt_predict
from utils import *
import numpy as np
from part_b.funksvd import svd, svd_predict

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
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    m = 10
    train_resamples = generate_resamples(train_data, m//2)
    val_resamples = generate_resamples(val_data, m//2)

    svd_train_acc = []
    svd_val_acc = []

    train_predictions = []
    val_predictions = []
    test_predictions = []

    # for i in range(m//2):
    #     print("m = " + str(i))
    #     curr_train, curr_val = train_resamples[i], val_resamples[i]
    #     train_svd = {'u_id': curr_train['user_id'],
    #                  'i_id': curr_train['question_id'],
    #                  'rating': curr_train['is_correct']}
    #     val_svd = {'u_id': curr_val['user_id'], 'i_id': curr_val['question_id'],
    #                'rating': curr_val['is_correct']}
    #     data = {"train_data": curr_train, "val_data": curr_val}
    #     svd_data = {"train_svd": train_svd, "val_svd": val_svd}
    #
    #     predictions = svd_predict(data, svd_data)
    #     train_predictions.append(predictions)
        # svd_val_acc.append(val_acc)

    irt_train_acc = []
    irt_val_acc = []
    for j in range(m//2, m):
        theta, a, beta, train_acc, val_acc, train_log_likes, val_log_likes, final = \
            irt(train_data, val_data, 0.005, 10)
        pred_vals, predictions = irt_predict(train_data, theta, a, beta)
        # irt_train_acc.append(train_acc[-1])
        # irt_val_acc.append(val_acc[-1])
        predictions.append(pred_vals)

    avg = sum(train_predictions) / m
    train_acc = sparse_matrix_evaluate(train_data, avg)
    val_acc = sparse_matrix_evaluate(val_data, avg)
    test_acc = sparse_matrix_evaluate(test_data, avg)
    # train_avg = (sum(svd_train_acc) + sum(irt_train_acc)) / m
    # val_avg = (sum(svd_val_acc) + sum(irt_val_acc)) / m
    print("Train Accuracy: ", train_acc)
    print("Validation Accuracy", val_acc)
    print("Test Accuracy", test_acc)


if __name__ == "__main__":
    main()
