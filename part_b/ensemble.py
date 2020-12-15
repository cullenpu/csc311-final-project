from utils import *
import numpy as np
from part_b.funksvd import svd

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

    for i in range(m//2):
        print("m = " + str(i))
        curr_train, curr_val = train_resamples[i], val_resamples[i]
        train_svd = {'u_id': curr_train['user_id'],
                     'i_id': curr_train['question_id'],
                     'rating': curr_train['is_correct']}
        val_svd = {'u_id': curr_val['user_id'], 'i_id': curr_val['question_id'],
                   'rating': curr_val['is_correct']}
        data = {"train_data": curr_train, "val_data": curr_val}
        svd_data = {"train_svd": train_svd, "val_svd": val_svd}

        train_acc, val_acc = svd(data, svd_data)
        svd_train_acc.append(train_acc)
        svd_val_acc.append(val_acc)

    # for j in range(m//2, m):
    #     print("m = " + str(i))

    avg = sum(svd_val_acc) / (m/2)
    print(avg)


if __name__ == "__main__":
    main()
