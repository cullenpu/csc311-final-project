from utils import *
import numpy as np
from matrix_factorization import *

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
    m = 3       # generate 3 resamples
    resamples = generate_resamples(train_data, m)
    predictions = []
    for i in range(m):
        print("m = " + str(i))
        pred, train_loss, val_loss = als(resamples[i], val_data, 50, 0.01, 500000)
        predictions.append(pred)
    avg = sum(predictions) / m
    train_acc = sparse_matrix_evaluate(train_data, avg)
    val_acc = sparse_matrix_evaluate(val_data, avg)
    test_acc = sparse_matrix_evaluate(test_data, avg)
    print("training accuracy: " + str(train_acc))
    print("validation accuracy: " + str(val_acc))
    print("test accuracy: " + str(test_acc))


if __name__ == "__main__":
    main()
