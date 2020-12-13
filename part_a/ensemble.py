from utils import *
import numpy as np


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
    resamples = generate_resamples(train_data, 3)
    print("hi")


if __name__ == "__main__":
    main()