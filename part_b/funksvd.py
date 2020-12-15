import pandas as pd
import numpy as np

from funk_svd import SVD
from utils import *


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    train_svd = {'u_id': train_data['user_id'], 'i_id': train_data['question_id'], 'rating': train_data['is_correct']}
    val_svd = {'u_id': val_data['user_id'], 'i_id': val_data['question_id'], 'rating': val_data['is_correct']}
    test_svd = {'u_id': test_data['user_id'], 'i_id': test_data['question_id'], 'rating': test_data['is_correct']}

    svd = SVD(learning_rate=0.01, regularization=0.01, n_epochs=1000, n_factors=30, min_rating=0, max_rating=1)
    svd.fit(X=pd.DataFrame(train_svd), X_val=pd.DataFrame(val_svd), early_stopping=True, shuffle=False)

    # Train Accuracy
    pred = svd.predict(train_svd)
    acc = evaluate(train_data, pred)
    print("Train Accuracy: ", acc)

    # Validate Accuracy
    pred = svd.predict(val_svd)
    acc = evaluate(val_data, pred)
    print("Validation Accuracy: ", acc)

    # Test accuracy
    pred = svd.predict(test_svd)
    acc = evaluate(test_data, pred)
    print("Test Accuracy: ", acc)


if __name__ == '__main__':
    main()
