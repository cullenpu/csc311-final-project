import pandas as pd
import numpy as np

from funk_svd import SVD
from utils import *


def svd(data, svd_data, lr=0.01, reg=0.1, k=10, iters=1000):
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



def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_private_test_csv('../data')

    train_svd = {'u_id': train_data['user_id'], 'i_id': train_data['question_id'], 'rating': train_data['is_correct']}
    val_svd = {'u_id': val_data['user_id'], 'i_id': val_data['question_id'], 'rating': val_data['is_correct']}
    test_svd = {'u_id': test_data['user_id'], 'i_id': test_data['question_id'], 'rating': test_data['is_correct']}

    svd = SVD(learning_rate=0.01, regularization=0.1, n_epochs=100, n_factors=5, min_rating=0, max_rating=1)
    svd.fit(X=pd.DataFrame(train_svd), X_val=pd.DataFrame(val_svd), early_stopping=False, shuffle=False)

    pred = svd.predict(test_svd)
    binary_pred = [0 if x < 0.5 else 1 for x in pred]
    print(pred)
    print(binary_pred)
    test_data['is_correct'] = binary_pred

    save_private_test_csv(test_data)


if __name__ == '__main__':
    main()
