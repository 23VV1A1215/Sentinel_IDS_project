import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

column_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty_level"
]


def load_data():
    train = pd.read_csv("data/KDDTrain+.txt", names=column_names)
    test = pd.read_csv("data/KDDTest+.txt", names=column_names)

    train = train.drop("difficulty_level", axis=1)
    test = test.drop("difficulty_level", axis=1)

    return train, test

def preprocess(save_objects=True):
    train, test = load_data()

    # Convert label to binary
    train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X_train = train.drop('label', axis=1)
    y_train = train['label']

    X_test = test.drop('label', axis=1)
    y_test = test['label']

    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    X_train_final = np.hstack((X_train_num, X_train_cat))
    X_test_final = np.hstack((X_test_num, X_test_cat))

    if save_objects:
        joblib.dump(encoder, "models/encoder.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(numerical_cols, "models/numerical_cols.pkl")
        joblib.dump(categorical_cols, "models/categorical_cols.pkl")

    return X_train_final, X_test_final, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Attack distribution in train:")
    print(y_train.value_counts())