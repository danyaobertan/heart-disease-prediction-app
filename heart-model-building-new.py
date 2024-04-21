import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier


def load_data(filename):
    return pd.read_csv(filename)


def preprocess_features(df):
    # Define the features to be encoded
    encode = ['Sex', 'ChestPainType', 'FastingBS',
              'RestingECG', 'ExerciseAngina', 'STSlope']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1).drop(columns=[col])
    return df


def encode_target(df, target_column):
    target_mapper = {0: 0, 1: 1}
    df[target_column] = df[target_column].apply(lambda x: target_mapper[x])
    return df


def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    Y = df[target_column]
    return X, Y


def train_model(X, Y):
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    return clf


def save_model(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)


def main():
    # Load and preprocess the data
    df = load_data('patients-cleansed-test-edited.csv')
    df = preprocess_features(df)
    df = encode_target(df, 'HeartDisease')
    X, Y = split_data(df, 'HeartDisease')

    # Train the model
    model = train_model(X, Y)

    # Save the model
    save_model(model, 'HeartDisease_clf_gpt.pkl')

    print("Model training and saving completed successfully.")


if __name__ == "__main__":
    main()
