import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def build_model_input():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    df = pd.concat([train_df, test_df])
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) >= 10:
            df.drop(columns=[col], inplace=True)

    df.fillna(0, inplace=True)
    df = pd.get_dummies(df)

    y = df[df['id'] < 30000]['label']
    X = df[df['id'] < 30000].drop(columns=['label'])
    X_pred = df[df['id'] >= 30000]

    return X, X_pred, y


if __name__ == '__main__':
    X, X_pred, y = build_model_input()
    print(X.shape, X_pred.shape)

    #clf = LogisticRegression()
    clf = svm.SVC()
    clf.fit(X, y)
   
    X_pred_id = X_pred['id']
    X_pred.drop(columns=['id'], inplace=True)

    y_pred = clf.predict(X_pred.values)
    res_df = pd.DataFrame({'id': X_pred_id, 'label': y_pred})
    res_df.to_csv('s_submission.csv', index=False)