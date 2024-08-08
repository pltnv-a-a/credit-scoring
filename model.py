from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    y = df['SeriousDlqin2yrs']
    X = df.drop(['SeriousDlqin2yrs', 'GroupAge', 'RealEstateLoansOrLines'],
                axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                         random_state=24)

    return X_train, X_test, y_train, y_test


def preprocessing(df: pd.DataFrame):
    df.dropna(inplace=True)
    
    X_train, X_test, y_train, y_test = split_data(df)

    ss = MinMaxScaler()
    ss.fit(X_train)

    print('Классы несбалансированы: ', y_test.value_counts()) 

    X_train = pd.DataFrame(ss.transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(ss.transform(X_test), columns = X_test.columns)

    return X_train, X_test, y_train, y_test


def fit_model_and_save(X_train, X_test, y_train, y_test, path1=
                       'data/model_weights.mw', path2='data/importances_report.csv'):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:,1]
    classes = pred_proba > 0.7

    accuracy = accuracy_score(classes, y_test)
    print(f'Model accuracy is {accuracy}')

    recall = recall_score(classes, y_test)
    print(f'Model recall is {recall}')

    importances_of_features = pd.DataFrame({'weights':model.coef_[0], 'features':
                                            X_train.columns}).sort_values(by='weights')
    print(f'Importances of features are {importances_of_features}')

    with open(path1, 'wb') as file:
        dump(model, file)

    importances_of_features.to_csv(path2)

    print(f'Model was savef to {path1}, report was saved to {path2}')



def load_model_and_make_scoring(df, path='data/model_weights.mw'):
    with open(path, 'rb') as file:
        model = load(file)

    prediction = model.predict(df)[0]

    prediction_proba = model.predict_proba(df)[0]

    encode_pred = {
        1 : 'Сожалеем, но кредит не одобрен',
        0 : 'Поздравляем, мы готовы выдать Вам кредит'
    }

    encode_pred_proba = {
        1 : 'Ваша заявка будет отклонена с вероятностью',
        0 : 'Ваша заявка будет одобрена с вероятностью'
    }

    prediction_data = {}
    for key, value in encode_pred_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_pred[prediction]

    return prediction, prediction_df


if __name__ == '__main__':
    df = pd.read_csv('data/credit_scoring.csv')
    X_train, X_test, y_train, y_test = preprocessing(df)
    fit_model_and_save(X_train, X_test, y_train, y_test)





