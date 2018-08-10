
from sklearn.svm import SVC


def test(clf, X, y):
    from sklearn.metrics import classification_report
    y_true, y_pred = y, clf.predict(X)
    print(classification_report(y_true, y_pred))


def evaluate_params(X, y, X_test, y_test, params):
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    clf = GridSearchCV(SVC(), params)
    clf.fit(X, y)

   
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        
        y_true, y_pred = y_test, clf.predict(X_test)
        


import pickle

def save_model(clf, model_name="clf.pkl"):
    # save the classifier
    with open(model_name, 'wb') as f:
        pickle.dump(clf, f)

def load_model(model_name="clf.pkl"):
    with open(model_name, 'rb') as f:
        clf = pickle.load(f)
    return clf

