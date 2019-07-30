import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
import warnings
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Loads a database into a pd.DataFrame and returns
    Args:
        database_filepath(str): file path to a .db database file

    Returns: [X(pd.Series), Y(pd.DataFrame), [(str)]]
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('response', engine)
    X = df['message']
    y = df[df.columns[3:]]
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    """
    Normalizes and tokenizes messages
    Args:
        text:

    Returns:

    """
    # Normalizing Text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenizing Text
    words = word_tokenize(text)

    # Remove Stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(x).strip() for x in words]

    return clean_tokens


def build_model():
    """
    Builds a pipeline and returns it

    Returns: sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(
            LinearSVC()))
    ])

    # Set parameters for gird search
    parameters = {
        'clf__estimator__C': range(1, 11),
        'clf__estimator__verbose': [3],
        'clf__n_jobs': [4]
    }

    # Set grid search
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Uses sklearn's classification report to evaluate the model

    Args:
        model(sklearn.multioutput.MultiOutputClassifier): a pipeline or ml model
        X_test(pd.Series): test data
        y_test(pd.Series): target values test data
        category_names(list): column names for target values

    Returns: None
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test.values, y_pred,
                                target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the model to a pickle file

    Args:
        model:
        model_filepath:

    Returns:

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function controlling script for training the classifier saving the
    model to a pickle file

    Returns: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()