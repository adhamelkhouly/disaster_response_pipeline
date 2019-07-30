import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

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

    # To be plotted later on, this extracts the most commonly used words
    messages = df['message'].values.tolist()
    text = ' '.join(messages)
    text = tokenize(text)
    dist = FreqDist(w for w in text)
    most_common = dict(dist.most_common(10))

    return X, y, category_names, most_common


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
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # set parameters for grid search
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    # Set grid search
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2,
                      verbose=3, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Uses sklearn's classification report to evaluate the model

    Args:
        model(GridSearchCV): a pipeline or ml model
        X_test(pd.DataFrame): test data
        y_test(pd.DataFrame): target values test data
        category_names(list): column names for target values

    Returns: None
    """
    y_pred = model.predict(X_test)
    print('Best parameters for model are: ', model.best_params_)
    for index, cat in enumerate(category_names):
        print(cat)
        print(classification_report(y_test.iloc[:, index], y_pred[:, index]))


def save_model(model, model_filepath, most_common_words):
    """
    Saves the model to a pickle file

    Args:
        model:
        model_filepath:
        most_common_words:

    Returns:

    """
    with open(model_filepath, 'wb') as f:
        pickle.dump((model, most_common_words), f)


def main():
    """
    Main function controlling script for training the classifier saving the
    model to a pickle file

    Returns: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names, most_common = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print(y.sum())
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath, most_common)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db model.pkl')


if __name__ == '__main__':
    main()
