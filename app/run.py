import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist


from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objects as go
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


def plot1(df):
    """
    Graph 1 of the categorical count

    Args:
        df (pd.DataFrame): loaded DataFrame from database file

    Returns: dict
    """
    categories_count = df[df.columns[4:]].sum(axis=0)
    graph = {
        'data': [
            go.Bar(
                x=df.columns[3:],
                y=categories_count
            )
        ],

        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }
    return graph


def plot2(df):
    """
    Graph 2 of the top words used

    Args:
        df (pd.DataFrame): loaded DataFrame from database file

    Returns: dict
    """
    messages = df['message'].values.tolist()
    text = ' '.join(messages)
    text = tokenize(text)
    dist = FreqDist(w for w in text)
    most_common = dict(dist.most_common(10))

    graph = {
        'data': [
            go.Bar(
                x=most_common.keys(),
                y=most_common.values()
            )
        ],

        'layout': {
            'title': 'Distribution of Top Words Used',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Word"
            }
        }
    }
    return graph


def plot3(df):
    """
    Graph 3 of the message genres

    Args:
        df (pd.DataFrame): loaded DataFrame from database file

    Returns: dict
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph = {
            'data': [
                go.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    return graph


def plot4(df):
    """
    Graph 4 of categories per genre

    Args:
        df (pd.DataFrame): loaded DataFrame from database file

    Returns: dict
    """
    pass


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('response', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                go.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()