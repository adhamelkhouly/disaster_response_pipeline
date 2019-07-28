import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Args:
        messages_filepath (str): filepath to messages csv file
        categories_filepath(str): filepath to categories csv file

    Returns: pd.DataFrame
    """
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)['categories']
    df_merge = df_messages.merge(df_categories, how='outer', on=['id'])
    return df_merge


def clean_data(df):
    """
    Args:
        df(pd.DataFrame): raw DataFrame to clean and process

    Returns: pd.DataFrame
    """
    # Split the category column into multiple columns to a new DataFrame
    df_categories = df['categories'].str.split(';', expand=True)

    # Extracting first row to grab column names, and renaming DataFrame columns
    first_row = df_categories.iloc[0]
    column_names = [x[:-2] for x in first_row]
    df_categories.columns = column_names

    # Clean all columns to just contain a number 1 or 0
    df_categories = df_categories.applymap(lambda x: int(x[-1:]))
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, df_categories], axis=1)

    # Dropping the original common, as the original message is not very useful
    # in classification, and many of those entries is null
    df.drop(columns=['original'], inplace=True)

    return df


def save_data(df, database_filename):
    """
    Args:
        df(pd.DataFrame): pandas DataFrame to save to a database file
        database_filename(str): path to database filename

    Returns: None
    """
    pass  


def main():
    """
    Main function controlling script for processing input data and saving it
    to a database file.

    Returns: None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(
            messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
