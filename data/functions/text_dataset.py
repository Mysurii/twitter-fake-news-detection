import os
import pandas as pd
from text_cleaner import basic_cleaning, extra_cleaning, cleaner_2021, clean_text
import warnings
warnings.filterwarnings("ignore")

BASE_PATH = os.path.join(os.path.dirname(__file__), "../cleaned_data")

data_file_2022 = os.path.join(BASE_PATH, "text_columns.csv")
data_cleaned_file = os.path.join(BASE_PATH, "cleaned_text_data_2022.csv")
data_cleaned_file_with_stopwords = os.path.join(BASE_PATH, "cleaned_text_data_with_stopwords_2022.csv")

def make_clean_dataset_2022() -> None:
    """Creates two cleaned csv files in the data/cleaned_data directory"""
    # load data
    df = pd.read_csv(data_file_2022)
    # since user_description can be null, change it to empty string if null
    df['user_description'] = df['user_description'].apply(lambda x: x if pd.notna(x) else '')
    # drop duplicates based on tweet
    df.drop_duplicates(subset=['full_text'])
    # add extra columns about metadata and a combination
    df['text_metadata'] = df['username'] + ' ' + df['user_description']
    df['combined'] = df['text_metadata'] + ' ' + df['full_text']
    # drop unneccesery columns
    df.drop(['username', 'user_description'], axis=1, inplace=True)

    df = df.dropna(subset='full_text')
    df = df.drop_duplicates()

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    df['combined'] = df['combined'].apply(basic_cleaning)
    df['text_metadata'] = df['text_metadata'].apply(basic_cleaning)
    df['full_text'] = df['full_text'].apply(basic_cleaning)

    df = df[df['text_metadata'] != '']

    # Save csv for BERT-models
    df.to_csv(data_cleaned_file_with_stopwords, index=False)

    # since our data is not cleaned yet, we have to clean it
    df['combined'] = df['combined'].apply(extra_cleaning)
    df['text_metadata'] = df['text_metadata'].apply(extra_cleaning)
    df['full_text'] = df['full_text'].apply(extra_cleaning)

    df.to_csv(data_cleaned_file, index=False)


# This script combines the constraint and cmu data,
# then cleans it using the text_cleaner.py scipt
# And finally saves it to the csv file clean_text.csv under the text_cleaned folder

# gets the current directory
path = os.path.dirname(os.path.realpath(__file__))

def make_clean_dataset_2021_cleaned():
     # Making a dataframe
    data = pd.DataFrame(columns=['text', 'label'])

    # Loading in all the parts of the constraint dataset
    constraint_train = pd.read_csv(path + '/../previous_datasets/CONSTRAINT-2021/original/Constraint_English_Train.csv', error_bad_lines=False, encoding='unicode_escape')
    constraint_test = pd.read_csv(path + '/../previous_datasets/CONSTRAINT-2021/original/Constraint_English_Test.csv', error_bad_lines=False, encoding='unicode_escape')
    constraint_val = pd.read_csv(path + '/../previous_datasets/CONSTRAINT-2021/original/Constraint_English_Val.csv',
                                 error_bad_lines=False, encoding='unicode_escape')
    constraint = pd.concat([constraint_train, constraint_test, constraint_val])

    # Changing the string labels to integers
    constraint['label'] = constraint['label'].map(
        lambda x: 2 if x == 'real' else 0)

    # Putting the contstraint data into the dataframe
    data[['text', 'label']] = constraint[[u'tweet', 'label']]

    # loading in the cmu dataset
    cmu = pd.read_csv(
        path + '/../previous_datasets/CMU-MisCov19/modified/MisCov_Complete.csv')

     # The cmu dataset has a lot of possible labels and we want to condense it down to 3 (true, fake and neutral)
    annotations_fake = ['fake cure', 'false fact or prevention',
                        'fake treatment', 'false public health response', 'conspiracy']
    annotations_neutral = ['ambiguous or hard to classify', 'irrelevant', 'politics',
                           'news', 'panic buying', 'commercial activity or promotion', 'emergency']
    annotations_real = ['calling out or correction', 'sarcasm or satire',
                        'true public health response', 'true prevention']
    cmu = cmu[['full_text', 'annotation1']]
    cmu['annotation1'] = cmu['annotation1'].map(
        lambda x: 0 if x in annotations_fake else 2 if x in annotations_real else 1)

    # Adding the cmu data to the dataframe
    data = pd.concat([cmu.rename(
        columns={'full_text': 'text', 'annotation1': 'label'}), data], ignore_index=True)
    data['text'] = data['text'].astype(str)

    # Making a copy of the text so we can look back at the uncleaned version of the text after cleaning it
    data['uncleaned_text'] = data['text']

    # copy the df
    df_with_2022_cleaning = data.copy(deep=True)
  

    # Cleaning the text
    df_with_2022_cleaning['text'] = df_with_2022_cleaning['text'].apply(clean_text)
    data = cleaner_2021(data, 'text')

    # Saving the cleaned dataset
    data.to_csv(os.path.join(BASE_PATH, "cleaned_text_2021.csv"), index=False)
    df_with_2022_cleaning.to_csv(os.path.join(BASE_PATH, "cleaned_text_2021_with_cleaner_2022.csv"), index=False)

    
if __name__ == '__main__':
    make_clean_dataset_2022()
    make_clean_dataset_2021_cleaned()
    
    df = pd.read_csv(data_cleaned_file)
    df_with_stopwords = pd.read_csv(data_cleaned_file_with_stopwords)
    print(df.head())
    print('---')
    print(df_with_stopwords.head())