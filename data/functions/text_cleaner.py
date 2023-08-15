import re
import string
import emoji

import preprocessor as pp
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

pp.set_options(pp.OPT.URL, pp.OPT.MENTION)

def basic_cleaning(text: str) -> str:
    """Deze functie doet de standaard tekst cleaning. 
    Het is bedoeld voor alle tekst modellen"""
    """This function is meant for all tekst-models. It does the cleaning. 
    Lemmatizing and stop_words removal are excluded (BERT-cleaning)"""

    # replace urls and twitter mentions with $MENTION$ en &URL&
    text = pp.tokenize(text)

    # remove punctuation
    text = text.translate(str.maketrans('','', string.punctuation))

    # replace emojis with the name of the emoji. E.g. 👍 = :thumbs-up:
    text = emoji.demojize(text)
    text = re.sub(":", '', text)

    text = text.lower()

    # # remove numbers
    text = re.sub(r"\b[\d.]+\b", "", text)

    # remove white space and return the cleaned text
    return re.sub(r"\s+", " ", text)

def extra_cleaning(text: str) -> str:
    """Deze functie haalt stop woorden, zoals 'are' en 'doing' weg. Bovendien worden de woorden lemmatized. 
    Deze functie is niet bedoelt voor BERT modellen."""
    """This function removes stop words, like 'are' and 'doing'. Furthermore, it lemmatizes the words. 
    note: it is not meant for BERT-models classification"""
    
    new_text = []

    tokenized = word_tokenize(text)

    for word in tokenized:
        word = word.strip()
        word = lemmatizer.lemmatize(word) # rocks -> rock, better -> good, running -> run
        new_text.append(word)

    text = ' '.join(lemmatizer.lemmatize(word) for word in tokenized if word not in stop_words)
    
    text = text.replace("’", '')
    text = text.replace("'", '')
    text = text.replace('"', '')


    # verwijder extra spaties
    text = re.sub(r"\s+", " ", text)

    return text

def clean_text(text:str) -> str:
    """Volledige tekst cleaner (BERT-modellen excluded)"""
    text = basic_cleaning(text)
    text = extra_cleaning(text)
    return text

def cleaner_2021(df, text_column, bert_cleaning=False):
    """Cleanup versie 2021"""

    stop_words = ["i","a","about","an","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","the","this","to","was","what","when","where","who","will","with","the","www"]

    # Changes all url's and mentions to a token that's the same for all urls and a token that's the same for all mentions
    # This: 
    # Preprocessor is #awesome 👍 https://github.com/s/preprocessor @test
    # Becomes this:
    # 'Preprocessor is #awesome 👍 $URL$ $MENTION$'
    # Go here for the documentation on pp: https://pypi.org/project/tweet-preprocessor/ 
    pp.set_options(pp.OPT.URL, pp.OPT.MENTION)
    df[text_column] = [pp.tokenize(text) for text in df[text_column]]

    # Removing punctuation
    # This: Won't !#$% *&^ hallo?, does this work?
    # becomes this: Wont hallo does this work
    df[text_column] = [re.sub("[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", '', text) for text in df[text_column]] 

    # Change emoji's into tokens too but give every emoji it's own token
    # 'Python is 👍' Becomes: 'Python is :thumbs_up:'
    # The documentation for the emoji module is here: https://pypi.org/project/emoji/ 
    df[text_column] = [emoji.demojize(text) for text in df[text_column]]
    df[text_column] = [re.sub(":", '', text) for text in df[text_column]] 

    # Changing every upper case letter to lower case
    df[text_column] = df[text_column].str.lower()

    # Removing unneeded white spaces from the text
    df[text_column] = [re.sub('\s+', ' ', text) for text in df[text_column]]

    # If we're cleaning the data for a bert model we might want to leave in stop words and unlemmatized versions of words
    # because context is important for bert.
    if not bert_cleaning:
        # Lemmatizing the text

        # For lemmatizing we need to know what type of word a word is to lemmatize it.
        # The pos_tag function that is used to figure out what the word types are uses 
        # strings to say what the word types are, but the algorithm that does the 
        # lemmatization needs a different variable type so this function translates 
        # between the two. (pos stands for part of speech, which is the same thing as word type)
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        
        # The object that is going to lemmatize the words
        lemmatizer = WordNetLemmatizer()

        def lemmatize_sentence(text):
            # Tagging the words with their type of word
            tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
            # lemmatizing the words
            lemmatized_sentence = [
                lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) for word in tagged_words
            ]
            return ' '.join(lemmatized_sentence)

        df[text_column] = [lemmatize_sentence(text) for text in df[text_column]]

        # removing stop words
        def remove_stop_words(text):
            tokenized_sentence = nltk.word_tokenize(text)
            tokenized_sentence = ["" if token in stop_words else token for token in tokenized_sentence]
            return ' '.join(tokenized_sentence)
        df[text_column] = [remove_stop_words(text) for text in df[text_column]]

    # dropping emty rows and duplicate rows (only looking at the text column)
    df = df.dropna(subset=[text_column]).drop_duplicates(subset=[text_column])
    
    # Remove "tweets" that are longer than 280 characters
    df = df[df[text_column].str.len() < 280]

    return df

if __name__ == '__main__':
    example_text = '🤴Princekhan🤴 a hungry stomach, an empty pocket and a broken heart can teach the best lessons of life.🌾🌾🌾@globaltimesnews It doesn’t effect randians coz they have CowUrine for cure🐄💦💁🏿‍♂️ after all they have bad smell to tackle COVID-19 with Cow-dung'
    print('before:')
    print(example_text)
    example_text = basic_cleaning(example_text)
    print('\nbasic cleaning:')
    print(example_text)
    print('\nextra_cleaning:')
    example_text = extra_cleaning(example_text)
    print(example_text)