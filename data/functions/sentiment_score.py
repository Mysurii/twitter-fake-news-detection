from transformers import  AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentScore():
  def __init__(self):
    self.model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

  def __return_input_tokenized(self, input_string):
    """
    Returns tokenized format of a given input string.

    :param input_string: input string for sentiment score.
    :return: tokenized format of a given input string.
    """
    return self.tokenizer.encode(input_string, return_tensors='pt')

  def __return_sentiment_score(self, tokens):
    """
    Returns integer for sentiment score higher is beter.

    :param tokens: tokenized format of an input string.
    :return: sentiment score integer
    """
    return int(torch.argmax(tokens.logits))

  def get_sentiment_score(self, text: str) -> int:
    tokenized = self.model(self.__return_input_tokenized(text))

    return self.__return_sentiment_score(tokenized)