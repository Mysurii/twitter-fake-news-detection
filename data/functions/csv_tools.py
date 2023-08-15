import os
from pandas import DataFrame

def to_csv(df: DataFrame, path: str, file_name: str):
    """Save an pandas dataframe to a csv file"""

    if path[len(path) - 1] != '/':
      path = path + '/'

    try:
      # check if the directories exists, if not, create the dirs
      if not os.path.exists(path):
        os.makedirs(path)
      
      file_path = path + file_name

      # check if the path (dirs) AND the file exists already
      if os.path.exists(file_path):
        os.remove(file_path)

      df.to_csv(file_path, columns=df.columns)
      print(f'Succesfully saved data to {file_path}')
    except Exception as e:
        print(f"Failed to save dataframe to a csv file: {e}")
