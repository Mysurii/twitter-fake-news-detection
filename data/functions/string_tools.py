import ast

def convert_string_to_object(string):
  try:
    return ast.literal_eval(string)
  except Exception as e:
    print(f"Something went wrong: {e}")

def print_split_shapes(xtrain, ytrain, xtest, ytest) -> None:
  print(f"Train shapes\n\tX:{xtrain.shape}\n\ty:{ytrain.shape}")
  print(f"Test shapes\n\tX:{xtest.shape}\n\ty:{ytest.shape}")
