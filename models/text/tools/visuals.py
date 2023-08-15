import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def show_bert_accuracy(model, test_x, test_y):
    prediction = model.predict({'input_ids':test_x['input_ids'],'attention_mask':test_x['attention_mask']})

    prediction = np.argmax(prediction, axis = 1)

    print(f"Accuracy score: {accuracy_score(test_y, prediction)}")