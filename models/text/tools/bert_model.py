import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPool1D
from transformers import TFBertModel

MAX_LEN = 128
bert = TFBertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

def get_model(maxlen=MAX_LEN):
    input_ids = Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(maxlen,), dtype=tf.int32, name="attention_mask")
    embeddings = bert(input_ids,attention_mask = input_mask)[1] # 0 is last hidden  -> 1 is pooler_output
    out = Dropout(0.1)(embeddings)
    # out = GlobalMaxPool1D()(out)
    out = Dense(128, activation='relu')(embeddings)
    out = Dropout(0.1)(out)
    out = Dense(32,activation = 'relu')(out)
    y = Dense(3,activation = 'softmax')(out)

    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True

    optimizer = Adam(
        learning_rate=2e-5, # this learning rate is for bert model, taken from huggingface website 
        epsilon=1e-08,
        decay=0.01,
        clipnorm=1.0) 

    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy'])

    return model