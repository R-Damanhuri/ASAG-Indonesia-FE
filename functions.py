import pandas as pd
import re
from datasets import Dataset
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, util
import math
from sentence_transformers import losses, evaluation
from torch.utils.data import DataLoader
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Sequential

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def preprocess(dataframe):
    col_filtered = dataframe.filter(['NIM','Soal','Jawaban'])
    col_filtered['Soal_Clean'] = col_filtered['Soal'].apply(clean)
    col_filtered['Jawaban_Clean'] = col_filtered['Jawaban'].apply(clean)
    return col_filtered

def sentence_embedding(model_path, data):
    dataset = Dataset.from_pandas(data)
    train_examples = [InputExample(texts=[item['Jawaban_Clean'], item['Jawaban_Clean']]) for item in dataset]
    train_examples.append(InputExample(texts=[dataset['Soal_Clean'][0], dataset['Soal_Clean'][0]]))
    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=8)

    model = SentenceTransformer(model_path)
    
    train_loss = losses.MultipleNegativesRankingLoss(model)
    num_epochs = 1
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps
    )

    q_vector = model.encode(data["Soal"])
    a_vector = model.encode(data["Jawaban"])
    return q_vector, a_vector

def split(data):
    similarities = util.cos_sim(data['Soal_Embed'], data['Jawaban_Embed'])
    pairs = []
    for i in range(similarities.shape[0]):
        for j in range(similarities.shape[1]):
            if i == j:
                pairs.append(similarities[i][j].item()) 

    data['Cosine'] = pairs
    data_sorted = data.sort_values(by=['Cosine'], ascending=False)

    total_rows = data_sorted.shape[0]
    top_6_percent = int(0.06 * total_rows)
    mid_8_percent = int(0.08 * total_rows)
    bottom_6_percent = int(0.06 * total_rows)

    mid_start = (total_rows - mid_8_percent) // 2
    mid_end = mid_start + mid_8_percent

    top_6_percent_data = data_sorted.head(top_6_percent)
    mid_8_percent_data = data_sorted.iloc[mid_start:mid_end]
    bottom_6_percent_data = data_sorted.tail(bottom_6_percent)

    data_selected = pd.concat([top_6_percent_data,mid_8_percent_data,bottom_6_percent_data],axis=0)
    data_test = data_sorted.drop(data_selected.index)

    return data_selected, data_test

def match_matrix(X):
  X_matrix = []
  for i in range(X['Jawaban_Embed'].shape[0]):
    Sq_i = X['Soal_Embed'].iloc[i] #Representasi satu kalimat soal ke-i dari dataset
    Sq_i = np.array(Sq_i).reshape(1, -1) #Menggabungkan tiap dimensi menjadi satu array [ , , ] --> [[ , , ]]

    Sa_i = X['Jawaban_Embed'].iloc[i] #Representasi satu kalimat jawaban ke-i dari dataset
    Sa_i = np.array(Sa_i).reshape(1, -1) #Menggabungkan tiap dimensi menjadi satu array [ , , ] --> [[ , , ]]

    X_matrix.append(np.dot(Sq_i.T,Sa_i)) #Sq_i.T melakukan tranpose sehingga tiap dimensi terpecah menjadi array sendiri-sendiri [[ , , ]] --> [[],[],[]]

  X_matrix_stacked = np.stack(X_matrix, axis=0)
  return X_matrix_stacked

@tf.keras.utils.register_keras_serializable(package="SMAPE", name="smape")
def smape(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)

  epsilon = K.epsilon()
  numerator = K.abs(y_pred - y_true)
  denominator = K.maximum((K.abs(y_true) + K.abs(y_pred)), epsilon)
  diff = numerator/denominator
  return 100.0 * K.mean(diff, axis=-1)

@tf.keras.utils.register_keras_serializable(package="SMAPE", name="SMAPE")
class SMAPE(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name="SMAPE", dtype=None):
        super().__init__(smape, name, dtype=dtype)

def model_builder():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(256, 256)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='relu'))

    smape = SMAPE()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError(),
            smape
        ]
    )
    return model