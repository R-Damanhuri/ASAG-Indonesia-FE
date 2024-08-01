import streamlit as st

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
    col_filtered = dataframe.filter(['Soal','Jawaban'])
    col_filtered['Soal'] = col_filtered['Soal'].apply(clean)
    col_filtered['Jawaban'] = col_filtered['Jawaban'].apply(clean)
    return col_filtered

def sentence_embedding(model_path, data):
    dataset = Dataset.from_pandas(data)
    train_examples = [InputExample(texts=[item['Jawaban'], item['Jawaban']]) for item in dataset]
    train_examples.append(InputExample(texts=[dataset['Soal'][0], dataset['Soal'][0]]))
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

@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

@st.fragment
def labeling():
    st.subheader("Nilai Sampel Data")
    df_labeled = st.data_editor(
    df_selected,
    disabled=("NIM", "Soal", "Jawaban"),
    column_config={
        "": None,
        "Soal_Embed": None,
        "Jawaban_Embed": None,
        "Cosine": None,
        "Nilai": st.column_config.NumberColumn(
        "Nilai",
        min_value=0,
        max_value=10,
        step = 0.01,
        format="%0.2f"
        )
    }
    )
    st.session_state.data_labeled = df_labeled

@st.fragment
def grading():
    if st.button("Mulai Penilaian"):
    with st.spinner("Sedang melakukan penilaian..."):
        df_train, df_val = train_test_split(st.session_state.data_labeled, random_state = 42, train_size=0.7)
        x_train = df_train[['Soal_Embed','Jawaban_Embed']]
        y_train = df_train['Nilai']

        x_val = df_val[['Soal_Embed','Jawaban_Embed']]
        y_val = df_val['Nilai']

        x_test = df_test[['Soal_Embed','Jawaban_Embed']]

        x_train = match_matrix(x_train)
        x_val = match_matrix(x_val)
        x_test = match_matrix(x_test)

        model = model_builder()
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_SMAPE', mode='min', baseline=2, start_from_epoch=85)
        model.fit(x_train, y_train, validation_data = [x_val,y_val], epochs=100, batch_size=32,callbacks=[stop_early])

        y_predict = model.predict(x_test)
        df_test['Nilai'] = y_predict

        df_result = pd.concat([df_test[['NIM','Soal','Jawaban','Nilai']], st.session_state.data_labeled[['NIM','Soal','Jawaban','Nilai']]],axis=0)
        df_result['Nilai'] = df_result['Nilai'].astype(float).round(2)
        csv_result = convert_df(df_result)
        
        st.success("Penilaian selesai!")
        
        st.download_button(
        label="Unduh CSV",
        data= csv_result,
        file_name="hasil_penilaian.csv",
        mime="text/csv"
        )