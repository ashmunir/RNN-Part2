# model_utils.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, Bidirectional

def simple_rnn_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=seq_length),
        SimpleRNN(rnn_units),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def lstm_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=seq_length),
        LSTM(rnn_units),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def bidirectional_lstm_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=seq_length),
        Bidirectional(LSTM(rnn_units)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def gru_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=seq_length),
        GRU(rnn_units),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
