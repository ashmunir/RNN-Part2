# models.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Conv1D, MaxPooling1D

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

def bidirectional_gru_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim),
        Bidirectional(GRU(rnn_units)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def stacked_bilstm_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim),
        Bidirectional(LSTM(rnn_units, return_sequences=True)),
        Bidirectional(LSTM(rnn_units)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def cnn_bilstm_model(max_features, seq_length, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(rnn_units)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def transformer_block_model(max_features, seq_length, embedding_dim=128, heads=2):
    inputs = tf.keras.Input(shape=(seq_length,))
    x = Embedding(max_features, embedding_dim)(inputs)

    attn_output = MultiHeadAttention(num_heads=heads, key_dim=embedding_dim)(x, x)
    x = LayerNormalization()(x + attn_output)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
