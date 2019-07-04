#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:10:11 2019

@author: Veloc1ty
"""
import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import optimizers

def load_data():
    filename = "evermore.txt"
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    return raw_text

def mapping(raw_text):    
    chars = sorted(list(set(str(raw_text))))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    return chars, char_to_int, int_to_char

def prepare_data(raw_text, seq_length, n_chars, n_vocab, char_to_int):
    seq_length = seq_length
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = np_utils.to_categorical(dataY)
    return dataX, dataY, X, y

def build(X, y, optimizer, pretrained=False, filename=None):    
    model = Sequential()
    model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LSTM(160, return_sequences=True))
    model.add(LSTM(160))
    model.add(Dense(y.shape[1], activation='softmax'))
    if pretrained:
        model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

if __name__ == '__main__':
    raw_text = load_data()
    chars, char_to_int, int_to_char = mapping(raw_text)
    seq_length, n_chars, n_vocab = 100, len(raw_text), len(chars)
    dataX, dataY, X, y = prepare_data(raw_text, seq_length, len(raw_text), 
                                  len(chars), char_to_int)
    adam = optimizers.adam(lr=1e-3, decay=1e-5)
    model = build(X, y, optimizer=adam)
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

    