# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers import Bidirectional, Masking, Embedding, concatenate
from keras.layers import BatchNormalization, Activation, TimeDistributed
from keras.layers import Conv1D, GlobalMaxPooling1D, Lambda
from keras.optimizers import Adam
from keras import backend
from attention_model import AttentionLayer
from sklearn.utils import shuffle
import numpy as np
from data_preprocessing import data
from keras.callbacks import TensorBoard
import os
import scipy.io as scio
from sklearn.metrics import confusion_matrix
import random
import pyexcel as pe
'''
@Ruiyu
2019.02.01
ToDo: Text branch only, hi mode RNN -> Attach result to mid mode RNN
'''
## TRAING PARAMS
batch_size = 32
epoch_count_1 = 10
epoch_count_2 = 10
acc_flag_threshould = 60 # threshould of flag to detect in-training effects, not must
acc_collection = [] # all accuracies
work_path = '/Volumes/Detchue Base II/731/CNMC/hospital_data'
saving_path = '/Volumes/Detchue Base II/731/CNMC'
saving_name = ['/result/train_text.mat', '/result/test_text.mat']

## LOAD DATA
cirno = data(path = work_path) # all train/test data
cirno.auto_process(merge_unclear = True)
cirno.label_mode = 'h'
numclass = len(cirno.label_dic_h)

## IN-TRAINING FUNCTIONS
def output_result(train_text, test_text):
    train_text_save_name = saving_path + saving_name[0]
    if os.path.exists(train_text_save_name): os.remove(train_text_save_name)
    scio.savemat(train_text_save_name)

    test_text_save_name = saving_path + saving_name[1]
    if os.path.exists(test_text_save_name): os.remove(test_text_save_name)
    scio.savemat(test_text_save_name)

def weight_expand(x):
    return backend.expand_dims(x)

def weight_dot(inputs):
    return inputs[0] * inputs[1]

def to_one_digit_label(onehot_labels):
    res = []
    for label in onehot_labels: res.append(np.argmax(label))
    return res

## TEXT MODEL HI-LEVEL-LABEL
# input and its shape
text_input = Input(shape = (30,))
# word embedding
em_text = Embedding(len(cirno.word_dic) + 1,
                    200,
                    weights = [cirno.get_embed_matrix()],
                    trainable = True)(text_input)
# masking layer
text = Masking(mask_value = 0.)(em_text)
# LSTM layer
text = LSTM(512,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'LSTM_text_1')(text)
text = LSTM(256,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'LSTM_text_2')(text)
# batch normalization
#text_l1 = BatchNormalization()(text_l1)
# attention layer
text_weight = AttentionLayer()(text)
text_weight = Lambda(weight_expand)(text_weight)
text_vector = Lambda(weight_dot)([text, text_weight])
text_feature_vector = Lambda(lambda x: backend.sum(x, axis = 1))(text_vector)
# dropout layer
dropout_text = Dropout(0.25)(text_feature_vector)
dense_text_1 = Dense(128, activation = 'relu')(dropout_text)
dropout_text = Dropout(0.25)(dense_text_1)
# decision-making
text_prediction = Dense(numclass, activation = 'softmax')(dropout_text)
text_model = Model(inputs = text_input, outputs = text_prediction)
inter_text = Model(inputs = text_input, outputs = text_feature_vector)
# optimizer
adam = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
text_model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
text_model.summary()

## TEXT MODEL MID-LEVEL-LABEL
numclass = len(cirno.label_dic_m)
# input and its shape
text_input_2 = Input(shape = (30,))
# word embedding
em_text_2 = Embedding(len(cirno.word_dic) + 1,
                    200,
                    weights = [cirno.get_embed_matrix()],
                    trainable = True)(text_input_2)
# masking layer
text_2 = Masking(mask_value = 0.)(em_text_2)
# LSTM layer
text_2 = LSTM(512,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'LSTM_text_1')(text_2)
text_2 = LSTM(256,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'LSTM_text_2')(text_2)
# batch normalization
#text_l1 = BatchNormalization()(text_l1)
# attention layer
text_weight_2 = AttentionLayer()(text_2)
text_weight_2 = Lambda(weight_expand)(text_weight_2)
text_vector_2 = Lambda(weight_dot)([text_2, text_weight_2])
text_feature_vector_2 = Lambda(lambda x: backend.sum(x, axis = 1))(text_vector_2)
# dropout layer
dropout_text_2 = Dropout(0.25)(text_feature_vector_2)
dense_text_1_2 = Dense(128, activation = 'relu')(dropout_text_2)
dropout_text_2 = Dropout(0.25)(dense_text_1_2)
# decision-making
text_prediction_2 = Dense(numclass, activation = 'softmax')(dropout_text_2)
text_model_2 = Model(inputs = text_input_2, outputs = text_prediction_2)
inter_text_2 = Model(inputs = text_input_2, outputs = text_feature_vector_2)
# optimizer
adam = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
text_model_2.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
text_model_2.summary()

## MAIN
if __name__ == "__main__":
    # HI-LEVEL-LABEL MODE TRAINING
    acc_max = 0
    for i in range(epoch_count_1):
        print('\n\n>>>High-level-label Text Training Epoch: ' + str(i + 1) + ' out of ' + str(epoch_count_1))
        # get data
        test_label, test_text, test_audio_left, test_audio_right = cirno.get_tester(average = True)
        train_label, train_text, train_audio_left, train_audio_right = cirno.get_trainer(average = True)
        # train
        text_model.fit(train_text,
                       train_label,
                       batch_size = batch_size,
                       epochs = 1,
                       verbose = 1)
        # evaluate
        loss, acc = text_model.evaluate(test_text,
                                            test_label,
                                            batch_size = batch_size,
                                            verbose = 0)
        acc_collection.append(acc)
        print('>Accuaracy = {:.2f} | Loss = {:.2f}'.format(acc, loss))
        #weights = text_model.get_weights()
        cirno.write_epoch_acc(i, acc) # record in analyze file
        if acc > acc_max:
            acc_max = acc
            text_model.save_weights(saving_path + 'entire_text_output_weights.h5')
            #inter_text.save_weights(saving_path+'inter_text_output_weights.h5')
    # load final(best) weights
    #inter_text.load_weights(saving_path+'inter_text_output_weights.h5')
    #text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    #train_text_inter = inter_text.predict(train_text, batch_size=batch_size)
    #test_text_inter = inter_text.predict(test_text, batch_size=batch_size)

    print('\n\n>>>High-level-label Training All Done')
    print('>Max Text Acc =', acc_max)

    # MID-LEVEL-LABEL MODE TRAINING
    cirno.label_mode = 'm'
    text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    acc_max = 0
    for i in range(epoch_count_2):
        print('\n\n>>>Mid-level-label Text Training Epoch: ' + str(i + 1) + ' out of ' + str(epoch_count_2))
        # get data


