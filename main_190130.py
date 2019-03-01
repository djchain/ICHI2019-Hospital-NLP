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
2019.01.30
ToDo: Text branch only, hi mode RNN
'''
## TRAING PARAMS
batch_size = 32
epoch_count = 300
acc_flag_threshould = 60 # threshould of flag to detect in-training effects, not must
acc_collection = [] # all accuracies
work_path = 'D:/CNMC/hospital_data'
saving_path = 'D:/CNMC'
saving_name = ['/result/train_text.mat', '/result/test_text.mat']
label_mode = 'lower_10'

## LOAD DATA
cirno = data(path = work_path) # all train/test data
cirno.auto_process(merge_unclear = True)
cirno.label_mode = label_mode
if label_mode == 'lower_10':
    numclass = 11
elif label_mode == 'h':
    numclass = len(cirno.label_dic_h)
elif label_mode == 'm':
    numclass = len(cirno.label_dic_m)
elif label_mode == 'l':
    numclass = len(cirno.label_dic_l)
else:
    numclass = len(cirno.trainer_lbl_statistics) - len(cirno.unclear_lbl) + 4
    print('>!>Warning, unknown label mode')

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

# callback for tensorboard
tb_callback = TensorBoard(log_dir = work_path + '/analyze/tensorboard',
                                         histogram_freq = 1,
                                         write_graph = True,
                                         write_images = True)

## TEXT MODEL
# input and its shape
text_input = Input(shape = (30,), name = 'ph1_input')
# word embedding
em_text = Embedding(len(cirno.word_dic) + 1,
                    200,
                    weights = [cirno.get_embed_matrix()],
                    trainable = True)(text_input)
# masking layer
text = Masking(mask_value = 0.,
               name = 'ph1_mask')(em_text)
# LSTM layer
text = LSTM(512,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'ph1_LSTM_text_1')(text)
text = LSTM(256,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'ph1_LSTM_text_2')(text)
# batch normalization
#text_l1 = BatchNormalization(name=)(text_l1)
# attention layer
text_weight = AttentionLayer(name = 'ph1_att')(text)
text_weight = Lambda(weight_expand, name = 'ph1_lam1')(text_weight)
text_vector = Lambda(weight_dot, name = 'ph1_lam2')([text, text_weight])
text_feature_vector = Lambda(lambda x: backend.sum(x, axis = 1), name = 'ph1_lam3')(text_vector)
# dropout layer
dropout_text = Dropout(0.25, name = 'ph1_drop1')(text_feature_vector)
dense_text_1 = Dense(128, activation = 'relu', name = 'ph1_dense')(dropout_text)
dropout_text = Dropout(0.25, name = 'ph1_drop2')(dense_text_1)
# decision-making
text_prediction = Dense(numclass, activation = 'softmax', name = 'ph1_dec')(dropout_text)
text_model = Model(inputs = text_input, outputs = text_prediction, name = 'ph1_model')
#inter_text = Model(inputs = text_input, outputs = text_feature_vector)
# optimizer
adam = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
text_model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
text_model.summary()

## MAIN
if __name__ == "__main__":
    acc_max = 0
    #text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    for i in range(1, epoch_count + 1):
        print('\n\n>>>High-level-label Text Training Epoch: ' + str(i) + ' out of ' + str(epoch_count))
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

    # Calc confusion matrix
    test_label, test_text, _1, _2 = cirno.get_tester(debug=True)
    predictions = text_model.predict(test_text)
    #np.savetxt(work_path + "test_label.txt", test_label, fmt='%.3f')
    #np.savetxt(work_path + "predictions.txt", predictions, fmt='%.3f')
    confusion = confusion_matrix(np.argmax(test_label, axis=1), np.argmax(predictions, axis=1))
    print(confusion)
    np.savetxt(work_path + "/analyze/confusion_matrix.csv", confusion, fmt = '%.0f', delimiter=",")


    #print(cirno.lower_10_transfer_dic)
    #print(cirno.tester_lbl_dic)
    index_to_label = {}
    for label, raw_index in cirno.tester_lbl_dic.items():
        index = cirno.lower_10_transfer_dic[raw_index]
        if index_to_label.get(index) != 'NULL':
            index_to_label[index] = label
    print(index_to_label)