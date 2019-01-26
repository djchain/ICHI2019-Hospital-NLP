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
#import data_preprocessing
from data_preprocessing import data
from keras.callbacks import TensorBoard
import os
import scipy.io as scio
from sklearn.metrics import confusion_matrix
import random
import pyexcel as pe
"""
@Ruiyu
1017 update
Trying to load data each epoch, with label averagely distributed, especially "NULL"/"NA" label.
"""
batch_size = 32
epo = [100, 100, 100]#audio,text,fusion
flag = 0
numclass = 7

result_t,result_a=[],[]

##### loading training testing data here
gakki=data(path=r'E:/Yue/Entire Data/CNMC/hospital_data')
saving_path = r'E:/Yue/Entire Data/CNMC/'
output_data = ['/result/train_audio.mat', '/result/train_text.mat', '/result/test_audio.mat', '/result/test_text.mat']

gakki.unclear_lbl.append('Monitor Vital Signs')
gakki.auto_process(merge_unclear=True)
gakki.label_mode='lower_10'

test_label,test_text,test_audio_left,test_audio_right = gakki.get_tester(average=True)
train_label,train_text,train_audio_left,train_audio_right = gakki.get_trainer()
numclass=len(gakki.trainer_lbl_statistics)-len(gakki.unclear_lbl)+4


# define the operations

def output_result(train_audio, train_text, test_audio, test_text):
    if os.path.exists(saving_path + output_data[0]):
        os.remove(saving_path + output_data[0])
    scio.savemat(saving_path + output_data[0], {'train_audio': train_audio})
    if os.path.exists(saving_path + output_data[1]):
        os.remove(saving_path + output_data[1])
    scio.savemat(saving_path + output_data[1], {'train_text': train_text})
    if os.path.exists(saving_path + output_data[2]):
        os.remove(saving_path + output_data[2])
    scio.savemat(saving_path + output_data[2], {'test_audio': test_audio})
    if os.path.exists(saving_path + output_data[3]):
        os.remove(saving_path + output_data[3])
    scio.savemat(saving_path + output_data[3], {'test_text': test_text})


def weight_expand(x):
    return backend.expand_dims(x)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def to_one_digit_label(onehot_labels):
    res = []
    for label in onehot_labels:
        index = np.argmax(label)
        res.append(index)
    return res
'''
def ramdon_select_from(lbl,txt,al,ar,each_count=60):
    #就是从train/test label 里面随机选一些出来
    dic={}
    lbl_,txt_,al_,ar_=[],[],[],[]
    #for i,(l,t,L,R) in enumerate(zip(lbl,txt,al,ar)):
    for i,label in enumerate(lbl):
        l=tuple(label)
        if l in dic:
            dic[l].append(i)
        else:
            dic[l]=[i]
    for l in dic.keys():
        indices=dic[l]
        random.shuffle(indices)
        for index in indices[:min(len(indices),each_count)]:
            lbl_.append(lbl[index])
            txt_.append(txt[index])
            al_.append(al[index])
            ar_.append(ar[index])
    return lbl_,txt_,al_,ar_
'''




###### Audio branch 2

# calculate left audio feature vector
left_input = Input(shape=(602, 64))
left_audio = Masking(mask_value=0.)(left_input)
left_audio = LSTM(256,
             return_sequences=True,
             recurrent_dropout=0.25,
             name='LSTM_left_audio_1')(left_audio)
left_audio = LSTM(128,
             return_sequences=True,
             recurrent_dropout=0.25,
             name='LSTM_left_audio_2')(left_audio)
left_audio_weight = AttentionLayer()(left_audio)
left_audio_weight = Lambda(weight_expand)(left_audio_weight)
left_audio_vector = Lambda(weight_dot)([left_audio, left_audio_weight])
left_audio_feature_vector = Lambda(lambda x: backend.sum(x, axis=1))(left_audio_vector)

# calculate right audio feature vector
right_input = Input(shape=(602, 64))
right_audio = Masking(mask_value = 0.)(right_input)
right_audio = LSTM(256,
              return_sequences=True,
              recurrent_dropout=0.25,
              name='LSTM_right_audio_1')(right_audio)
right_audio = LSTM(128,
              return_sequences=True,
              recurrent_dropout=0.25,
              name='LSTM_right_audio_2')(right_audio)
right_audio_weight = AttentionLayer()(right_audio)
right_audio_weight = Lambda(weight_expand)(right_audio_weight)
right_audio_vector = Lambda(weight_dot)([right_audio, right_audio_weight])
right_audio_feature_vector = Lambda(lambda x: backend.sum(x, axis=1))(right_audio_vector)

# merge layer
audio_feature_vector = concatenate([left_audio_feature_vector, right_audio_feature_vector], name='merge_layer')

# dropout layer
#dropout_audio = Dropout(0.5)(audio)

# decision-making
dropout_audio = Dropout(0.25)(audio_feature_vector)
dense_audio = Dense(128, activation='relu')(dropout_audio)
dropout_audio = Dropout(0.25)(dense_audio)

audio_prediction = Dense(numclass, activation='softmax')(dropout_audio)
audio_model = Model(inputs=[left_input, right_input], outputs=audio_prediction)
inter_audio = Model(inputs=[left_input, right_input], outputs=audio_feature_vector)

# optimizer
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
audio_model.summary()



##### Text Branch

# define text input and shape
text_input = Input(shape=(30,))

# word embedding
em_text = Embedding(len(gakki.word_dic) + 1, 200, weights=[gakki.get_embed_matrix()], trainable=True)(text_input)

# setup masking layer before LSTM layer
text = Masking(mask_value=0.)(em_text)

# LSTM layer
text = LSTM(512,
            return_sequences=True,
            recurrent_dropout=0.25,
            name='LSTM_text_1')(text)

text = LSTM(256,
            return_sequences=True,
            recurrent_dropout=0.25,
            name='LSTM_text_2')(text)

#text_l1 = BatchNormalization()(text_l1)

# attention layer
text_weight = AttentionLayer()(text)
text_weight = Lambda(weight_expand)(text_weight)
text_vector = Lambda(weight_dot)([text, text_weight])
text_feature_vector = Lambda(lambda x: backend.sum(x, axis=1))(text_vector)

# dropout layer
dropout_text = Dropout(0.25)(text_feature_vector)
dense_text_1 = Dense(128, activation='relu')(dropout_text)
dropout_text = Dropout(0.25)(dense_text_1)

# decision-making
text_prediction = Dense(numclass, activation='softmax')(dropout_text)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text = Model(inputs=text_input, outputs=text_feature_vector)

# optimizer
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
text_model.summary()


# Fusion Model
text_fusion_input = Input(shape=(256,))
audio_fusion_input = Input(shape=(256,))
merge = concatenate([text_fusion_input, audio_fusion_input], name='fusion_merge_layer')

merge = Dropout(0.5)(merge)
fusion_1 = Dense(256, activation='relu')(merge)
fusion_1 = Dropout(0.5)(fusion_1)
fusion_2 = Dense(128, activation='relu')(fusion_1)
fusion_2 = Dropout(0.5)(fusion_2)
fusion_3 = Dense(64, activation='relu')(fusion_2)
fusion_3 = Dropout(0.5)(fusion_3)

final_prediction = Dense(numclass, activation='softmax')(fusion_3)
fusion_model = Model(inputs=[text_fusion_input, audio_fusion_input], outputs=final_prediction)

# optimizer
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
fusion_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
fusion_model.summary()


if __name__ == "__main__":
    '''
    test_label, test_text, test_audio_left, test_audio_right = gakki.get_tester()
    #fusion_model.load_weights(saving_path + 'entire_fusion_output_weights.h5')
    #test_pred = fusion_model.predict([test_text_inter, test_audio_inter])
    text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    test_pred = text_model.predict(test_text)
    p = to_one_digit_label(test_pred)
    gt = to_one_digit_label(test_label)
    cm = confusion_matrix(gt, p)

    print(cm)
    '''
    train_label, train_text, train_audio_left, train_audio_right = gakki.get_trainer(average=True)
    inter_text.load_weights(saving_path + 'inter_text_output_weights.h5')
    text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    train_text_inter = inter_text.predict(train_text, batch_size=batch_size)
    test_text_inter = inter_text.predict(test_text, batch_size=batch_size)
    inter_audio.load_weights(saving_path + 'inter_audio_output_weights.h5')
    audio_model.load_weights(saving_path + 'entire_audio_output_weights.h5')
    train_audio_inter = inter_audio.predict([train_audio_left, train_audio_right], batch_size=batch_size)
    test_audio_inter = inter_audio.predict([test_audio_left, test_audio_right], batch_size=batch_size)
    fusion_model.load_weights(saving_path + 'entire_fusion_output_weights.h5')
    '''
    '''
    # text modeling
    text_acc = 0
    for i in range(epo[1]):

        #now balance ratio for data
        test_label, test_text, test_audio_left, test_audio_right = gakki.get_tester(average=True)
        train_label, train_text, train_audio_left, train_audio_right = gakki.get_trainer(average=True)
        print('\n\nText branch, epoch: ', str(i))
        #train_label_t, train_text_t, train_audio_left_t, train_audio_right_t = gakki.crop(train_label0,train_text0,train_audio_left0,train_audio_right0,120)
        text_model.fit(train_text,
                       train_label,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1)
                       #callbacks=[TensorBoard(log_dir=saving_path+'\\hospital_data\\analyze\\log_dir\\')])

        loss_t, acc_t = text_model.evaluate(test_text,
                                            test_label,
                                            batch_size=batch_size,
                                            verbose=0)
        print('>>>epoch: ', str(i))
        print('>loss_t', loss_t, ' ', 'acc_t', acc_t)
        W = text_model.get_weights()
        result_t.append(acc_t)
        gakki.write_epoch_acc(i,acc_t)
        if acc_t >= text_acc:
            text_acc = acc_t
            text_model.save_weights(saving_path+'entire_text_output_weights.h5')
            inter_text.save_weights(saving_path+'inter_text_output_weights.h5')

    inter_text.load_weights(saving_path+'inter_text_output_weights.h5')
    text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    train_text_inter = inter_text.predict(train_text, batch_size=batch_size)
    test_text_inter = inter_text.predict(test_text, batch_size=batch_size)


    # Audio modeling
    audio_acc = 0
    for i in range(epo[0]):
        #now balance ratio for data
        test_label, test_text, test_audio_left, test_audio_right = gakki.get_tester(average=True)
        train_label, train_text, train_audio_left, train_audio_right = gakki.get_trainer(average=True)

        print('audio branch, epoch: ', str(i))
        #train_label_a, train_text_a, train_audio_left_a, train_audio_right_a=gakki.crop(train_label0,train_text0,train_audio_left0,train_audio_right0,120)
        audio_model.fit([train_audio_left, train_audio_right],
                        train_label,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1)

        loss_a, acc_a = audio_model.evaluate([test_audio_left, test_audio_right],
                                             test_label,
                                             batch_size=batch_size,
                                             verbose=0)

        print('epoch: ', str(i))
        print('loss_a', loss_a, ' ', 'acc_a', acc_a)
        result_a.append(acc_a)
        gakki.write_epoch_acc(i, acc_a,name='Audio')
        if acc_a >= audio_acc and acc_a >= flag:
            audio_acc = acc_a
            if i >= 0:
                audio_model.save_weights(saving_path + 'entire_audio_output_weights.h5')
                inter_audio.save_weights(saving_path + 'inter_audio_output_weights.h5')

    inter_audio.load_weights(saving_path + 'inter_audio_output_weights.h5')
    audio_model.load_weights(saving_path + 'entire_audio_output_weights.h5')
    train_audio_inter = inter_audio.predict([train_audio_left, train_audio_right], batch_size=batch_size)
    test_audio_inter = inter_audio.predict([test_audio_left, test_audio_right], batch_size=batch_size)

    print('train_audio_output shape:', train_audio_inter.shape)
    print('train_text_output shape:', train_text_inter.shape)
    print('test_audio_output shape:', test_audio_inter.shape)
    print('test_text_output shape:', test_text_inter.shape)
    output_result(train_audio_inter, train_text_inter, test_audio_inter, test_text_inter)

    # fusion
    final_acc = 0
    for i in range(epo[2]):
        print('fusion branch, epoch: ', str(i))

        fusion_model.fit([train_text_inter, train_audio_inter],
                         train_label,
                         batch_size=batch_size,
                         epochs=1)
        loss_f, acc_f = fusion_model.evaluate([test_text_inter, test_audio_inter],
                                              test_label,
                                              batch_size=batch_size,
                                              verbose=0)
        print('epoch: ', str(i))
        print('loss_f', loss_f, ' ', 'acc_f', acc_f)
        gakki.write_epoch_acc(i, acc_f,name='Fusion')
        if acc_f >= final_acc:
            final_acc = acc_f
            fusion_model.save_weights(saving_path + 'entire_fusion_output_weights.h5')


    print('>>>Training done.')
    print('>Text Acc:' ,result_t)
    print('>Audio Acc:' ,result_a)
    print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
