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

batch_size = 32
epo = [1, 1, 1]
flag = 0
numclass = 10

result_t,result_a=[],[]

##### loading training testing data here
gakki=data(path=r'E:\Yue\Entire Data\CNMC\hospital_data')
saving_path = r'E:\Yue\Entire Data\CNMC'
gakki.auto_process()
test_label,test_text,test_audio_left,test_audio_right=gakki.get_tester(average=True)
train_label,train_text,train_audio_left,train_audio_right=gakki.get_trainer(average=True)

# define the operations

def weight_expand(x):
    return backend.expand_dims(x)

def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y

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
    # text modeling
    text_acc = 0
    for i in range(epo[1]):
        print('\n\nText branch, epoch: ', str(i))
        text_model.fit(train_text,
                       train_label,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       callbacks=[TensorBoard(log_dir=saving_path+'\\hospital_data\\analyze\\log_dir\\')])

        loss_t, acc_t = text_model.evaluate(test_text,
                                            test_label,
                                            batch_size=batch_size,
                                            verbose=0)
        print('epoch: ', str(i))
        print('loss_t', loss_t, ' ', 'acc_t', acc_t)
        W = text_model.get_weights()
        result_t.append(acc_t)
        gakki.write_epoch_acc(i,acc_t)
        if acc_t >= text_acc:
            text_acc = acc_t
            text_model.save_weights(saving_path+'entire_text_output_weights.h5')
            inter_text.save_weights(saving_path+'inter_text_output_weights.h5')
    inter_text.load_weights(saving_path+'inter_text_output_weights.h5')
    train_text_inter = inter_text.predict(train_text, batch_size=batch_size)
    test_text_inter = inter_text.predict(test_text, batch_size=batch_size)


    # Audio modeling
    audio_acc = 0
    for i in range(epo[0]):
        print('audio branch, epoch: ', str(i))
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
    train_audio_inter = inter_audio.predict([train_audio_left, train_audio_right], batch_size=batch_size)
    test_audio_inter = inter_audio.predict([test_audio_left, test_audio_right], batch_size=batch_size)

    print('train_audio_output shape:', train_audio_inter.shape)
    print('train_text_output shape:', train_text_inter.shape)
    print('test_audio_output shape:', test_audio_inter.shape)
    print('test_text_output shape:', test_text_inter.shape)


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
        if acc_f >= final_acc:
            final_acc = acc_f
            fusion_model.save_weights(saving_path + 'entire_fusion_output_weights.h5')


    # print result
    print('>>>Training done.')
    print('>Text Acc:' ,result_t)
    print('>Audio Acc:' ,result_a)
    print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)

