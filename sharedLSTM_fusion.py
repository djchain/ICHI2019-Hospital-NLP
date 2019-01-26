# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers import Bidirectional, Masking, concatenate
from keras.layers import BatchNormalization, Activation, Lambda
from keras.optimizers import Adam
from keras import backend
from attention_model import AttentionLayer
import numpy as np

from data_preprocessing import data

batch_size = 32
epo = 100
numclass = 7

result_t, result_a = [], []

# loading training testing data here
"""
fill the data processing in this part
"""

path = r'E:/Yue/Entire Data/CNMC/result/'
lay = data(path)
train_context, train_ori = lay.get_train_data(path)
test_context, test_ori = lay.get_test_data()
test_label, test_text, test_audio_left, test_audio_right = lay.get_tester()
train_label, train_text, train_audio_left, train_audio_right = lay.get_trainer()

# define the operations

def fusion_weight_expand(x):
    a = np.zeros((1, 512), dtype='float32')
    a[0:] = x[0]
    t = np.zeros((1, 512), dtype='float32')
    t[0:] = x[1]
    r = np.concatenate((a, t))
    return r

def weight_expand(x):
    return backend.expand_dims(x)

def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y

# Contextual branch
context_input = Input(shape=(537, 64))
context_input = Masking(mask_value=0.)(context_input)
context_l1 = Bidirectional(LSTM(256, return_sequences=True,
                                recurrent_dropout=0.25, name='contextual_LSTM'))(context_input)
context_weight = AttentionLayer()(context_l1)
context_weight_exp = Lambda(weight_expand)(context_weight)
context_attention = Lambda(weight_dot)([context_l1, context_weight_exp])
context_att = Lambda(lambda x: backend.sum(x, axis=1))(context_attention)
dropout_context = Dropout(0.25)(context_att)

# Original Branch
ori_input = Input(shape=(537, 64))
ori_input = Masking(mask_value=0.)(ori_input)
ori_l1 = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.25, name='contextual_LSTM'))(ori_input)
ori_weight = AttentionLayer()(ori_l1)
ori_weight_exp = Lambda(weight_expand)(ori_weight)
ori_attention = Lambda(weight_dot)([ori_l1, ori_weight_exp])
ori_att = Lambda(lambda x: backend.sum(x, axis=1))(ori_attention)
dropout_ori = Dropout(0.25)(ori_att)

# Fusion Model

# Contextual ConvNet
c_1 = Dense(256)(dropout_context)
batch_c1 = BatchNormalization()(c_1)
c_1 = Activation('tanh')(batch_c1)
c_2 = Dense(128)(c_1)
batch_c2 = BatchNormalization()(c_2)
c_2 = Activation('tanh')(batch_c2)
c_3 = Dense(64)(c_2)
batch_c3 = BatchNormalization()(c_3)
c_3 = Activation('tanh')(batch_c3)
c_4 = Dense(16)(c_3)
batch_c4 = BatchNormalization()(c_4)
c_4 = Activation('tanh')(batch_c4)
c_5 = Dense(1)(c_4)
batch_c5 = BatchNormalization()(c_5)
c_5 = Activation('tanh')(batch_c5)


# Original ConvNet
o_1 = Dense(256)(dropout_ori)
batch_o1 = BatchNormalization()(o_1)
o_1 = Activation('tanh')(batch_o1)
o_2 = Dense(128)(o_1)
batch_o2 = BatchNormalization()(o_2)
o_2 = Activation('tanh')(batch_o2)
o_3 = Dense(64)(o_2)
batch_o3 = BatchNormalization()(o_3)
o_3 = Activation('tanh')(batch_o3)
o_4 = Dense(16)(o_3)
batch_o4 = BatchNormalization()(o_4)
o_4 = Activation('tanh')(batch_o4)
o_5 = Dense(1)(o_4)
batch_o5 = BatchNormalization()(o_5)
o_5 = Activation('tanh')(batch_o5)

# Merge Layer
merge_dim_normal = concatenate([dropout_context, dropout_ori], name='merge_dim_normal')
merge_dim_2 = concatenate([c_5, o_5], name='merge_dim_2')
fusion_weight = AttentionLayer()(merge_dim_2)
fusion_weight_exp = Lambda(fusion_weight_expand)(fusion_weight)
fusion_attention = Lambda(weight_dot)([merge_dim_normal, fusion_weight_exp])
fusion_att = Lambda(lambda x: backend.sum(x, axis=1))(fusion_attention)
fusion_dense_1 = Dense(256)(fusion_att)
fusion_dense_1 = BatchNormalization()(fusion_dense_1)
fusion_dense_1 = Activation('relu')(fusion_dense_1)
fusion_dropout_1 = Dropout(0.25)(fusion_dense_1)
fusion_dense_2 = Dense(64)(fusion_dropout_1)
fusion_dense_2 = BatchNormalization()(fusion_dense_2)
fusion_dense_2 = Activation('relu')(fusion_dense_2)
fusion_dropout_2 = Dropout(0.25)(fusion_dense_2)

# Decision Making
f_prediction = Dense(numclass, activation='softmax')(fusion_dropout_2)
final_model = Model(inputs=[context_input, ori_input], outputs=f_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
final_model.summary()


if __name__ == "__main__":
    final_acc = 0
    for i in range(epo):
        print('fusion branch, epoch: ', str(i))
        final_model.fit([train_context, train_ori], train_label, batch_size=batch_size, epochs=1)
        loss_f, acc_f = final_model.evaluate([test_context, test_ori], test_label,
                                             batch_size=batch_size, verbose=0)
        print('epoch: ', str(i))
        print('loss_f', loss_f, ' ', 'acc_f', acc_f)
        if acc_f >= final_acc:
            final_acc = acc_f

    # Result
    print('final result: ')
    print(' final acc: ', final_acc)
