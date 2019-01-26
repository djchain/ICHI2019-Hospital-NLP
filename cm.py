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

batch_size = 32
epo = [5, 5, 5]#audio,text,fusion
flag = 0
numclass = 7

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

text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
text_model._make_predict_function()

test_label, test_text, test_audio_left, test_audio_right = gakki.get_tester()
print('Number of instances: ', len(test_text))
test_pred = text_model.predict(test_text)
p = to_one_digit_label(test_pred)
gt = to_one_digit_label(test_label)

matches = [i for i, j in zip(p, gt) if i == j]
print('Number of instances: ', len(test_text))
print('Number of matches: ', len(matches))
print('Accuracy of this round: ', (float)(len(matches)/len(test_text)))

cm = confusion_matrix(gt, p)

print(cm)
