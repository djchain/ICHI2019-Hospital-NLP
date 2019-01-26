"""
Created on Jan 11 2019
Group activity label classification based on audio data.
Using transformer structure (self-attention) without any fusion models.
Experiment is based on 67 trauma cases, input samples is sentence-level data.
@author: Yue Gu, Ruiyu Zhang, Xinwei Zhao
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Masking
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.optimizers import Adam
from data_preprocessing import data
from transformer import Attention
from transformer import Position_Embedding

# Parameter setting
gakki = data(path=r'E:/Yue/Entire Data/CNMC/hospital_data')
saving_path = r'E:/Yue/Entire Data/CNMC/'

gakki.unclear_lbl.append('Monitor Vital Signs')
gakki.auto_process(merge_unclear=True)
gakki.label_mode = 'lower_10'
num_class = 11

epoch = 20000
batch_size = 16
head_num = 8
head_size = 16


# Model Architecture
# Left audio feature vector
left_input = Input(shape=(602, 64))
#x = Masking(mask_value=0.)(left_input)
x = Position_Embedding()(left_input)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)

left_vector = GlobalMaxPooling1D()(x)

# Right audio feature vector
right_input = Input(shape=(602, 64))
#y = Masking(mask_value=0.)(right_input)
y = Position_Embedding()(right_input)

y = Attention(head_num, head_size)([y, y, y])
y = BatchNormalization()(y)
y = Dropout(0.15)(y)

y = Attention(head_num, head_size)([y, y, y])
y = BatchNormalization()(y)
y = Dropout(0.15)(y)

y = Attention(head_num, head_size)([y, y, y])
y = BatchNormalization()(y)
y = Dropout(0.15)(y)

y = Attention(head_num, head_size)([y, y, y])
y = BatchNormalization()(y)
y = Dropout(0.15)(y)

y = Attention(head_num, head_size)([y, y, y])
y = BatchNormalization()(y)
y = Dropout(0.15)(y)

y = Attention(head_num, head_size)([y, y, y])
y = BatchNormalization()(y)

right_vector = GlobalMaxPooling1D()(y)

# merge layer
fusion_vector = concatenate([left_vector, right_vector], name='merge_layer')
print('right_vector:', right_vector.shape)
print('left_vector:', left_vector.shape)
print("fusion vector shape: ", fusion_vector.shape)

# decision-making
d = Dense(32)(fusion_vector)
d = BatchNormalization()(d)
d = Activation('relu')(d)
d = Dropout(0.15)(d)
d = Dense(16)(d)
d = BatchNormalization()(d)
d = Activation('relu')(d)
# d = Dropout(0.5)(d)
# d = Dense(32)(d)
# d = BatchNormalization()(d)
# d = Activation('relu')(d)
# d = Dropout(0.5)(d)
# d = Dense(16)(d)
# d = BatchNormalization()(d)
# d = Activation('relu')(d)
prediction = Dense(num_class, activation='softmax')(d)
print('prediction shape: ', prediction.shape)
audio_model = Model(inputs=[left_input, right_input], outputs=prediction)

# optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
audio_model.summary()


if __name__ == "__main__":
    # Audio model training
    audio_acc = 0
    for i in range(epoch):
        # data loader (balance data)
        test_label, test_text, test_audio_left, test_audio_right = gakki.get_tester(average=True)
        train_label, train_text, train_audio_left, train_audio_right = gakki.get_trainer(average=True)
        # print('train_label shape: ', train_label.shape)
        # print('train_text shape: ', train_text.shape)
        # print('train_audio_left shape: ', train_audio_left.shape)
        # print('train_audio_right shape: ', train_audio_right.shape)
        # print('test_label shape: ', test_label.shape)
        # print('test_text shape: ', test_text.shape)
        # print('test_audio_left shape: ', test_audio_left.shape)
        # print('test_audio_right shape: ', test_audio_right.shape)
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
        gakki.write_epoch_acc(i, acc_a, name='Audio')
        if acc_a >= audio_acc:
            audio_acc = acc_a
            """
            if i >= 0:
                audio_model.save_weights(saving_path + 'audio_transformer_weights.h5')
            """
    print('final_acc: ', audio_acc)
