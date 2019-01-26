"""
Created on Jan 17, 2019
Group activity label classification based on text data.
Using transformer structure (self-attention) without any fusion models.
Experiment is based on 67 trauma cases, input samples is sentence-level data.
@author: Yue Gu, Ruiyu Zhang, Xinwei Zhao
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
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

epoch = 2000
batch_size = 32
head_num = 8
head_size = 16


# Model Architecture
# Text data
# define text input and shape
text_input = Input(shape=(30,))
# word embedding
em_text = Embedding(len(gakki.word_dic) + 1, 200, weights=[gakki.get_embed_matrix()], trainable=True)(text_input)
x = Position_Embedding()(em_text)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)

x = GlobalMaxPooling1D()(x)

# decision-making
x = Dense(32)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Dense(16)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
prediction = Dense(num_class, activation='softmax')(x)
print('prediction shape: ', prediction.shape)
text_model = Model(inputs=text_input, outputs=prediction)

# optimizer
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
text_model.summary()


if __name__ == "__main__":
    # Text model training
    text_acc = 0
    for i in range(epoch):
        # data loader (balance data)
        test_label, test_text, test_audio_left, test_audio_right = gakki.get_tester(average=True)
        train_label, train_text, train_audio_left, train_audio_right = gakki.get_trainer(average=True)

        print('text branch, epoch: ', str(i))
        text_model.fit(train_text,
                       train_label,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1)
        # callbacks=[TensorBoard(log_dir=saving_path+'\\hospital_data\\analyze\\log_dir\\')])

        loss_t, acc_t = text_model.evaluate(test_text,
                                            test_label,
                                            batch_size=batch_size,
                                            verbose=0)

        print('epoch: ', str(i))
        print('loss_a', loss_t, ' ', 'acc_a', acc_t)
        gakki.write_epoch_acc(i, acc_t, name='Text')
        if acc_t >= text_acc:
            text_acc = acc_t
            """
            if i >= 0:
                text_model.save_weights(saving_path + 'text_transformer_weights.h5')
            """
    print('final_acc: ', text_acc)
