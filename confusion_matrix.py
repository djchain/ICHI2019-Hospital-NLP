from keras import Model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Masking, Embedding, concatenate
from keras.layers import BatchNormalization, Activation, TimeDistributed, Conv1D, GlobalMaxPooling1D, Lambda
from keras.optimizers import Adam
from keras import backend
from attention_model import AttentionLayer
import numpy as np
from data_preprocessing import data
from sklearn.metrics import confusion_matrix

'''
@Ruiyu
2019.02.18
ToDo: complete confusion matrix helper tool
'''

def get_text_confusion_matrix(model, test_text, test_label):
    pass


if __name__ == "__main__":
    saving_path = '/Volumes/Detchue Base II/731/CNMC'
    work_path = '/Volumes/Detchue Base II/731/CNMC/hospital_data'
    cirno = data(path=work_path)
    cirno.label_mode = 'lower_10'
    numclass = 11
    ####################################################### MODEL ##################################
    ## TEXT MODEL
    def weight_expand(x): return backend.expand_dims(x)
    def weight_dot(inputs): return inputs[0] * inputs[1]
    # input and its shape
    text_input = Input(shape=(30,), name='ph1_input')
    # word embedding
    em_text = Embedding(len(cirno.word_dic) + 1,
                        200,
                        weights=[cirno.get_embed_matrix()],
                        trainable=True)(text_input)
    # masking layer
    text = Masking(mask_value=0.,
                   name='ph1_mask')(em_text)
    # LSTM layer
    text = LSTM(512,
                return_sequences=True,
                recurrent_dropout=0.25,
                name='ph1_LSTM_text_1')(text)
    text = LSTM(256,
                return_sequences=True,
                recurrent_dropout=0.25,
                name='ph1_LSTM_text_2')(text)
    # batch normalization
    # text_l1 = BatchNormalization(name=)(text_l1)
    # attention layer
    text_weight = AttentionLayer(name='ph1_att')(text)
    text_weight = Lambda(weight_expand, name='ph1_lam1')(text_weight)
    text_vector = Lambda(weight_dot, name='ph1_lam2')([text, text_weight])
    text_feature_vector = Lambda(lambda x: backend.sum(x, axis=1), name='ph1_lam3')(text_vector)
    # dropout layer
    dropout_text = Dropout(0.25, name='ph1_drop1')(text_feature_vector)
    dense_text_1 = Dense(128, activation='relu', name='ph1_dense')(dropout_text)
    dropout_text = Dropout(0.25, name='ph1_drop2')(dense_text_1)
    # decision-making
    text_prediction = Dense(numclass, activation='softmax', name='ph1_dec')(dropout_text)
    text_model = Model(inputs=text_input, outputs=text_prediction, name='ph1_model')
    # inter_text = Model(inputs = text_input, outputs = text_feature_vector)
    # optimizer
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    text_model.summary()
    ####################################################### END OF MODEL ############################
    text_model.load_weights(saving_path + 'entire_text_output_weights.h5')
    test_label, test_text, _1, _2 = cirno.get_tester()
    predictions = text_model.predict(test_text)
    confusion = confusion_matrix(test_label, predictions)
    print(confusion)