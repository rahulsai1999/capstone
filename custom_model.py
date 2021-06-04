import os
import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GlobalMaxPooling2D, Dropout, Add, MaxPooling2D, GRU, AveragePooling2D
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Embedding, LSTM, Dot, Reshape, Concatenate, BatchNormalization


chexnet_weights = "weights/chexnet.h5"


def create_chexnet(weights=chexnet_weights, input_size=(224, 224)):
    model = tf.keras.applications.DenseNet121(
        include_top=False, input_shape=input_size+(3,))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)

    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(weights)
    chexnet = tf.keras.Model(
        inputs=model.input, outputs=chexnet.layers[-3].output)
    return chexnet


class Image_encoder(tf.keras.layers.Layer):
    def __init__(self, name="image_encoder"):
        super().__init__()
        self.chexnet = create_chexnet(input_size=(224, 224))
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D()

    def call(self, data):
        output = self.chexnet(data)
        output = self.avgpool(output)
        output = tf.reshape(
            output, shape=(-1, output.shape[1]*output.shape[2], output.shape[3]))
        return output


def encoder(img1, img2, dense_dim, dropout_rate):
    img_enc = Image_encoder()
    dense = Dense(dense_dim, name='enc_dense',
                  activation='relu')
    imf1 = img_enc(img1)
    imf1 = dense(imf1)

    imf2 = img_enc(img2)
    imf2 = dense(imf2)

    concat = Concatenate(axis=1)([imf1, imf2])
    bn = BatchNormalization(name="encoder_batch_norm")(concat)
    dropout = Dropout(dropout_rate, name="encoder_dropout")(bn)
    return dropout


class global_attention(tf.keras.layers.Layer):
    def __init__(self, dense_dim):
        super().__init__()
        self.W1 = Dense(units=dense_dim)
        self.W2 = Dense(units=dense_dim)
        self.V = Dense(units=1)

    def call(self, encoder_output, decoder_h):
        decoder_h = tf.expand_dims(decoder_h, axis=1)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)
        tanh_output = tf.nn.tanh(tanh_input)
        attention_weights = tf.nn.softmax(self.V(tanh_output), axis=1)
        output = attention_weights*encoder_output
        context_vector = tf.reduce_sum(output, axis=1)
        return context_vector, attention_weights


class One_Step_Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim, name="onestepdecoder"):

        super().__init__()
        self.dense_dim = dense_dim
        self.embedding = Embedding(input_dim=vocab_size+1,
                                   output_dim=embedding_dim,
                                   input_length=max_pad,
                                   mask_zero=True,
                                   name='onestepdecoder_embedding'
                                   )
        self.LSTM = GRU(units=self.dense_dim,
                        return_state=True,
                        name='onestepdecoder_LSTM'
                        )
        self.attention = global_attention(dense_dim=dense_dim)
        self.concat = Concatenate(axis=-1)
        self.dense = Dense(
            dense_dim, name='onestepdecoder_embedding_dense', activation='relu')
        self.final = Dense(vocab_size+1, activation='softmax')
        self.concat = Concatenate(axis=-1)
        self.add = Add()

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_h):
        embedding_op = self.embedding(
            input_to_decoder)
        context_vector, attention_weights = self.attention(
            encoder_output, decoder_h)
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        concat_input = self.concat([context_vector_time_axis, embedding_op])
        output, decoder_h = self.LSTM(concat_input, initial_state=decoder_h)
        output = self.final(output)
        return output, decoder_h, attention_weights


class decoder(tf.keras.Model):
    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size):
        super().__init__()
        self.onestepdecoder = One_Step_Decoder(
            vocab_size=vocab_size, embedding_dim=embedding_dim, max_pad=max_pad, dense_dim=dense_dim)
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim

    @tf.function
    def call(self, encoder_output, caption):
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:, 0]), tf.zeros_like(
            encoder_output[:, 0])
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        for timestep in range(self.max_pad):
            output, decoder_h, attention_weights = self.onestepdecoder(
                caption[:, timestep:timestep+1], encoder_output, decoder_h)
            output_array = output_array.write(timestep, output)
        self.output_array = tf.transpose(output_array.stack(), [1, 0, 2])
        return self.output_array


def create_model():
    input_size = (224, 224)
    tokenizer = joblib.load('tkn.pkl')
    max_pad = 29
    batch_size = 100
    vocab_size = len(tokenizer.word_index)
    embedding_dim = 300
    dense_dim = 512
    lstm_units = dense_dim
    dropout_rate = 0.2

    tf.keras.backend.clear_session()
    img1 = Input(shape=(input_size + (3,)))
    img2 = Input(shape=(input_size + (3,)))
    caption = Input(shape=(max_pad,))
    encoder_output = encoder(img1, img2, dense_dim,
                             dropout_rate)
    output = decoder(max_pad, embedding_dim, dense_dim,
                     batch_size, vocab_size)(encoder_output, caption)
    model = tf.keras.Model(inputs=[img1, img2, caption], outputs=output)
    model_filename = 'weights/model.h5'
    model_save = model_filename
    model.load_weights(model_save)

    return model, tokenizer


def greedy_search_predict(img1, img2, model, tokenizer, input_size=(224, 224)):
    img1 = tf.expand_dims(cv2.resize(
        img1, input_size, interpolation=cv2.INTER_NEAREST), axis=0)  # introduce batch and resize
    img2 = tf.expand_dims(cv2.resize(
        img2, input_size, interpolation=cv2.INTER_NEAREST), axis=0)

    img1 = model.get_layer('image_encoder')(img1)
    img2 = model.get_layer('image_encoder')(img2)
    img1 = model.get_layer('enc_dense')(img1)
    img2 = model.get_layer('enc_dense')(img2)
    concat = model.get_layer('concatenate')([img1, img2])
    enc_op = model.get_layer('encoder_batch_norm')(concat)
    enc_op = model.get_layer('encoder_dropout')(
        enc_op)

    decoder_h, decoder_c = tf.zeros_like(
        enc_op[:, 0]), tf.zeros_like(enc_op[:, 0])
    a = []
    pred = []
    max_pad = 29
    for i in range(max_pad):
        if i == 0:
            caption = np.array(tokenizer.texts_to_sequences(
                ['<cls>']))
        output, decoder_h, attention_weights = model.get_layer(
            'decoder').onestepdecoder(caption, enc_op, decoder_h)
        max_prob = tf.argmax(output, axis=-1)
        caption = np.array([max_prob])
        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])):
            break
        else:
            a.append(tf.squeeze(max_prob).numpy())
    return tokenizer.sequences_to_texts([a])[0]


def get_bleu(reference, prediction):

    reference = [reference.split()]
    prediction = prediction.split()
    bleu1 = sentence_bleu(reference, prediction, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, prediction,
                          weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


def predict1(img1, img2=None, model_tokenizer=None):
    if img2 is None:
        img2 = img1

    if model_tokenizer == None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(img1, img2, model, tokenizer)

    return predicted_caption


def predict2(true_caption, img1, img2=None, model_tokenizer=None):
    if img2 == None:
        img2 = img1

    try:
        img1 = cv2.imread(img1, cv2.IMREAD_UNCHANGED)/255
        img2 = cv2.imread(img2, cv2.IMREAD_UNCHANGED)/255
    except:
        return print("Must be an image")

    if model_tokenizer == None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(img1, img2, model, tokenizer)

    _ = get_bleu(true_caption, predicted_caption)
    _ = list(_)
    return pd.DataFrame([_], columns=['bleu1', 'bleu2', 'bleu3', 'bleu4'])


def function1(img1, img2, model_tokenizer=None):
    if model_tokenizer is None:
        model_tokenizer = list(create_model())
    predicted_caption = []
    for i1, i2 in zip(img1, img2):
        caption = predict1(i1, i2, model_tokenizer)
        predicted_caption.append(caption)

    return predicted_caption


def function2(true_caption, img1, img2):
    model_tokenizer = list(create_model())
    predicted = pd.DataFrame(columns=['bleu1', 'bleu2', 'bleu3', 'bleu4'])
    for c, i1, i2 in zip(true_caption, img1, img2):
        caption = predict2(c, i1, i2, model_tokenizer)
        predicted = predicted.append(caption, ignore_index=True)

    return predicted
