import os
import sys
import numpy as np
import tensorflow as tf
from config import config
from gensim.models import Word2Vec

import keras
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.regularizers import l2
from keras.layers.experimental.preprocessing import TextVectorization


def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_embedding_matrix(vocab, w2v):
    # convert w2v vectors into matrix
    embedding_matrix = np.zeros((len(vocab)+2, config['embedding_dim']))
    for i, word in enumerate(vocab):
        word = word.decode('utf-8')
        try:
            embedding_vector = w2v.wv[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def main(data_dir):
    print('Loading data')
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir)
    # build vocabulary
    vectorizer = TextVectorization(max_tokens=config['max_vocab_size'], output_sequence_length=config['max_seq_len'])
    text_data = tf.data.Dataset.from_tensor_slices(x_train).batch(config['batch_size'])
    print('Building vocabulary')
    vectorizer.adapt(text_data)
    vocab = vectorizer.get_vocabulary()
    # load pre-trained w2v model
    w2v = Word2Vec.load(os.path.join(data_dir, 'w2v.model'))
    # build embedding matrix
    print('Building embedding matrix')
    embedding_matrix = build_embedding_matrix(vocab, w2v)
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))
    print('Building model')

    model = Sequential()
    model.add(Embedding(
        input_dim=len(vocab)+2,
        output_dim=config['embedding_dim'],
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        name='embedding_layer'))
    # add hidden layer with activation, L2 regularization, and dropout
    model.add(LSTM(32, activation=sys.argv[2],
                   kernel_regularizer=l2(0.0001),
                   dropout=0.1,
                   return_sequences=False, name='hidden_layer'))
    # last layer with activation
    model.add(Dense(2, activation='softmax', name='output_layer'))
    model.summary()

    print('train the model')
    # train the model
    # convert words to indices, put them in arrays
    num_classes = 2
    x_train = vectorizer(np.array([[w] for w in x_train])).numpy()
    x_val = vectorizer(np.array([[w] for w in x_val])).numpy()
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    # convert labels to binary class
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=config['batch_size'], epochs=12,
              validation_data=(x_val, y_val))

    model.save(data_dir + 'nn_' + sys.argv[2] + '.model')

    score = model.evaluate(x_val, y_val)
    print("Accuracy: {0: .2f}%".format((score[1]*100)))


if __name__ == '__main__':
    main(sys.argv[1])
