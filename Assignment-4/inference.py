import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.layers.experimental.preprocessing import TextVectorization
from main import load_data
from config import config


def main(text_path, classifier):
    x_train, _, x_test, _, _, _ = load_data(text_path)
    x_test = x_test[-20:]
    print(x_test)
    model = keras.models.load_model(os.path.join(text_path, classifier))
    print(model.summary())

    vectorizer = TextVectorization(max_tokens=config['max_vocab_size'], output_sequence_length=config['max_seq_len'])
    train_data = tf.data.Dataset.from_tensor_slices(x_train).batch(config['batch_size'])
    vectorizer.adapt(train_data)
    x_test = vectorizer(np.array([[w] for w in x_test])).numpy()
    prediction = model.predict(x_test)
    print(prediction)
    classes = np.argmax(prediction, axis=-1)
    print(classes)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])