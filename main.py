import os
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import io
import shutil
import string
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras import losses


# nltk.download('all')


def tokenize(text: str) -> list:
    tokens = nltk.word_tokenize(text)
    return [w for w in tokens if not w.lower() in stopwords.words('english')]


def read_data():
    dirname = os.path.join(os.getcwd(), 'data/bbc-train')
    textpath = dirname + os.sep

    texts = []
    directories = []
    dircount = []
    prev_root = ''
    cant = 0

    print("leyendo archivos de ", textpath)

    for root, dirnames, filenames in os.walk(textpath):
        print(root)
        for filename in filenames:
            if re.search("\.(txt)$", filename):
                cant = cant + 1
                filepath = os.path.join(root, filename)
                """texts.append({
                    "class":,
                    "text":open(filepath, 'r').read()
                })"""
                b = "Leyendo..." + str(cant)
                print(b, end="\r")
                if prev_root != root:
                    print(root, cant)
                    prev_root = root
                    directories.append(root)
                    dircount.append(cant)
                    cant = 0
    dircount.append(cant)

    pass

    dircount = dircount[1:]
    dircount[0] = dircount[0] + 1
    print('Directorios leidos:', len(directories))
    print("Textos en cada directorio", dircount)
    print('Suma total de textos:', sum(dircount))

    labels = []
    indice = 0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(indice)
        indice = indice + 1
    print("Cantidad etiquetas creadas: ", len(labels))

    topics = []
    indice = 0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice, name[len(name) - 1])
        topics.append(name[len(name) - 1])
        indice = indice + 1

    print("---Tokenizando Textos---")
    tokenize_texts = [tokenize(text) for text in texts]
    print("---Tokenizado Terminado---")


# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def red():
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'data/bbc-train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print("Review", text_batch.numpy()[i])
            print("Label", label_batch.numpy()[i])

    for i in range(0, len(raw_train_ds.class_names)):
        print("Label ", i, " corresponds to ", raw_train_ds.class_names[i])

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'data/bbc-train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'data/bbc-train',
        batch_size=batch_size)

    max_features = 10000
    sequence_length = 250

    vectorize_layer = layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", raw_train_ds.class_names[first_label])
    print("Vectorized review", vectorize_text(first_review, first_label))

    print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
    print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim = 16

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(5)])

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)

    predicted_classes2 = model.predict(test_ds)
    predicted_classes = []
    for predicted_sport in predicted_classes2:
        predicted_classes.append(predicted_sport.tolist().index(max(predicted_sport)))
    predicted_classes = np.array(predicted_classes)

    #print(classification_report(test_ds, predicted_classes, target_names=raw_train_ds.class_names))


if __name__ == "__main__":
    # read_data()
    red()
