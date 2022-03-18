import os
import re

import nltk
from nltk.corpus import stopwords


# nltk.download('all')

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
        for filename in filenames:
            if re.search("\.(txt)$", filename):
                cant = cant + 1
                filepath = os.path.join(root, filename)
                textfile = open(filepath, 'r')
                texts.append(textfile.read())
                b = "Leyendo..." + str(cant)
                print(b, end="\r")
                if prev_root != root:
                    print(root, cant)
                    prev_root = root
                    directories.append(root)
                    dircount.append(cant)
                    cant = 0
    dircount.append(cant)

    dircount = dircount[1:]
    dircount[0] = dircount[0] + 1
    print('Directorios leidos:', len(directories))
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:', sum(dircount))

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


if __name__ == "__main__":
    read_data()
    sentence = """At eight o'clock on Thursday morning
    ... Arthur didn't feel very good."""
    tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [w for w in tokens if not w.lower() in stopwords.words('english')]
    print(filtered_sentence)
