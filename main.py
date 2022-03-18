import os
import re

import nltk
from nltk.corpus import stopwords

nltk.download('all')


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
        for filename in filenames:
            if re.search("\.(txt)$", filename):
                cant = cant + 1
                filepath = os.path.join(root, filename)
                texts.append(open(filepath, 'r').read())
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


if __name__ == "__main__":
    read_data()
