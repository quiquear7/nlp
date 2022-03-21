import os
import re

import numpy
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Palabras vacías en inglés
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Stemizador en inglés
snow_stemmer = SnowballStemmer(language='english')

# nltk.download('all')

tags = ['business', 'entertainment', 'politics', 'sports', 'tech']


# Eliminar caracteres + minusculas
def limpiar(documento):
    # Eliminar caracteres especiales, solo nos quedamos con letras y numeros
    sinCaracteresEsp = re.sub(r'[^\'a-zA-Z0-9\s]', '', documento)
    ##Si hay más de dos espacios dejamos 1 espacio
    sinEspacios = re.sub(r'[^\S]{2,}', ' ', sinCaracteresEsp)
    lowerFinalClean = sinEspacios.lower()

    return lowerFinalClean


def tokenization(documentoLimpio):
    # Tokenizamos despues de limpiar
    tokenizado = WhitespaceTokenizer().tokenize(documentoLimpio)

    return tokenizado


def deleteStopWords(arrayTokens):
    result = [t for t in arrayTokens if not t in stop_words]

    return result


def stemmer(tokensNoStopWords):
    stem_words = []

    for w in tokensNoStopWords:
        x = snow_stemmer.stem(w)
        stem_words.append(x)

    return stem_words


def graficoTerminosGlobal(textos):
    print('Funcion graficoTerminosGlobal')
    long_string = ''
    for text in textos.values():
        long_string = long_string + ','.join(text)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    im = wordcloud.to_image()
    im.show()


def graficoTerminosTag(textos):
    print('Funcion graficoTerminosTag')
    for tag in tags:
        print(tag)
        long_string = ''
        for text in textos.keys():
            if text.startswith(tag):
                long_string = long_string + ','.join(textos[text])
        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        # Generate a word cloud
        wordcloud.generate(long_string)
        # Visualize the word cloud
        im = wordcloud.to_image()
        im.show()


def diccAListString(textos):
    print('Funcion diccAListString')
    textosStr = []
    for t in textos.values():
        str = " ".join(t)
        textosStr.append(str)
    return textosStr


def matrizTFxIDF(x_counts):
    print('Funcion matrizTFxIDF')
    # print(textosStr)
    tfidf_transformer = TfidfTransformer()
    matriz_tfidf = tfidf_transformer.fit_transform(x_counts)
    print(numpy.shape(matriz_tfidf))
    return matriz_tfidf


def contadorTerminos(textosstr):
    print('Funcion contadorTerminos')
    count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
    x_counts = count_vect.fit_transform(textosstr)
    print(x_counts.todense())
    feature_names = count_vect.get_feature_names_out()
    print(feature_names)
    print('Hay ' + str(len(count_vect.get_feature_names_out())) + ' términos')
    return x_counts, feature_names


def modelarLDA(n):
    print('Funcion modelar')
    lda = LDA(n_components=n)
    return lda


def probaLDA(lda, matriz_tfxidf):
    print('Funcion probalDA')
    lda_array = lda.fit_transform(matriz_tfxidf)
    return lda_array


def recuentoLDA(probabilidades):
    print('Funcion recuento')
    cero = 0
    uno = 0
    dos = 0
    tres = 0
    igual = 0
    print(probabilidades)
    for p in probabilidades:
        p_list = p.tolist()
        tmp = max(p_list)
        index = p_list.index(tmp)
        print(index)
        if index == 0:
            cero = cero + 1
        elif index == 1:
            uno = uno + 1
        elif index == 2:
            dos = dos + 1
        elif index == 3:
            tres = tres + 1
        else:
            igual = igual + 1

    print("Del grupo 0 hay " + str(cero) + ' textos,\n del grupo 1 hay ' +
            str(uno) + ' textos,\n del grupo 2 hay ' + str(dos) + ' textos,\n del grupo 3 hay ' +
            str(tres) + ' textos\n y iguales ' + str(igual))


def palabrasClaveLDA(lda, feature_names):
    print('Funcion palabrasClaveLDA')
    feature_names_list = feature_names.tolist()
    components = [lda.components_[i] for i in range(len(lda.components_))]
    important_words = [sorted(feature_names_list, key=lambda x: components[j][feature_names_list.index(x)], reverse=True)[:3] for
                       j in range(len(components))]
    print(important_words)


def globalLDA(n, matriz_tfxidf):
    print('Funcion globalLDA')
    lda = modelarLDA(n)
    probabilidades = probaLDA(lda, matriz_tfxidf)
    recuentoLDA(probabilidades)
    feature_names = contadorTerminos(textosStr)[1]
    palabrasClaveLDA(lda, feature_names)


def lecturaTextos():
    print('Funcion lecturaTextos')
    dirname = os.path.join(os.getcwd(), 'data/bbc-train')
    textpath = dirname + os.sep
    textos = {}

    cant = 0

    # Acceso a carpetas
    for root, dirnames, filenames in os.walk(textpath):
        last_root = os.path.basename(os.path.normpath(root))
        print("Lectura de textos de " + last_root)
        # Se crea un diccionario con clave 'tipo de texto' y tendrá como valor otro diccionario
        globals()[last_root] = {}

        # Acceso a archivos dentro de una carpeta
        for filename in filenames:
            if re.search("\.(txt)$", filename):
                # Nombre + etiqueta
                filename_etiq = last_root + '-' + filename
                print('Leyendo el archivo ' + filename_etiq)
                # Contador
                cant = cant + 1
                # Lectura archivo
                filepath = os.path.join(root, filename)
                f = open(filepath, 'r')
                doc = f.read()
                # Procesado de datos
                limpio = limpiar(doc)
                tokenizado = tokenization(limpio)
                noStopWords = deleteStopWords(tokenizado)
                stemizado = stemmer(noStopWords)
                # Entrada de diciconario en su etiqueta
                globals()[last_root][filename_etiq] = stemizado

        # print(globals()[last_root].keys())
        textos[last_root] = globals()[last_root]


def lecturaTextosSinTag():
    print('Funcion lecturaTextosSinTag')
    dirname = os.path.join(os.getcwd(), 'data/bbc-train')
    textpath = dirname + os.sep
    textos = {}

    cant = 0

    # Acceso a carpetas
    for root, dirnames, filenames in os.walk(textpath):
        last_root = os.path.basename(os.path.normpath(root))
        print("Lectura de textos de " + last_root)

        # Acceso a archivos dentro de una carpeta
        for filename in filenames:
            if re.search("\.(txt)$", filename):
                # Nombre + etiqueta
                filename_etiq = last_root + '-' + filename
                # print('Leyendo el archivo ' + filename_etiq)
                # Contador
                cant = cant + 1
                # Lectura archivo
                filepath = os.path.join(root, filename)
                f = open(filepath, 'r')
                doc = f.read()
                # Procesado de datos
                limpio = limpiar(doc)
                tokenizado = tokenization(limpio)
                noStopWords = deleteStopWords(tokenizado)
                stemizado = stemmer(noStopWords)
                # Entrada de diciconario en su etiqueta
                textos[filename_etiq] = stemizado

    # print(textos.keys())
    # print(textos['sports'])
    print('Hay ' + str(len(textos)) + ' archivos txt')

    return textos


if __name__ == "__main__":
    textos = lecturaTextosSinTag()
    # print(textos.values())
    # contadorTerminosGlobal(textos)
    # contadorTerminosTag(textos)
    textosStr = diccAListString(textos)
    # print(matriz_tfxidf)
    x_counts = contadorTerminos(textosStr)[0]
    matriz_tfxidf = matrizTFxIDF(x_counts)
    globalLDA(4, matriz_tfxidf)
    print('globalLDA finished')