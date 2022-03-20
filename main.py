import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora

# Palabras vacías en inglés
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Stemizador en inglés
snow_stemmer = SnowballStemmer(language='english')

# nltk.download('all')


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


def contadorTerminos(textos):
    long_string = ''
    for text in textos.values():
        long_string = long_string + ','.join(text)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    im = wordcloud.to_image()
    im.save('data/imagenes', format="JPEG")


def lecturaTextos():
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
    # print(textos.keys())
    # print(textos['sports'])

def lecturaTextosSinTag():
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

    print(textos.keys())
    # print(textos['sports'])
    print(len(textos))

    return textos


if __name__ == "__main__":
    textos = lecturaTextosSinTag()
    #print(textos.values())
    #contadorTerminos(textos)