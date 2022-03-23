import os
import re
import numpy as np
from num2words import num2words
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD

# Palabras vacías en inglés
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Stemizador en inglés
snow_stemmer = SnowballStemmer(language='english')

# nltk.download('all')

# Etiquetas de los textos
tags = ['business', 'entertainment', 'politics', 'sports', 'tech']


# ------------------ Preprocesado ------------------


# Eliminar caracteres + minusculas
def limpiar(documento):
    # Eliminar caracteres especiales, solo nos quedamos con letras y numeros
    sinCaracteresEsp = re.sub(r'[^\'a-zA-Z0-9\s]', '', documento)
    ##Si hay más de dos espacios dejamos 1 espacio
    sinEspacios = re.sub(r'[^\S]{2,}', ' ', sinCaracteresEsp)
    lowerFinalClean = sinEspacios.lower()

    return lowerFinalClean


# Tokenizacion
def tokenization(documentoLimpio):
    # Tokenizamos despues de limpiar
    tokenizado = WhitespaceTokenizer().tokenize(documentoLimpio)

    return tokenizado


# Eliminación de palabras vacías
def deleteStopWords(arrayTokens):
    result = [t for t in arrayTokens if not t in stop_words]

    return result


# Stemización
def stemmer(tokensNoStopWords):
    stem_words = []

    for w in tokensNoStopWords:
        x = snow_stemmer.stem(w)
        stem_words.append(x)

    return stem_words


# ------------------ Gráficos term frequency ------------------


# Sacar imagen de las palabras mas usadas a nivel global
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


# Sacar imagen de las palabras mas usadas a de etiqueta
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


# ------------------ Preparación modelados ------------------


# Método que pasa el diccionario de textos a lista de strings
def diccAListString(textos):
    print('Funcion diccAListString')
    textosStr = []
    for t in textos.values():
        str = " ".join(t)
        textosStr.append(str)
    return textosStr


# Método que consigue la amtriz tfxidf
def matrizTFxIDF(x_counts):
    print('Funcion matrizTFxIDF')
    # print(textosStr)
    tfidf_transformer = TfidfTransformer()
    matriz_tfidf = tfidf_transformer.fit_transform(x_counts)
    print('Tamaño Matriz TfxIDF:')
    print(np.shape(matriz_tfidf))
    return matriz_tfidf


# Método que consigue el x_counts y la lista de términos usados
def contadorTerminos(textosstr):
    print('Funcion contadorTerminos')
    # Crea el vector que cuenta la repetición de términos
    count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
    x_counts = count_vect.fit_transform(textosstr)
    print(x_counts.todense())
    # A partir de él podemos conseguir la lista de todas las palabras usadas
    feature_names = count_vect.get_feature_names_out()
    print(feature_names)
    print('Hay ' + str(len(count_vect.get_feature_names_out())) + ' términos')
    # x_counts será usado en el main para invocar a matrizTFxIDF y feature_names en globalLDA para poder ver las palbaras más usadas por agrupación
    return x_counts, feature_names


# ------------------ LDA ------------------


# Modelo LDA con n agrupaciones
def modelarLDA(n):
    print('Funcion modelarLDA')
    lda = LDA(n_components=n)
    return lda


# Calcula la semejanza con cada agrupación a partir del modelo y la matriz tfxidf
def probaLDA(lda, matriz_tfxidf):
    print('Funcion probalDA')
    lda_array = lda.fit_transform(matriz_tfxidf)
    return lda_array


# Cuenta cuantos textos hay en cada agrupación seleccionando en cada texto el grupo con mayor porcentaje
def recuentoLDA(n, probabilidades):
    print('Funcion recuentoLDA')
    print(probabilidades)
    # Crear n variables
    for i in range(n):
        num = num2words(i)
        globals()[num] = 0
    # Contadores
    for p in probabilidades:
        p_list = p.tolist()
        tmp = max(p_list)
        index = p_list.index(tmp)
        # print(index)
        for i in range(n):
            num = num2words(i)
            if index == i:
                globals()[num] = globals()[num] + 1
    # Print de resultados
    for i in range(n):
        num = num2words(i)
        print("Del grupo " + str(i) + " hay " + str(globals()[num]) + ' textos')


# Consigue las 5 palabras más repetidas de cada agrupación
def palabrasClaveLDA(lda, feature_names):
    print('Funcion palabrasClaveLDA')
    feature_names_list = feature_names.tolist()
    components = [lda.components_[i] for i in range(len(lda.components_))]
    important_words = [
        sorted(feature_names_list, key=lambda x: components[j][feature_names_list.index(x)], reverse=True)[:5] for
        j in range(len(components))]
    print('Palabras clave:')
    for i in important_words:
        print('Topic ' + str(important_words.index(i)) + ': ' + str(i))


# Ejecuta todas las funciones para modelar el LDA y conseguir resultados
def globalLDA(n, matriz_tfxidf):
    print('Funcion globalLDA')
    lda = modelarLDA(n)
    probabilidades = probaLDA(lda, matriz_tfxidf)
    recuentoLDA(n, probabilidades)
    feature_names = contadorTerminos(textosStr)[1]
    palabrasClaveLDA(lda, feature_names)


# ------------------ LSA ------------------


# Modelo LSA con n agrupaciones
def modelarLSA(n):
    print('Funcion modelarLSA')
    lsa = TruncatedSVD(n_components=n, algorithm='randomized', n_iter=10, random_state=42)
    return lsa


# Calcula la semejanza con cada agrupación a partir del modelo y la matriz tfxidf
def probaLSA(lsa, matriz_tfxidf):
    print('Funcion probalSA')
    lsa_array = lsa.fit_transform(matriz_tfxidf)
    return lsa_array


# Cuenta cuantos textos hay en cada agrupación seleccionando en cada texto el grupo con mayor porcentaje
def recuentoLSA(n, probabilidades):
    print('Funcion recuentoLSA')
    print(probabilidades)
    # Crear n variables
    for i in range(n):
        num = num2words(i)
        globals()[num] = 0
    # Contadores
    for p in probabilidades:
        p_list = p.tolist()
        tmp = max(p_list)
        index = p_list.index(tmp)
        # print(index)
        for i in range(n):
            num = num2words(i)
            if index == i:
                globals()[num] = globals()[num] + 1
    # Print de resultados
    for i in range(n):
        num = num2words(i)
        print("Del grupo " + str(i) + " hay " + str(globals()[num]) + ' textos')


# Consigue las 5 palabras más repetidas de cada agrupación
def palabrasClaveLSA(lsa, feature_names):
    print('Funcion palabrasClaveLDA')
    feature_names_list = feature_names.tolist()
    components = [lsa.components_[i] for i in range(len(lsa.components_))]
    important_words = [
        sorted(feature_names_list, key=lambda x: components[j][feature_names_list.index(x)], reverse=True)[:5] for
        j in range(len(components))]
    print('Palabras clave:')
    for i in important_words:
        print('Topic ' + str(important_words.index(i)) + ': ' + str(i))


# Ejecuta todas las funciones para modelar el LDA y conseguir resultados
def globalLSA(n, matriz_tfxidf):
    print('Funcion globalLSA')
    lsa = modelarLSA(n)
    probabilidades = probaLSA(lsa, matriz_tfxidf)
    recuentoLSA(n, probabilidades)
    feature_names = contadorTerminos(textosStr)[1]
    palabrasClaveLDA(lsa, feature_names)


# ------------------ Lecturas ------------------


# Lectura y estructuración del data con etiquetas
def lecturaTextos():
    print('Funcion lecturaTextos')
    dirname = os.path.join(os.getcwd(), 'data/bbc')
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


# Lectura y estructuración del data sin etiquetas
def lecturaTextosSinTag():
    print('Funcion lecturaTextosSinTag')
    dirname = os.path.join(os.getcwd(), 'data/bbc')
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
    x_counts = contadorTerminos(textosStr)[0]
    matriz_tfxidf = matrizTFxIDF(x_counts)
    n = 5
    globalLDA(n, matriz_tfxidf)
    globalLSA(n, matriz_tfxidf)
    print('globalLSA finished')
