import nltk
nltk.download('all')

if __name__ == "__main__":
    sentence = """At eight o'clock on Thursday morning
    ... Arthur didn't feel very good."""
    tokens = nltk.word_tokenize(sentence)
    print(tokens)
