from gensim import corpora, models, similarities


def getTexts(data):
    # 去除字符中得空格等
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in data]

    # 将只出现一次的单词移除
    from collections import defaultdict
    frequency = defaultdict(int)

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts


def getDictionary(data):
    texts = getTexts(data)
    dictionary = corpora.Dictionary(texts)
    return dictionary


def getCorpus(data):
    texts = getTexts(data)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus


def getTFIDF(data):
    corpus = getCorpus(data)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf
