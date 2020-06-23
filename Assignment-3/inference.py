import sys
from pprint import pprint
from gensim.models import Word2Vec

def main(word_path):

    w2v = Word2Vec.load(word_path + '/' + 'w2v.model')
    with open(word_path + '/' + 'word.txt') as f:
        word_txt = f.readlines()
        words = [line.strip() for line in word_txt]

    similarity = {}
    for word in words:
        similarity[word] = w2v.wv.similar_by_word(word, topn=20)

    return similarity


if __name__ == '__main__':
    pprint(main(sys.argv[1]))