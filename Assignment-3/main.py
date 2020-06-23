import sys
from gensim.models import Word2Vec

def main(data_path):

    with open(data_path + '/' + 'out.csv') as f:
        data = f.readlines()
    all_lines = [' '.join(line.strip().split(',')) for line in data]

    print('Splitting lines in the dataset')
    all_lines = [line.strip().split() for line in all_lines]
    print('Training word2vec model')
    w2v = Word2Vec(all_lines, size=100, window=5, min_count=1, workers=4)
    w2v.save(data_path + '/' + 'w2v.model')
    return

if __name__ == '__main__':
    main(sys.argv[1])
