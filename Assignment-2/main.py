import os
import sys
import pickle
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from inference import *


def clf_indicator(mnb_number):
    cv_dict = {}
    cv_dict["mnb_uni"] = (1, 1)
    cv_dict["mnb_bi"] = (2, 2)
    cv_dict["mnb_uni_bi"] = (1, 2)
    cv_dict["mnb_uni_ns"] = (1, 1)
    cv_dict["mnb_bi_ns"] = (2, 2)
    cv_dict["mnb_uni_bi_ns"] = (1, 2)
    return cv_dict[mnb_number]

def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]

    x_train_ns = read_csv(os.path.join(data_dir, 'train_ns.csv'))
    x_val_ns = read_csv(os.path.join(data_dir, 'val_ns.csv'))
    x_test_ns = read_csv(os.path.join(data_dir, 'test_ns.csv'))
    y_train_ns = labels[:len(x_train_ns)]
    y_val_ns = labels[len(x_train_ns): len(x_train_ns)+len(x_val_ns)]
    y_test_ns = labels[-len(x_test_ns):]
    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_ns, x_val_ns, x_test_ns, y_train_ns, y_val_ns, y_test_ns



def train(X_train, Y_train, clf_type):
    print('Calling CountVectorizer')
    count_vect = CountVectorizer(ngram_range=clf_indicator(clf_type))
    X_train_count = count_vect.fit_transform(X_train)
    print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
    print('Training MNB classifier')
    clf = MultinomialNB().fit(X_train_tfidf, Y_train)
    return clf, count_vect, tfidf_transformer


def clf_to_pickle(classifier, pickle_name):
    pickle_out = open(pickle_name+".pkl", "wb")
    pickle.dump(classifier, pickle_out)
    pickle_out.close()
    return pickle_name



def score(X_val, Y_val, clf, count_vect, tfidf_transformer):
    X_val_count = count_vect.transform(X_val)
    X_val_tfidf = tfidf_transformer.transform(X_val_count)
    preds = clf.predict(X_val_tfidf)
    return {
        'accuracy': accuracy_score(Y_val, preds),
        }


def main(data_dir):

    clf_type = sys.argv[2]
    if clf_type[-3:] == "_ns":
        clf_type = clf_type[:-3]

    # load data
    x_train, x_val, x_test, y_train, y_val, y_test,x_train_ns, x_val_ns, x_test_ns, y_train_ns, y_val_ns, y_test_ns = load_data(data_dir)
    # train
    clf, count_vect, tfidf_transformer = train(x_train, y_train, clf_type)
    clf_ns, count_vect_ns, tfidf_transformer_ns = train(x_train_ns, y_train_ns, clf_type)

    """
    convert clf to pickle files
    """
    clf_to_pickle(clf, clf_type)
    clf_to_pickle(clf_ns, clf_type+"_ns")


    scores = {}
    # accuracy scores for val
    print('with stopwords')
    scores['val'] = score(x_val, y_val, clf, count_vect, tfidf_transformer)
    print('without stopwords')
    scores['val_ns'] = score(x_val_ns, y_val_ns, clf_ns, count_vect_ns, tfidf_transformer_ns)
    # accuracy scores for test
    print('with stopwords')
    scores['test'] = score(x_test, y_test, clf, count_vect, tfidf_transformer)
    print('without stopwords')
    scores['test_ns'] = score(x_test_ns, y_test_ns, clf_ns, count_vect_ns, tfidf_transformer_ns)

    """
    run classifier on test data
    """
    testing_result = {}
    # classify both test and test_ns data, and create an array for each data set
    print('with stopwords')
    testing_result['predict'] = prediction(x_test, clf, count_vect, tfidf_transformer)
    print('without stopwords')
    testing_result['predict_ns'] = prediction(x_test_ns, clf_ns, count_vect_ns, tfidf_transformer_ns)


    return scores, testing_result


if __name__ == '__main__':
    pprint(main(sys.argv[1])[0])
    pprint(main(sys.argv[1])[1])
