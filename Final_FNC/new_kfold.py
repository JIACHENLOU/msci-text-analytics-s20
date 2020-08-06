#Adapted from https://github.com/JayanthRR/fake-news-challenge/blob/master/baseline/fnc_kfold.py
import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, agree_features, discuss_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version
import nltk
nltk.download('wordnet')

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_agree = gen_or_load_feats(agree_features, h, b, "features/agree." + name + ".npy")
    X_discuss = gen_or_load_feats(discuss_features, h, b, "features/discuss." + name + ".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_discuss, X_agree, X_refuting, X_overlap]
    return X,y


if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    MAP = {0: 0, 1: 0, 2: 0, 3: 1}

    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        y_train_1 = [MAP[y] for y in y_train]

        clf_1 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=14128, verbose=True)
        clf_1.fit(X_train, y_train_1)

        X_train_2 = [X_train[i] for i in range(len(X_train)) if y_train[i] != 3]
        y_train_2 = [y for y in y_train if y != 3]

        clf_2 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=14128, verbose=True)
        clf_2.fit(X_train_2, y_train_2)


        predictions_1 = [int(a) for a in clf_1.predict(X_test)]

        related_ids = [i for i in range(len(predictions_1)) if predictions_1[i] == 0]

        X_test_2 = [X_test[i] for i in related_ids]
        predictions_2 = [int(a) for a in clf_2.predict(X_test_2)]

        final_predictions = []
        for i in range(len(predictions_1)):
            if i in related_ids:
                prediction = predictions_2[related_ids.index(i)]
                final_predictions.append(prediction)
            else:
                final_predictions.append(3)

        predicted = [LABELS[int(a)] for a in final_predictions]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold_1 = clf_1
            best_fold_2 = clf_2

    #Run on Holdout set and report the final score on the holdout set
    holdout_prediction_1 = [int(a) for a in best_fold_1.predict(X_holdout)]

    related_ids = [i for i in range(len(holdout_prediction_1)) if holdout_prediction_1[i] == 0]

    X_holdout_stage2 = [X_holdout[i] for i in related_ids]
    holdout_prediction_2 = [int(a) for a in best_fold_2.predict(X_holdout_stage2)]

    holdout_predictions = []
    for i in range(len(holdout_prediction_1)):
        if i in related_ids:
            h_prediction = holdout_prediction_2[related_ids.index(i)]
            holdout_predictions.append(h_prediction)
        else:
            holdout_predictions.append(3)

    predicted = [LABELS[int(a)] for a in holdout_predictions]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")
    print("")

    #Run on competition dataset
    competition_prediction_1 = [int(a) for a in best_fold_1.predict(X_competition)]

    related_ids = [i for i in range(len(competition_prediction_1)) if competition_prediction_1[i] == 0]

    X_competition_stage2 = [X_competition[i] for i in related_ids]
    competition_prediction_2 = [int(a) for a in best_fold_2.predict(X_competition_stage2)]

    competition_predictions = []
    for i in range(len(competition_prediction_1)):
        if i in related_ids:
            c_prediction = competition_prediction_2[related_ids.index(i)]
            competition_predictions.append(c_prediction)
        else:
            competition_predictions.append(3)

    predicted = [LABELS[int(a)] for a in competition_predictions]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted)


