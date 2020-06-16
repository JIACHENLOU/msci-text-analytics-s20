def prediction(X_test, clf, count_vect, tfidf_transformer):
    X_test_count = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_count)
    predict = clf.predict(X_test_tfidf)
    return predict

