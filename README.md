# msci-text-analytics-s20

## Assignment 2 
| Stopwords removed | Text features  | Accuracy (test set) |
| :---         |     :---:           |          ---:       |
| yes          | unigrams            | 0.8064625           |
| yes          | bigrams             | 0.784625            |
| yes          | unigrams+bigrams    | 0.8242375           |
| no           | unigrams            | 0.8082              |
| no           | bigrams             | 0.820425            |
| no           | unigrams+bigrams    | 0.8322625           |

**a.**	Condition with stopwords performs better. Removing stopwords has higher chance to change the sentence’s context. For example, by removing “not” from “I am not happy” will completely change the meaning. The classifier will put this negative sentence into positive class. 

**b.**	Condition with unigrams + bigrams performs better. Unigrams only counts a single word, it ignores the sentence’s context. Bigrams takes the preceding word into consideration, and expands the size of features, may cause more noises (meaningless word unit). Unigrams + bigrams together not only consider the word’s context, but also increase the probability of using a meaningful feature. 



## Assignment 3
The words most similar to “good” are positive, and words most similar to “bad” are negative. The word2vec embeddings help us learn the meaning of the words. Words with similar meaning have similar vector. Therefore, those top 20 words are words that have similar vector scores as the given word (good or bad).


## Assignment 4
|  Model | L2         | Dropout  | ReLu accuracy  |      Sigmoid accuracy   |Tanh accuracy |
|:---:   | :---:      | :---:    |   :---:        |       :---:             |     :---:    |
|   1    | NA         | NA       | 79.29%         |       78.25%            |79.22%        |
|   2    | NA         | 0.1      | 78.70%         |       77.17%            |78.35%        |
|   3    | NA         | 0.3      | 77.06%         |       75.49%            |76.49%        |
|   4    | 0.0001     | NA       | 78.68%         |       77.65%            |79.01%        |
|   5    | 0.0001     | 0.1      | 79.01%         |       76.90%            |78.24%        |

ReLu always has the highest accuracy score, and sigmoid has the lowest. ReLU, sigmoid, and tanh are non-linear functions that help to normalize the output. Sigmoid set output bound between 0 and 1, and always gives a positive value. Tanh is zero-centered. ReLU converts all negatives to 1. 
Adding dropout did not help improve the accuracy score. Perhaps the dataset size is not large enough, shrinking the input size would reduce model’s performance. L2 regularization put a penalty on vectors that have very high weight. It does not increase accuracy because original model was not largely overfitted.


**Model 5 with ReLu Result**

**Prediction:** 	[1 0 0 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 0 0]

**Actual:** 	    [1 0 0 1 1 0 1 0 0 1 1 1 1 1 1 0 1 0 0 1]


