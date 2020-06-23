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
