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
