
![Alt Text](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)

![Alt Text](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png)

#### question 2

first : by copying the lines of code it shows us that it has a many kinds of errors


```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import pipeline
from sklearn_linear_model import SGDClassifier

categories = [ 'rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='test', categories=categories)
twenty_test = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42)),
])
text_clf.fit(twenty_train.data, twenty_test.target)  
predicted = text_clf.predict(twenty_test.data)
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-1-9d8664377183> in <module>()
          1 from sklearn.datasets import fetch_20newsgroups
          2 from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    ----> 3 from sklearn.pipeline import pipeline
          4 from sklearn_linear_model import SGDClassifier
          5 


    ImportError: cannot import name 'pipeline'


#### the first error is that the (p) in pipelines should be in capital form


```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn_linear_model import SGDClassifier

categories = [ 'rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='test', categories=categories)
twenty_test = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42)),
])
text_clf.fit(twenty_train.data, twenty_test.target)  
predicted = text_clf.predict(twenty_test.data)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-2-c68106d0301a> in <module>()
          2 from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
          3 from sklearn.pipeline import Pipeline
    ----> 4 from sklearn_linear_model import SGDClassifier
          5 
          6 categories = [ 'rec.sport.baseball','rec.sport.hockey']


    ModuleNotFoundError: No module named 'sklearn_linear_model'


#### 2- then we have to include that we imported the correct libaries

-import numpy as np -from sklearn import datasets, linear_model


```python
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn_linear_model import SGDClassifier

categories = [ 'rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='test', categories=categories)
twenty_test = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42)),
])
text_clf.fit(twenty_train.data, twenty_test.target)  
predicted = text_clf.predict(twenty_test.data)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-3-b6f0427c642d> in <module>()
          4 from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
          5 from sklearn.pipeline import Pipeline
    ----> 6 from sklearn_linear_model import SGDClassifier
          7 
          8 categories = [ 'rec.sport.baseball','rec.sport.hockey']


    ModuleNotFoundError: No module named 'sklearn_linear_model'


#### 3- (correcting sklearn_linear_model ) to (sklearn.linear_model )


```python
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

categories = ['rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='test', categories=categories)
twenty_test = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42)),
])
text_clf.fit(twenty_train.data, twenty_test.target)  
predicted = text_clf.predict(twenty_test.data)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-5-2f25475c1a04> in <module>()
         15                                            alpha=1e-3, random_state=42)),
         16 ])
    ---> 17 text_clf.fit(twenty_train.data, twenty_test.target)
         18 predicted = text_clf.predict(twenty_test.data)


    /opt/conda/lib/python3.6/site-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
        268         Xt, fit_params = self._fit(X, y, **fit_params)
        269         if self._final_estimator is not None:
    --> 270             self._final_estimator.fit(Xt, y, **fit_params)
        271         return self
        272 


    /opt/conda/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py in fit(self, X, y, coef_init, intercept_init, sample_weight)
        543                          loss=self.loss, learning_rate=self.learning_rate,
        544                          coef_init=coef_init, intercept_init=intercept_init,
    --> 545                          sample_weight=sample_weight)
        546 
        547 


    /opt/conda/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py in _fit(self, X, y, alpha, C, loss, learning_rate, coef_init, intercept_init, sample_weight)
        387             self.classes_ = None
        388 
    --> 389         X, y = check_X_y(X, y, 'csr', dtype=np.float64, order="C")
        390         n_samples, n_features = X.shape
        391 


    /opt/conda/lib/python3.6/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        529         y = y.astype(np.float64)
        530 
    --> 531     check_consistent_length(X, y)
        532 
        533     return X, y


    /opt/conda/lib/python3.6/site-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
        179     if len(uniques) > 1:
        180         raise ValueError("Found input variables with inconsistent numbers of"
    --> 181                          " samples: %r" % [int(l) for l in lengths])
        182 
        183 


    ValueError: Found input variables with inconsistent numbers of samples: [796, 1197]


#### 4-there is an error also in lines 8 and 9 we have to change test by train and train by test to be a readable code


```python
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

categories = ['rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='train', categories=categories)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42)),])
text_clf.fit(twenty_train.data, twenty_test.target)  
predicted = text_clf.predict(twenty_test.data)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-6-69922e4f3771> in <module>()
         14                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
         15                                            alpha=1e-3, random_state=42)),])
    ---> 16 text_clf.fit(twenty_train.data, twenty_test.target)
         17 predicted = text_clf.predict(twenty_test.data)


    /opt/conda/lib/python3.6/site-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
        268         Xt, fit_params = self._fit(X, y, **fit_params)
        269         if self._final_estimator is not None:
    --> 270             self._final_estimator.fit(Xt, y, **fit_params)
        271         return self
        272 


    /opt/conda/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py in fit(self, X, y, coef_init, intercept_init, sample_weight)
        543                          loss=self.loss, learning_rate=self.learning_rate,
        544                          coef_init=coef_init, intercept_init=intercept_init,
    --> 545                          sample_weight=sample_weight)
        546 
        547 


    /opt/conda/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py in _fit(self, X, y, alpha, C, loss, learning_rate, coef_init, intercept_init, sample_weight)
        387             self.classes_ = None
        388 
    --> 389         X, y = check_X_y(X, y, 'csr', dtype=np.float64, order="C")
        390         n_samples, n_features = X.shape
        391 


    /opt/conda/lib/python3.6/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        529         y = y.astype(np.float64)
        530 
    --> 531     check_consistent_length(X, y)
        532 
        533     return X, y


    /opt/conda/lib/python3.6/site-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
        179     if len(uniques) > 1:
        180         raise ValueError("Found input variables with inconsistent numbers of"
    --> 181                          " samples: %r" % [int(l) for l in lengths])
        182 
        183 


    ValueError: Found input variables with inconsistent numbers of samples: [1197, 796]


#### 5- the last two errors are in the extra comma after the brackets in the 12th line.
#### and in the 13 you have to change the (test) with (train)




```python
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

categories = ['rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='test', categories=categories)
twenty_test = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42))])
text_clf.fit(twenty_train.data, twenty_train.target)  
predicted = text_clf.predict(twenty_test.data)

```

#### - now it can be read

#### How many observations are in the training dataset? , How many features are in the training dataset?


```python
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```




    (796, 15074)



#### here are 796 observations and 15074 features in the dataset

#### How well did your model perform?


```python
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, target_names = twenty_test.target_names))
```

                        precision    recall  f1-score   support
    
    rec.sport.baseball       0.95      0.98      0.97       597
      rec.sport.hockey       0.98      0.95      0.97       600
    
           avg / total       0.97      0.97      0.97      1197
    


#### the model is preforming well with the precision of 0.97
