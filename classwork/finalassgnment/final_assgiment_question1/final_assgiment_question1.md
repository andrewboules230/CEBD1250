
![Alt Text](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png)

![Alt Text](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)

#### final assgiment (question 1)

first we will upload the data 'cancer_data.csv' to the notebook

then,we will solve this assignment we will be using scikit-learn to identify client segemnts in a database of customers from a wholesale retailer

In cancer_data.csv you will find the following variables,that we will use for clustering.

radius (mean of distances from center to points on the perimeter)

texture (standard deviation of gray-scale values)

perimeter

area

smoothness (local variation in radius lengths)

compactness (perimeter^2 / area - 1.0)

concavity (severity of concave portions of the contour)

concave_points (number of concave portions of the contour)

symmetry

fractal_dimension ("coastline approximation" - 1)

cancer (0 = Benign, 1 = Malignant) target



then, we need to load the appropriate libraries for cleaning the data and then running a kMeans clustering algorithm. Again, we also set a random seed for replication.


```python
from sklearn import cluster
import pandas as pd
import numpy as np
np.random.seed(100)
```

here we use the read_csv function from pandas to read in our dataset and take a subset of variables used for training.


```python
df = pd.read_csv('cancer_data.csv')
df = df[['radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry','fractal_dimension','cancer']]
```

Next, we use the KMeans function to group the data into two natural clusters,0 for benign and 1 for malignant.We can then print out the assigned labels using .labels_


```python
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(df) 
k_means.labels_
```




    array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
           1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
           1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=int32)



Here we take a subset of the observations in ocancer patient database that were assigned to cluster 0 and print out the average bening category of cancer as well as the total. We can inspect the observations in cluster 0 by using: df.loc[i,]


```python
# cluster(0)
i, = np.where(k_means.labels_==0)
print('Mean:')
print(df.loc[i,].mean())

print(df.loc[i,].mean().sum()/11)

# df.loc[i,]
```

    Mean:
    radius                12.600809
    texture               18.577326
    perimeter             81.456629
    area                 499.714607
    smoothness             0.095258
    compactness            0.092777
    concavity              0.064502
    concave_points         0.034593
    symmetry               0.178703
    fractal_dimension      0.063593
    cancer                 0.797753
    dtype: float64
    55.7887771514


 #### cluster1


```python
i, = np.where(k_means.labels_==1)
print('Mean:')
print(df.loc[i,].mean())

print(df.loc[i,].mean().sum()/11)

# df.loc[i,]
```

    Mean:
    radius                 19.620968
    texture                21.847581
    perimeter             129.733871
    area                 1212.258065
    smoothness              0.100318
    compactness             0.145895
    concavity               0.175990
    concave_points          0.100317
    symmetry                0.190089
    fractal_dimension       0.059940
    cancer                  0.016129
    dtype: float64
    125.840832918


#### by comparing the 2 clusters by this prediction

radius: 14

texture: 14

perimeter: 88

area: 566

smoothness: 1

compactness: 0.08

concavity: 0.06

concae points: 0.04

symmetry: 0.18

fractal dimension: 0.05



#### the final conclusion is that it's approximated to cluster = 0
