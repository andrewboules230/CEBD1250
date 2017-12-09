
![Alt Text](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
![Alt Text](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png)
# kmeans Clustering with Scikit-Learn Assignment


In this assignment we will be using scikit-learn to identify client segemnts in a database of customers from a wholesale retailer.

FRESH: annual spending (m.u.) on fresh products (Continuous);

MILK: annual spending (m.u.) on milk products (Continuous);

GROCERY: annual spending (m.u.)on grocery products (Continuous);

FROZEN: annual spending (m.u.)on frozen products (Continuous)

DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)

DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);

First, we need to load the appropriate libraries for cleaning the data and then running a kMeans clustering algorithm. Again, we also set a random seed for replication.



```python
from sklearn import cluster
import pandas as pd
import numpy as np
np.random.seed(100)
```

### Here we use the read_csv function from pandas to read in our dataset and take a subset of variables used for training.


```python
df = pd.read_csv('Wholesale customers data.csv')
df = df[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]


```

Next, we use the KMeans function to group the data into three natural clusters. We can then print out the assigned labels using .labels_


```python
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(df) 
k_means.labels_
```




    array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 1, 1, 0, 0, 2, 0, 2,
           2, 2, 0, 2, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 1, 0, 0,
           2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
           0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 1, 0, 0, 2, 0,
           0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
           1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0,
           1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0,
           0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0,
           0, 0, 1, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           2, 0, 0], dtype=int32)




```python
i, = np.where(k_means.labels_==0)
print('Mean:')
print(df.loc[i,].mean())
print('\nTotal:')
print(df.loc[i,].mean().sum()/6)

```

    Mean:
    Fresh               8253.469697
    Milk                3824.603030
    Grocery             5280.454545
    Frozen              2572.660606
    Detergents_Paper    1773.057576
    Delicassen          1137.496970
    dtype: float64
    
    Total:
    3806.95707071



```python
#cluster 1
i, = np.where(k_means.labels_==1)
print('Mean:')
print(df.loc[i,].mean())
print('\nTotal:')
print(df.loc[i,].mean().sum()/6)

```

    Mean:
    Fresh               35941.400000
    Milk                 6044.450000
    Grocery              6288.616667
    Frozen               6713.966667
    Detergents_Paper     1039.666667
    Delicassen           3049.466667
    dtype: float64
    
    Total:
    9846.26111111



```python
#cluster 2
i, = np.where(k_means.labels_==2)
print('Mean:')
print(df.loc[i,].mean())
print('\nTotal:')
print(df.loc[i,].mean().sum()/6)
```

    Mean:
    Fresh                8000.04
    Milk                18511.42
    Grocery             27573.90
    Frozen               1996.68
    Detergents_Paper    12407.36
    Delicassen           2252.02
    dtype: float64
    
    Total:
    11790.2366667


#### It looks like the clusters indicate that there are three customer segments: small (~3,800), medium ( ~9,845 ), and high ( ~11,790) spending groups.
