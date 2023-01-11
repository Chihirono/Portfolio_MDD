# Analysis of purchasing history
This project aims to give a deeper insight in consumption behavior by using purchasing data. 

### Import packages


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import Counter
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
```

### Read the data


```python
csv_url = 'https://raw.githubusercontent.com/Chihirono/Analysis-of-perchase-history-and-Knn/main/purchase%20history.csv'
df = pd.read_csv(csv_url)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
  </tbody>
</table>
</div>



## Analysis of purchasing tendency
### Data preparation
Check the contents of the original data frame


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 550068 entries, 0 to 550067
    Data columns (total 12 columns):
     #   Column                      Non-Null Count   Dtype  
    ---  ------                      --------------   -----  
     0   User_ID                     550068 non-null  int64  
     1   Product_ID                  550068 non-null  object 
     2   Gender                      550068 non-null  object 
     3   Age                         550068 non-null  object 
     4   Occupation                  550068 non-null  int64  
     5   City_Category               550068 non-null  object 
     6   Stay_In_Current_City_Years  550068 non-null  object 
     7   Marital_Status              550068 non-null  int64  
     8   Product_Category_1          550068 non-null  int64  
     9   Product_Category_2          376430 non-null  float64
     10  Product_Category_3          166821 non-null  float64
     11  Purchase                    550068 non-null  int64  
    dtypes: float64(2), int64(5), object(5)
    memory usage: 50.4+ MB



```python
# Check the number of the user recorded in the data by counting the unique rows of user ID
df.groupby('User_ID').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
    <tr>
      <th>User_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000001</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>21</td>
      <td>14</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1000002</th>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>77</td>
      <td>54</td>
      <td>26</td>
      <td>77</td>
    </tr>
    <tr>
      <th>1000003</th>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>23</td>
      <td>13</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1000004</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>12</td>
      <td>9</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1000005</th>
      <td>106</td>
      <td>106</td>
      <td>106</td>
      <td>106</td>
      <td>106</td>
      <td>106</td>
      <td>106</td>
      <td>106</td>
      <td>58</td>
      <td>16</td>
      <td>106</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1006036</th>
      <td>514</td>
      <td>514</td>
      <td>514</td>
      <td>514</td>
      <td>514</td>
      <td>514</td>
      <td>514</td>
      <td>514</td>
      <td>319</td>
      <td>110</td>
      <td>514</td>
    </tr>
    <tr>
      <th>1006037</th>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>122</td>
      <td>74</td>
      <td>33</td>
      <td>122</td>
    </tr>
    <tr>
      <th>1006038</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>9</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1006039</th>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>60</td>
      <td>27</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1006040</th>
      <td>180</td>
      <td>180</td>
      <td>180</td>
      <td>180</td>
      <td>180</td>
      <td>180</td>
      <td>180</td>
      <td>180</td>
      <td>103</td>
      <td>34</td>
      <td>180</td>
    </tr>
  </tbody>
</table>
<p>5891 rows × 11 columns</p>
</div>




```python
# Check the number of products. These products will be columns for independent variables.
df.groupby('Product_ID').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
    <tr>
      <th>Product_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>P00000142</th>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
      <td>1152</td>
    </tr>
    <tr>
      <th>P00000242</th>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
      <td>376</td>
    </tr>
    <tr>
      <th>P00000342</th>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>244</td>
      <td>0</td>
      <td>244</td>
    </tr>
    <tr>
      <th>P00000442</th>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>0</td>
      <td>92</td>
    </tr>
    <tr>
      <th>P00000542</th>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>0</td>
      <td>0</td>
      <td>149</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>P0099442</th>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>0</td>
      <td>200</td>
    </tr>
    <tr>
      <th>P0099642</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>P0099742</th>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
      <td>126</td>
    </tr>
    <tr>
      <th>P0099842</th>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
      <td>102</td>
    </tr>
    <tr>
      <th>P0099942</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>3631 rows × 11 columns</p>
</div>




```python
# Check the distribution of age per gender
user_distribution = df.groupby(['Gender','Age']).count()[['User_ID']]
user_distribution
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>User_ID</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>Age</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">F</th>
      <th>0-17</th>
      <td>5083</td>
    </tr>
    <tr>
      <th>18-25</th>
      <td>24628</td>
    </tr>
    <tr>
      <th>26-35</th>
      <td>50752</td>
    </tr>
    <tr>
      <th>36-45</th>
      <td>27170</td>
    </tr>
    <tr>
      <th>46-50</th>
      <td>13199</td>
    </tr>
    <tr>
      <th>51-55</th>
      <td>9894</td>
    </tr>
    <tr>
      <th>55+</th>
      <td>5083</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">M</th>
      <th>0-17</th>
      <td>10019</td>
    </tr>
    <tr>
      <th>18-25</th>
      <td>75032</td>
    </tr>
    <tr>
      <th>26-35</th>
      <td>168835</td>
    </tr>
    <tr>
      <th>36-45</th>
      <td>82843</td>
    </tr>
    <tr>
      <th>46-50</th>
      <td>32502</td>
    </tr>
    <tr>
      <th>51-55</th>
      <td>28607</td>
    </tr>
    <tr>
      <th>55+</th>
      <td>16421</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(user_distribution.loc['M'].index,user_distribution.loc[(['M']),:]['User_ID']
        ,align="edge",width=0.3, label='M')
plt.bar(user_distribution.loc['F'].index,user_distribution.loc[(['F']),:]['User_ID']
        ,align="edge",width=-0.3, label='F')
plt.title("Distribution of age groups by male and female")

plt.legend(loc=2)
```




    <matplotlib.legend.Legend at 0x7f9e0c5c1640>




    
![png](output_10_1.png)
    


The result indicates that male aged 26-35 is the largest Gender Age group.<br>
Male users are more than female users. <br>

### The product list that has the largest total "Purchase" of each gender and age group


```python
sum_purchase = df.groupby(['Gender','Age','Product_ID']).sum("Purchase")
sum_purchase = sum_purchase.reset_index()
sum_purchase = sum_purchase.sort_values(by='Purchase', ascending=False)
sum_purchase = sum_purchase.drop_duplicates(['Gender','Age'])
sum_purchase = sum_purchase.sort_values(by=['Gender','Age'])
most_poular = sum_purchase.drop(sum_purchase.columns[[3,4,5,6,7,8]], axis=1).reset_index(drop=True)
most_poular.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Product_ID</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>0-17</td>
      <td>P00255842</td>
      <td>332467</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>18-25</td>
      <td>P00110842</td>
      <td>1339792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>26-35</td>
      <td>P00255842</td>
      <td>2647648</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>36-45</td>
      <td>P00255842</td>
      <td>1459796</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>46-50</td>
      <td>P00025442</td>
      <td>600126</td>
    </tr>
  </tbody>
</table>
</div>



# Prediction of gender and age
## Data preparation 1
### Create a data frame to make independent variables and a dependent variable for modelling.


```python
#Create data frame that indicates how many products each user bought
count_product = df.pivot('User_ID','Product_ID','Purchase')
# Fill 0 to missing value
count_product = count_product.fillna(0) 
count_product.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Product_ID</th>
      <th>P00000142</th>
      <th>P00000242</th>
      <th>P00000342</th>
      <th>P00000442</th>
      <th>P00000542</th>
      <th>P00000642</th>
      <th>P00000742</th>
      <th>P00000842</th>
      <th>P00000942</th>
      <th>P00001042</th>
      <th>...</th>
      <th>P0098942</th>
      <th>P0099042</th>
      <th>P0099142</th>
      <th>P0099242</th>
      <th>P0099342</th>
      <th>P0099442</th>
      <th>P0099642</th>
      <th>P0099742</th>
      <th>P0099842</th>
      <th>P0099942</th>
    </tr>
    <tr>
      <th>User_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000001</th>
      <td>13650.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1000002</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1000003</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1000004</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1000005</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3631 columns</p>
</div>




```python
#Create a data frame that does not include the same user ID 
customer_info = df.drop_duplicates(['User_ID'])
customer_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1000003</td>
      <td>P00193542</td>
      <td>M</td>
      <td>26-35</td>
      <td>15</td>
      <td>A</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>15227</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1000004</td>
      <td>P00184942</td>
      <td>M</td>
      <td>46-50</td>
      <td>7</td>
      <td>B</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>8.0</td>
      <td>17.0</td>
      <td>19215</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1000005</td>
      <td>P00274942</td>
      <td>M</td>
      <td>26-35</td>
      <td>20</td>
      <td>A</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7871</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Check which index is not necessary
print(customer_info.columns[[1,4,5,6,7,8,9,10,11]])
```

    Index(['Product_ID', 'Occupation', 'City_Category',
           'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
           'Product_Category_2', 'Product_Category_3', 'Purchase'],
          dtype='object')



```python
purchasing_history = pd.merge((customer_info.drop(customer_info.columns[[1,4,5,6,7,8,9,10,11]], axis=1)),count_product, on='User_ID')
purchasing_history.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>P00000142</th>
      <th>P00000242</th>
      <th>P00000342</th>
      <th>P00000442</th>
      <th>P00000542</th>
      <th>P00000642</th>
      <th>P00000742</th>
      <th>...</th>
      <th>P0098942</th>
      <th>P0099042</th>
      <th>P0099142</th>
      <th>P0099242</th>
      <th>P0099342</th>
      <th>P0099442</th>
      <th>P0099642</th>
      <th>P0099742</th>
      <th>P0099842</th>
      <th>P0099942</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>F</td>
      <td>0-17</td>
      <td>13650.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000002</td>
      <td>M</td>
      <td>55+</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000003</td>
      <td>M</td>
      <td>26-35</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000004</td>
      <td>M</td>
      <td>46-50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000005</td>
      <td>M</td>
      <td>26-35</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3634 columns</p>
</div>




```python
purchasing_history.insert(0,'GenderAge',(purchasing_history['Gender'] + purchasing_history['Age']))
purchasing_history.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GenderAge</th>
      <th>User_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>P00000142</th>
      <th>P00000242</th>
      <th>P00000342</th>
      <th>P00000442</th>
      <th>P00000542</th>
      <th>P00000642</th>
      <th>...</th>
      <th>P0098942</th>
      <th>P0099042</th>
      <th>P0099142</th>
      <th>P0099242</th>
      <th>P0099342</th>
      <th>P0099442</th>
      <th>P0099642</th>
      <th>P0099742</th>
      <th>P0099842</th>
      <th>P0099942</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F0-17</td>
      <td>1000001</td>
      <td>F</td>
      <td>0-17</td>
      <td>13650.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M55+</td>
      <td>1000002</td>
      <td>M</td>
      <td>55+</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M26-35</td>
      <td>1000003</td>
      <td>M</td>
      <td>26-35</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M46-50</td>
      <td>1000004</td>
      <td>M</td>
      <td>46-50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M26-35</td>
      <td>1000005</td>
      <td>M</td>
      <td>26-35</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3635 columns</p>
</div>




```python
# Make independent variable
X = pd.DataFrame(purchasing_history.drop(purchasing_history.columns[[0,1,2,3]], axis=1))

# Make dependent variable
Y = pd.DataFrame(purchasing_history, columns = ["GenderAge","Gender","Age"])
```

### Standardization


```python
# The definition of the function for standardization
def standard_p(p, ddof = 1):
  # Calculation for minimum value 
    mean_p = p.mean()
  # Calculation for maximum value 
    std_p = p.std(ddof = ddof)
  # Standardization
    standard_p = (p - mean_p) / (std_p)
    return standard_p
```


```python
# Apply the function
X = standard_p(X,ddof=1)
```

### Check multicolinearity


```python
threshold = 0.8

feat_corr = set()
corr_matrix = X.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feat_name = corr_matrix.columns[i]
            feat_corr.add(feat_name)

print(len(set(feat_corr)))

X.drop(labels=feat_corr, axis='columns', inplace=True)

print(len(X.columns))
```

    46
    3585


### Principle components analysis


```python
pca = PCA(n_components=0.95)                     # 95% of the variance will remain
X_reduced = pca.fit_transform(X.iloc[:,:-1].values)  # Reduce the demention 
X_reduced_pd = pd.DataFrame(X_reduced)
print(len(X_reduced_pd.columns))
```

    2019



```python
# Separate data for training and check the qulity of the prediction
X_train,X_test,Y_train,Y_test = train_test_split(X_reduced_pd,Y, test_size=0.3, shuffle=True, random_state=3)
print(len(X_train.columns))
print(len(X_test.columns))
```

    2019
    2019


## Model 1
K Nearest Neighbors classifier (Knn) is used for the modelling.
- Independent variable: factors based on products. Values and features are processed by standardization, multicolinearity, and principle components analysis.
- Dependent variables: GenderAge


```python
#Observe the accuracy rate, precision rate, and recall rate when the value of k is changed from 1 to 90
import matplotlib.pyplot as plt

accuracy  = []
precision = []
recall    = []

k_range = range(1,100)

for k in k_range:
    
    # Create a model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Model learning
    knn.fit(X_train,Y_train["GenderAge"])
    
    # Performance Evaluation
    Y_pred = knn.predict(X_test)
    accuracy.append(round(accuracy_score(Y_test["GenderAge"],Y_pred),3))
    precision.append(round(precision_score(Y_test["GenderAge"],Y_pred, average="macro"),3))
    recall.append(round(recall_score(Y_test["GenderAge"],Y_pred, average="macro"),3))
    
# Plotting
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.show()

# Export the result
max_accuracy = max(accuracy)
index        = accuracy.index(max_accuracy)
best_k_range = k_range[index]
print("When k="+str(best_k_range)+", Accuracy score becomes the maximum,"+str(max_accuracy))
model1_accuracy = str(max_accuracy)
```


    
![png](output_30_0.png)
    


    When k=35, Accuracy score becomes the maximum,0.249


## Model2
Random forest classification is used for the modelling.

- Independent variable: Factors based on products. Values and features are not processed because random forest does not need to do standardization. Moreover,it has 3% higher accuracy than with processed independent variables.
- Dependent variables: GenderAge


```python
X_original = pd.DataFrame(purchasing_history.drop(purchasing_history.columns[[0,1,2,3]], axis=1))
# Separate data for training and check the qulity of the prediction
X_train_o,X_test_o,Y_train,Y_test = train_test_split(X_original,Y, test_size=0.3, shuffle=True, random_state=3)
```


```python
# Create a model
model = RandomForestClassifier()

# Make the model learn by providing the data
model.fit(X_train_o, Y_train["GenderAge"])
```




    RandomForestClassifier()




```python
# Make a model predict by providing the test data
test = model.predict(X_test_o)
# Check the accuracy score 
score = accuracy_score(Y_test["GenderAge"], test)
print(f"Accuracy Score：{score * 100}%")
model2_accuracy = str(score.round(3))
```

    Accuracy Score：26.923076923076923%


## Model 3
Assumption: If the classifier can well predict only gender, the data separated based on the gender prediction is used to predict age group. By taking two steps, the classifier can reduce the number of dependent variables to predict each time.

First step: Classification of Gender. This is done by both Knn and random forest classification.

- Independent variable: Factors based on products. Values and features are processed by standardization, multicollinearity, and principle components analysis only for Knn.
- Dependent variables for both: Gender
Second step: separate the result of prediction by gender (m and f).

Third step: Classification of age group per gender. This is done by Knn clasification.

- Independent variable: Factors from the previous result. It is separated by predicted gender. Therefore, the number of data set is reduced, and classification is implemented twice for the gender group.
- Dependent variables: Age


### Knn - Gender clasification


```python
#Observe the accuracy rate, precision rate, and recall rate when the value of k is changed from 1 to 90
import matplotlib.pyplot as plt

accuracy  = []
precision = []
recall    = []

k_range = range(1,100)

for k in k_range:
    
    # Create a model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Model learning
    knn.fit(X_train,Y_train['Gender'])
    
    # Performance Evaluation
    Y_pred = knn.predict(X_test)
    accuracy.append(round(accuracy_score(Y_test['Gender'],Y_pred),3))
    precision.append(round(precision_score(Y_test['Gender'],Y_pred, average="macro"),3))
    recall.append(round(recall_score(Y_test['Gender'],Y_pred, average="macro"),3))
    
# Plotting
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.show()

# Export the result
max_accuracy = max(accuracy)
index        = accuracy.index(max_accuracy)
best_k_range = k_range[index]
print("When k="+str(best_k_range)+", Accuracy score becomes the maximum,"+str(max_accuracy))

```


    
![png](output_37_0.png)
    


    When k=18, Accuracy score becomes the maximum,0.72


### Random forest - Gender clasification


```python
# Create a model
model = RandomForestClassifier()

# Make the model learn by providing the data
model.fit(X_train, Y_train['Gender'])

# Make a model predict by providing the test data
test = model.predict(X_test)

# Check the accuracy score 
score = accuracy_score(Y_test['Gender'], test)
print(f"Accuracy Score：{score * 100}%")
```

    Accuracy Score：69.28733031674209%



```python
print("The maximum accuracy score of Knn is "+str(max_accuracy*100)+"% (When k="+str(best_k_range)+")")
print(f"Accuracy Score of Random forest is {(score * 100).round(1)}%")
```

    The maximum accuracy score of Knn is 72.0% (When k=18)
    Accuracy Score of Random forest is 69.3%


### The reason to choose the result from Knn for the second model
Because Knn can predict more accurately than random forest, the prediction result from Knn is used for the next model. 


```python
# Instance
knn = KNeighborsClassifier(n_neighbors=18)

# Learning 
knn.fit(X_train, Y_train['Gender'])
Y_pred_Gender = knn.predict(X_test)
```


```python
# make a data frame to use for the second modelling. 
Y_pred_Gender=pd.DataFrame(Y_pred)
Y_pred_Gender = Y_pred_Gender.rename(columns={0:"Gender_pred"})
Y_pred_Gender.value_counts()
```




    Gender_pred
    M              1767
    F                 1
    dtype: int64




```python
Y_test['Gender'].value_counts() # Original number of values in Gender
```




    M    1224
    F     544
    Name: Gender, dtype: int64




```python
Y_test.reset_index(drop=True, inplace=True)
Y_pred_Gender.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

purchasing_history_gender = pd.concat([X_test,Y_test, Y_pred_Gender], axis=1)
purchasing_history_gender.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>GenderAge</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Gender_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.854629</td>
      <td>2.908667</td>
      <td>2.182706</td>
      <td>-7.361682</td>
      <td>-3.504150</td>
      <td>0.308527</td>
      <td>3.306749</td>
      <td>5.569913</td>
      <td>3.604187</td>
      <td>0.136688</td>
      <td>...</td>
      <td>1.168037</td>
      <td>-0.799182</td>
      <td>-0.145168</td>
      <td>-0.226525</td>
      <td>0.288373</td>
      <td>-0.725431</td>
      <td>M26-35</td>
      <td>M</td>
      <td>26-35</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.096280</td>
      <td>5.820239</td>
      <td>2.575192</td>
      <td>-2.525558</td>
      <td>0.822316</td>
      <td>1.126972</td>
      <td>2.169727</td>
      <td>-0.556302</td>
      <td>0.672175</td>
      <td>2.310392</td>
      <td>...</td>
      <td>-0.217829</td>
      <td>0.717193</td>
      <td>-0.203026</td>
      <td>0.139541</td>
      <td>0.633803</td>
      <td>0.703770</td>
      <td>M46-50</td>
      <td>M</td>
      <td>46-50</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-8.070015</td>
      <td>-0.712997</td>
      <td>0.044450</td>
      <td>1.254946</td>
      <td>0.693127</td>
      <td>0.394767</td>
      <td>0.564321</td>
      <td>-0.204361</td>
      <td>0.760783</td>
      <td>0.873937</td>
      <td>...</td>
      <td>-0.427908</td>
      <td>0.113035</td>
      <td>-0.218865</td>
      <td>-0.658559</td>
      <td>0.053793</td>
      <td>-0.518427</td>
      <td>M0-17</td>
      <td>M</td>
      <td>0-17</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-7.286044</td>
      <td>0.839662</td>
      <td>0.946132</td>
      <td>0.905755</td>
      <td>1.408409</td>
      <td>0.779473</td>
      <td>1.661693</td>
      <td>-0.302576</td>
      <td>0.813348</td>
      <td>0.815790</td>
      <td>...</td>
      <td>-0.268358</td>
      <td>-0.313503</td>
      <td>-0.053790</td>
      <td>-0.213342</td>
      <td>-0.187012</td>
      <td>0.937507</td>
      <td>F36-45</td>
      <td>F</td>
      <td>36-45</td>
      <td>M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.774329</td>
      <td>-2.577051</td>
      <td>0.021344</td>
      <td>0.638057</td>
      <td>-1.526873</td>
      <td>-0.172317</td>
      <td>-0.981898</td>
      <td>0.610453</td>
      <td>-2.273251</td>
      <td>-0.952488</td>
      <td>...</td>
      <td>0.220458</td>
      <td>-0.244064</td>
      <td>-0.484998</td>
      <td>0.999324</td>
      <td>0.678254</td>
      <td>-0.426697</td>
      <td>M36-45</td>
      <td>M</td>
      <td>36-45</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2023 columns</p>
</div>



### Knn - Age group clasification


```python
# Data preparation for this second model

# Separate data by the result from the first prediction
purchasing_history_male = purchasing_history_gender[purchasing_history_gender['Gender_pred']=='M']
purchasing_history_female = purchasing_history_gender[purchasing_history_gender['Gender_pred']=='F']

# Make independent variable
X_male = pd.DataFrame(purchasing_history_male.drop(purchasing_history_male.columns[[2019,2020,2021,2022]], axis=1))
# Becuase "purchasing_history_female" only has one row, it cannot be used for the model. Therefore, only X_male is created.

# Make dependent variable
Y_Gender_pred = pd.DataFrame(purchasing_history_male, columns = ["GenderAge","Gender","Age",'Gender_pred'])
```


```python
# Standardization
X_male = standard_p(X_male,ddof=1)

# Multicolinearity
threshold = 0.8

feat_corr = set()
corr_matrix = X_male.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feat_name = corr_matrix.columns[i]
            feat_corr.add(feat_name)

print(len(set(feat_corr)))

X_male.drop(labels=feat_corr, axis='columns', inplace=True)

print(len(X_male.columns))
```

    0
    2019



```python
# Principle components analysis
pca_m = PCA(n_components=0.95)                     # Instance
X_reduced_m = pca_m.fit_transform(X_male.iloc[:,:-1].values)  # Reduce dimentions
X_reduced_pd_m = pd.DataFrame(X_reduced_m)
print(len(X_reduced_pd_m.index))
```

    1767



```python
# Separate data 
X_train_m,X_test_m,Y_train_m,Y_test_m = train_test_split(X_reduced_pd_m,Y_Gender_pred, test_size=0.3, shuffle=True, random_state=3)
```


```python
#Observe the accuracy rate, precision rate, and recall rate when the value of k is changed from 1 to 90
import matplotlib.pyplot as plt

accuracy  = []
precision = []
recall    = []

k_range = range(1,100)

for k in k_range:
    
    # Create a model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Model learning
    knn.fit(X_train_m,Y_train_m['Age'])
    
    # Performance Evaluation
    Y_pred_Age = knn.predict(X_test_m)
    accuracy.append(round(accuracy_score(Y_test_m['Age'],Y_pred_Age),3))
    precision.append(round(precision_score(Y_test_m['Age'],Y_pred_Age, average="macro"),3))
    recall.append(round(recall_score(Y_test_m['Age'],Y_pred_Age, average="macro"),3))
    
# Plotting
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.show()

# Export the result
max_accuracy = max(accuracy)
index        = accuracy.index(max_accuracy)
best_k_range = k_range[index]
print("When k="+str(best_k_range)+", Accuracy score becomes the maximum,"+str(max_accuracy))

```


    
![png](output_51_0.png)
    


    When k=34, Accuracy score becomes the maximum,0.335



```python
Y_pred_Age = pd.DataFrame(Y_pred_Age)
Y_pred_Age = Y_pred_Age.rename(columns={0:"Age_pred"})


Y_test_m.reset_index(drop=True, inplace=True)
Y_pred_Age.reset_index(drop=True, inplace=True)
X_test_m.reset_index(drop=True, inplace=True)

#Show how many predictions are successful. The result indicates 118 out of 531
purchasing_history_gender_age = pd.concat([X_test_m,Y_test_m, Y_pred_Age], axis=1)
purchasing_history_gender_age['Accurate_Age']=(purchasing_history_gender_age['Age']==purchasing_history_gender_age['Age_pred'])#&purchasing_history_gender_age['Gender']==purchasing_history_gender_age['Gender_pred'])
purchasing_history_gender_age['Accurate_Gender']=purchasing_history_gender_age['Gender']==purchasing_history_gender_age['Gender_pred']
purchasing_history_gender_age['Accurate'] = purchasing_history_gender_age['Accurate_Age']*purchasing_history_gender_age['Accurate_Gender']
print('Counts of False and True value')
print(purchasing_history_gender_age['Accurate'].value_counts())
print()
print('Ratio of False and True value')
print(purchasing_history_gender_age['Accurate'].value_counts("True"))
model3_accuracy = str(round((118.0/(118.0+413.0)),3))
```

    Counts of False and True value
    False    413
    True     118
    Name: Accurate, dtype: int64
    
    Ratio of False and True value
    False    0.777778
    True     0.222222
    Name: Accurate, dtype: float64


The resutlt indicates 22% of accuracy rate. This is lower than Modelling 1 and Modelling 2

## Model 4
Assumption: Considering the curse of dimensionality, the reduction of independet variables is the most important. This time, products are grouped based on their characters, and groups are used as independent variables to predict gender and age.

Process: Analyze the products based on the number of purchases by gender and age group, and divide products into groups based on analysis. Use the groups as independent variables and apply them to the classifier.

- Independent variable: Factors based on groups of products. Products are categorized based on their character.
- Dependent variables: Gender and Age group
### Data preparation
Do principle components analysis for each product based on Gender and Age group. In each cell,the  sum of amount purchased is shown.



```python
pre_product_character = df.groupby(['Gender','Age','Product_ID']).aggregate('sum')
pre_product_character = pre_product_character.reset_index(level=[0,1])
pre_product_character.insert(0,'GenderAge',(pre_product_character['Gender'] + pre_product_character['Age']))
pre_product_character.reset_index(inplace= True)
pre_product_character = pre_product_character.rename(columns={'index': 'Product_ID'})
pre_product_character
product_character = pre_product_character.pivot('Product_ID','GenderAge','Purchase')
product_character = product_character.fillna(0)
product_character
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>GenderAge</th>
      <th>F0-17</th>
      <th>F18-25</th>
      <th>F26-35</th>
      <th>F36-45</th>
      <th>F46-50</th>
      <th>F51-55</th>
      <th>F55+</th>
      <th>M0-17</th>
      <th>M18-25</th>
      <th>M26-35</th>
      <th>M36-45</th>
      <th>M46-50</th>
      <th>M51-55</th>
      <th>M55+</th>
    </tr>
    <tr>
      <th>Product_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>P00000142</th>
      <td>238737.0</td>
      <td>853506.0</td>
      <td>1410698.0</td>
      <td>871432.0</td>
      <td>311108.0</td>
      <td>157339.0</td>
      <td>72783.0</td>
      <td>355189.0</td>
      <td>1949906.0</td>
      <td>3583444.0</td>
      <td>1786483.0</td>
      <td>587984.0</td>
      <td>462558.0</td>
      <td>196309.0</td>
    </tr>
    <tr>
      <th>P00000242</th>
      <td>94569.0</td>
      <td>217222.0</td>
      <td>251927.0</td>
      <td>265361.0</td>
      <td>65151.0</td>
      <td>46067.0</td>
      <td>12708.0</td>
      <td>102361.0</td>
      <td>669341.0</td>
      <td>1113022.0</td>
      <td>579796.0</td>
      <td>234294.0</td>
      <td>246954.0</td>
      <td>68723.0</td>
    </tr>
    <tr>
      <th>P00000342</th>
      <td>19743.0</td>
      <td>37874.0</td>
      <td>164987.0</td>
      <td>54516.0</td>
      <td>30030.0</td>
      <td>45679.0</td>
      <td>29737.0</td>
      <td>46051.0</td>
      <td>214363.0</td>
      <td>339113.0</td>
      <td>131853.0</td>
      <td>52852.0</td>
      <td>78564.0</td>
      <td>51113.0</td>
    </tr>
    <tr>
      <th>P00000442</th>
      <td>10652.0</td>
      <td>46227.0</td>
      <td>127628.0</td>
      <td>33874.0</td>
      <td>0.0</td>
      <td>10639.0</td>
      <td>7157.0</td>
      <td>0.0</td>
      <td>30681.0</td>
      <td>94568.0</td>
      <td>47716.0</td>
      <td>23095.0</td>
      <td>5291.0</td>
      <td>3645.0</td>
    </tr>
    <tr>
      <th>P00000542</th>
      <td>37138.0</td>
      <td>36001.0</td>
      <td>92113.0</td>
      <td>63738.0</td>
      <td>10658.0</td>
      <td>36950.0</td>
      <td>7142.0</td>
      <td>21634.0</td>
      <td>125361.0</td>
      <td>210296.0</td>
      <td>117126.0</td>
      <td>33136.0</td>
      <td>15919.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>P0099442</th>
      <td>12354.0</td>
      <td>236887.0</td>
      <td>302714.0</td>
      <td>94901.0</td>
      <td>44419.0</td>
      <td>28692.0</td>
      <td>28845.0</td>
      <td>44835.0</td>
      <td>590837.0</td>
      <td>1058740.0</td>
      <td>259209.0</td>
      <td>107046.0</td>
      <td>11971.0</td>
      <td>48933.0</td>
    </tr>
    <tr>
      <th>P0099642</th>
      <td>0.0</td>
      <td>9707.0</td>
      <td>10121.0</td>
      <td>8097.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10056.0</td>
      <td>23818.0</td>
      <td>13873.0</td>
      <td>0.0</td>
      <td>4135.0</td>
      <td>3903.0</td>
    </tr>
    <tr>
      <th>P0099742</th>
      <td>43427.0</td>
      <td>65568.0</td>
      <td>128499.0</td>
      <td>79746.0</td>
      <td>5649.0</td>
      <td>10598.0</td>
      <td>0.0</td>
      <td>76686.0</td>
      <td>155677.0</td>
      <td>212905.0</td>
      <td>128659.0</td>
      <td>84534.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>P0099842</th>
      <td>7190.0</td>
      <td>54196.0</td>
      <td>96218.0</td>
      <td>94408.0</td>
      <td>45628.0</td>
      <td>55897.0</td>
      <td>30294.0</td>
      <td>0.0</td>
      <td>33355.0</td>
      <td>62700.0</td>
      <td>99013.0</td>
      <td>68363.0</td>
      <td>49642.0</td>
      <td>40408.0</td>
    </tr>
    <tr>
      <th>P0099942</th>
      <td>0.0</td>
      <td>5319.0</td>
      <td>19374.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17754.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8837.0</td>
      <td>0.0</td>
      <td>10776.0</td>
      <td>0.0</td>
      <td>5190.0</td>
      <td>10769.0</td>
    </tr>
  </tbody>
</table>
<p>3631 rows × 14 columns</p>
</div>




```python
# Standardization of sum of amount purchased
product_character_s = product_character.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
product_character_s.head()
product_character_s
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>GenderAge</th>
      <th>F18-25</th>
      <th>F26-35</th>
      <th>F36-45</th>
      <th>F46-50</th>
      <th>F51-55</th>
      <th>F55+</th>
      <th>M0-17</th>
      <th>M18-25</th>
      <th>M26-35</th>
      <th>M36-45</th>
      <th>M46-50</th>
      <th>M51-55</th>
      <th>M55+</th>
    </tr>
    <tr>
      <th>Product_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>P00000142</th>
      <td>6.975440</td>
      <td>5.748997</td>
      <td>6.426149</td>
      <td>4.633491</td>
      <td>2.775837</td>
      <td>2.173713</td>
      <td>5.427949</td>
      <td>4.208950</td>
      <td>3.622471</td>
      <td>3.743656</td>
      <td>3.162658</td>
      <td>2.497976</td>
      <td>1.736237</td>
    </tr>
    <tr>
      <th>P00000242</th>
      <td>1.406024</td>
      <td>0.579622</td>
      <td>1.584325</td>
      <td>0.548267</td>
      <td>0.448225</td>
      <td>0.003581</td>
      <td>1.265648</td>
      <td>1.137497</td>
      <td>0.777788</td>
      <td>0.867790</td>
      <td>0.944194</td>
      <td>1.103055</td>
      <td>0.294290</td>
    </tr>
    <tr>
      <th>P00000342</th>
      <td>-0.163816</td>
      <td>0.191776</td>
      <td>-0.100089</td>
      <td>-0.035076</td>
      <td>0.440109</td>
      <td>0.618732</td>
      <td>0.338618</td>
      <td>0.046226</td>
      <td>-0.113366</td>
      <td>-0.199781</td>
      <td>-0.193872</td>
      <td>0.013599</td>
      <td>0.095266</td>
    </tr>
    <tr>
      <th>P00000442</th>
      <td>-0.090702</td>
      <td>0.025114</td>
      <td>-0.264996</td>
      <td>-0.533859</td>
      <td>-0.292865</td>
      <td>-0.196941</td>
      <td>-0.419519</td>
      <td>-0.394338</td>
      <td>-0.394959</td>
      <td>-0.400302</td>
      <td>-0.380518</td>
      <td>-0.460465</td>
      <td>-0.441207</td>
    </tr>
    <tr>
      <th>P00000542</th>
      <td>-0.180210</td>
      <td>-0.133321</td>
      <td>-0.026416</td>
      <td>-0.356835</td>
      <td>0.257514</td>
      <td>-0.197483</td>
      <td>-0.063359</td>
      <td>-0.167247</td>
      <td>-0.261698</td>
      <td>-0.234879</td>
      <td>-0.317538</td>
      <td>-0.391703</td>
      <td>-0.482401</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>P0099442</th>
      <td>1.578152</td>
      <td>0.806187</td>
      <td>0.222541</td>
      <td>0.203918</td>
      <td>0.084771</td>
      <td>0.586510</td>
      <td>0.318599</td>
      <td>0.949204</td>
      <td>0.715282</td>
      <td>0.103744</td>
      <td>0.146051</td>
      <td>-0.417246</td>
      <td>0.070628</td>
    </tr>
    <tr>
      <th>P0099642</th>
      <td>-0.410363</td>
      <td>-0.499094</td>
      <td>-0.470925</td>
      <td>-0.533859</td>
      <td>-0.515414</td>
      <td>-0.455479</td>
      <td>-0.419519</td>
      <td>-0.443807</td>
      <td>-0.476427</td>
      <td>-0.480959</td>
      <td>-0.525378</td>
      <td>-0.467944</td>
      <td>-0.438291</td>
    </tr>
    <tr>
      <th>P0099742</th>
      <td>0.078591</td>
      <td>0.029000</td>
      <td>0.101470</td>
      <td>-0.440032</td>
      <td>-0.293723</td>
      <td>-0.455479</td>
      <td>0.842961</td>
      <td>-0.094533</td>
      <td>-0.258694</td>
      <td>-0.207393</td>
      <td>0.004848</td>
      <td>-0.494697</td>
      <td>-0.482401</td>
    </tr>
    <tr>
      <th>P0099842</th>
      <td>-0.020949</td>
      <td>-0.115009</td>
      <td>0.218603</td>
      <td>0.223999</td>
      <td>0.653851</td>
      <td>0.638853</td>
      <td>-0.419519</td>
      <td>-0.387924</td>
      <td>-0.431655</td>
      <td>-0.278047</td>
      <td>-0.096582</td>
      <td>-0.173521</td>
      <td>-0.025720</td>
    </tr>
    <tr>
      <th>P0099942</th>
      <td>-0.448771</td>
      <td>-0.457816</td>
      <td>-0.535611</td>
      <td>-0.533859</td>
      <td>-0.144032</td>
      <td>-0.455479</td>
      <td>-0.419519</td>
      <td>-0.446731</td>
      <td>-0.503853</td>
      <td>-0.488340</td>
      <td>-0.525378</td>
      <td>-0.461118</td>
      <td>-0.360693</td>
    </tr>
  </tbody>
</table>
<p>3631 rows × 13 columns</p>
</div>




```python
# Do PCA
pca = PCA()
pca.fit(product_character_s)

feature = pca.transform(product_character_s)
product_character_pca=pd.DataFrame(feature, columns=["PC{}".format(x + 1) 
                               for x in range(len(product_character_s.columns))])
product_character_pca
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
      <th>PC11</th>
      <th>PC12</th>
      <th>PC13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.735925</td>
      <td>-3.862583</td>
      <td>3.449364</td>
      <td>-0.753462</td>
      <td>1.133920</td>
      <td>-1.455482</td>
      <td>-0.515337</td>
      <td>-1.341288</td>
      <td>-0.536697</td>
      <td>0.189753</td>
      <td>0.270224</td>
      <td>0.145883</td>
      <td>0.233582</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.056736</td>
      <td>-1.076092</td>
      <td>-0.062035</td>
      <td>-0.101608</td>
      <td>0.369371</td>
      <td>-0.166191</td>
      <td>-0.201808</td>
      <td>-0.553527</td>
      <td>-0.631242</td>
      <td>0.318711</td>
      <td>0.484988</td>
      <td>-0.102655</td>
      <td>0.018105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.227436</td>
      <td>0.334544</td>
      <td>0.347798</td>
      <td>0.570975</td>
      <td>0.036764</td>
      <td>0.227049</td>
      <td>-0.178341</td>
      <td>0.221706</td>
      <td>0.223469</td>
      <td>-0.215876</td>
      <td>0.065418</td>
      <td>-0.143050</td>
      <td>0.045038</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.182833</td>
      <td>-0.071146</td>
      <td>0.384275</td>
      <td>-0.138310</td>
      <td>-0.102558</td>
      <td>0.174032</td>
      <td>-0.098542</td>
      <td>-0.139915</td>
      <td>-0.051598</td>
      <td>-0.134166</td>
      <td>-0.139959</td>
      <td>-0.067644</td>
      <td>0.044596</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.715838</td>
      <td>-0.067830</td>
      <td>0.280153</td>
      <td>0.012641</td>
      <td>0.339878</td>
      <td>0.279735</td>
      <td>-0.341099</td>
      <td>0.073673</td>
      <td>0.042490</td>
      <td>0.090448</td>
      <td>0.037440</td>
      <td>-0.003480</td>
      <td>0.059916</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3626</th>
      <td>1.455748</td>
      <td>-1.012095</td>
      <td>1.139908</td>
      <td>0.016813</td>
      <td>-0.635797</td>
      <td>0.286827</td>
      <td>0.299921</td>
      <td>0.161524</td>
      <td>-0.079012</td>
      <td>0.346034</td>
      <td>-0.338305</td>
      <td>0.094073</td>
      <td>-0.114216</td>
    </tr>
    <tr>
      <th>3627</th>
      <td>-1.703864</td>
      <td>-0.064165</td>
      <td>-0.023250</td>
      <td>0.007535</td>
      <td>-0.029414</td>
      <td>0.016969</td>
      <td>0.024014</td>
      <td>-0.061977</td>
      <td>-0.006415</td>
      <td>-0.002037</td>
      <td>0.039862</td>
      <td>0.025792</td>
      <td>0.001642</td>
    </tr>
    <tr>
      <th>3628</th>
      <td>-0.474778</td>
      <td>-0.871865</td>
      <td>0.154563</td>
      <td>0.450931</td>
      <td>0.441884</td>
      <td>-0.151205</td>
      <td>-0.384356</td>
      <td>-0.391039</td>
      <td>0.015337</td>
      <td>0.007810</td>
      <td>-0.235284</td>
      <td>-0.147127</td>
      <td>0.052266</td>
    </tr>
    <tr>
      <th>3629</th>
      <td>-0.080125</td>
      <td>0.972793</td>
      <td>0.695295</td>
      <td>0.042739</td>
      <td>0.231414</td>
      <td>0.101917</td>
      <td>-0.102883</td>
      <td>0.112818</td>
      <td>-0.069895</td>
      <td>0.207696</td>
      <td>0.062982</td>
      <td>-0.001881</td>
      <td>0.015726</td>
    </tr>
    <tr>
      <th>3630</th>
      <td>-1.607456</td>
      <td>0.091761</td>
      <td>0.005053</td>
      <td>0.029685</td>
      <td>0.180268</td>
      <td>0.251127</td>
      <td>0.041427</td>
      <td>0.047549</td>
      <td>0.056988</td>
      <td>-0.024014</td>
      <td>0.003975</td>
      <td>0.025910</td>
      <td>0.021266</td>
    </tr>
  </tbody>
</table>
<p>3631 rows × 13 columns</p>
</div>




```python
# Use Elbow Method for finding the optimal number of clusters
distortions = []

for i  in range(1,11):                # Calculate when the number of cluster is from 1 to 10
    km = KMeans(n_clusters=i,
                init='k-means++',     # Choose center of cluster based on k-means++
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(product_character_pca)                         # Calculation
    distortions.append(km.inertia_)   # Get km.inertia_ by km.fit

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
```


    
![png](output_58_0.png)
    


Because the value drops at number of cluster 3, 3 is chosen for the number of cluster.


```python
product_character_pca_k = KMeans(n_clusters=3).fit_predict(product_character_pca)
product_character_pca_k
```




    array([2, 0, 1, ..., 1, 1, 1], dtype=int32)




```python
productgroup = pd.DataFrame(product_character_pca_k)
productgroup.value_counts()
```




    1    3122
    0     456
    2      53
    dtype: int64




```python
# Get the ideal order of Product_ID that corresponds to the productgroup. 
# Productgroup is created by the kmeans that allocates the number for the group of products. 
colproduct = pre_product_character.pivot('Product_ID','GenderAge','Purchase')
colproduct = colproduct.fillna(0)
colproduct.reset_index(inplace= True)
colproduct = colproduct.rename(columns={'index': 'Items'})

#Correspondence table of products and categories #product_character
product_category = pd.merge(colproduct.drop(colproduct.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14]],axis=1),productgroup,left_index=True,right_index=True)
product_category = product_category.rename(columns={0: "Category"})
product_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P00000142</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P00000242</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P00000342</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P00000442</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P00000542</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Allocate product category number to original data frame
df_and_category = (pd.merge(df, product_category, on=['Product_ID']))
```


```python
# Add a columns of standardized amount of purchase
df_and_category['Purchase_s']= (df_and_category['Purchase']-df_and_category['Purchase'].mean())/df_and_category['Purchase'].std(ddof=1)
df_and_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
      <th>Category</th>
      <th>Purchase_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
      <td>1</td>
      <td>-0.177973</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000149</td>
      <td>P00069042</td>
      <td>M</td>
      <td>26-35</td>
      <td>1</td>
      <td>B</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10715</td>
      <td>1</td>
      <td>0.288874</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000541</td>
      <td>P00069042</td>
      <td>F</td>
      <td>18-25</td>
      <td>4</td>
      <td>C</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11037</td>
      <td>1</td>
      <td>0.352978</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000698</td>
      <td>P00069042</td>
      <td>M</td>
      <td>18-25</td>
      <td>4</td>
      <td>A</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8009</td>
      <td>1</td>
      <td>-0.249841</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000951</td>
      <td>P00069042</td>
      <td>M</td>
      <td>46-50</td>
      <td>2</td>
      <td>B</td>
      <td>4+</td>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13346</td>
      <td>1</td>
      <td>0.812657</td>
    </tr>
  </tbody>
</table>
</div>




```python
#count_category: The total amount of purchases for each product category is calculated for each User_ID. 
count_category = df_and_category.groupby(['User_ID','Gender','Age','Category']).sum()
count_category = count_category.reset_index()
#count_category_pivot: Data frame for independent variables
count_category_pivot = count_category.pivot('User_ID','Category','Purchase')
count_category_pivot = count_category_pivot.fillna(0)
count_category_pivot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>User_ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000001</th>
      <td>117128.0</td>
      <td>101063.0</td>
      <td>115902.0</td>
    </tr>
    <tr>
      <th>1000002</th>
      <td>472316.0</td>
      <td>173482.0</td>
      <td>164674.0</td>
    </tr>
    <tr>
      <th>1000003</th>
      <td>144049.0</td>
      <td>55069.0</td>
      <td>142517.0</td>
    </tr>
    <tr>
      <th>1000004</th>
      <td>73318.0</td>
      <td>481.0</td>
      <td>132669.0</td>
    </tr>
    <tr>
      <th>1000005</th>
      <td>270018.0</td>
      <td>404808.0</td>
      <td>146175.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# By connectig on User_ID, the order of dependent variable is fixed for the model
purchasing_history_category = pd.merge((customer_info.drop(customer_info.columns[[1,4,5,6,7,8,9,10,11]], axis=1)),count_category_pivot, on='User_ID')
purchasing_history_category.insert(0,'GenderAge',(purchasing_history_category['Gender'] + purchasing_history_category['Age']))
purchasing_history_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GenderAge</th>
      <th>User_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F0-17</td>
      <td>1000001</td>
      <td>F</td>
      <td>0-17</td>
      <td>117128.0</td>
      <td>101063.0</td>
      <td>115902.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M55+</td>
      <td>1000002</td>
      <td>M</td>
      <td>55+</td>
      <td>472316.0</td>
      <td>173482.0</td>
      <td>164674.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M26-35</td>
      <td>1000003</td>
      <td>M</td>
      <td>26-35</td>
      <td>144049.0</td>
      <td>55069.0</td>
      <td>142517.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M46-50</td>
      <td>1000004</td>
      <td>M</td>
      <td>46-50</td>
      <td>73318.0</td>
      <td>481.0</td>
      <td>132669.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M26-35</td>
      <td>1000005</td>
      <td>M</td>
      <td>26-35</td>
      <td>270018.0</td>
      <td>404808.0</td>
      <td>146175.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make independent variable
X_c = pd.DataFrame(purchasing_history_category.drop(purchasing_history_category.columns[[0,1,2,3]], axis=1))

# Make dependent variable
Y_c = purchasing_history_category["GenderAge"]

```


```python
# Separate data for training and check the qulity of the prediction
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_c,Y_c, test_size=0.3, shuffle=True, random_state=3)
```

### Knn as Modelling 4


```python
#Observe the accuracy rate, precision rate, and recall rate when the value of k is changed from 1 to 90
import matplotlib.pyplot as plt

accuracy  = []
precision = []
recall    = []

k_range = range(1,100)

for k in k_range:
    
    # Create a model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Model learning
    knn.fit(X_train2,Y_train2)
    
    # Performance Evaluation
    Y_pred = knn.predict(X_test2)
    accuracy.append(round(accuracy_score(Y_test2,Y_pred),3))
    precision.append(round(precision_score(Y_test2,Y_pred, average="macro"),3))
    recall.append(round(recall_score(Y_test2,Y_pred, average="macro"),3))
    
# Plotting
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.show()

# Export the result
max_accuracy = max(accuracy)
index        = accuracy.index(max_accuracy)
best_k_range = k_range[index]
print("When k="+str(best_k_range)+", Accuracy score becomes the maximum,"+str(max_accuracy))
model4_accuracy = str(max_accuracy)
```


    
![png](output_70_0.png)
    


    When k=87, Accuracy score becomes the maximum,0.231


The accuracy score is 23% maximum. It's lower than Model 2.

## Model 5
Assumption: Considering the curse of dimensionality, the reduction of independent variables is the most important. This time, products are grouped based on their characters, and groups are used as independent variables to predict gender and age. The previous model 4 only had 3 independent variables. This time, categories of products are classified twice based on the first clustering group.

Process: Cluster products twice. The second clustering is done based on the cluster created by the first clustering. The combination of clustering groups is used as independent variables.

- Independent variable: Factors based on groups of products. Products are categorized by clustering (K-means) two times.
- Dependent variables: Gender and Age group
### Data preparation
Do clustering twice to get a category that becomes an independent variable for modelling instead of Product_ID.





```python
X_cluster = X.T

X_cluster.reset_index(inplace= True)
X_cluster = X_cluster.rename(columns={'index': 'Product_ID'})
X_cluster.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>5881</th>
      <th>5882</th>
      <th>5883</th>
      <th>5884</th>
      <th>5885</th>
      <th>5886</th>
      <th>5887</th>
      <th>5888</th>
      <th>5889</th>
      <th>5890</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P00000142</td>
      <td>2.528750</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.860785</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>...</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.893632</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P00000242</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>...</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>3.411828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P00000342</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>...</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P00000442</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>8.437701</td>
      <td>-0.118953</td>
      <td>...</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P00000542</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>...</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5892 columns</p>
</div>




```python
# Use Elbow Method for finding the optimal number of clusters
distortions = []

for i  in range(1,11):                # Calculate when the number of cluster is from 1 to 10 
    km = KMeans(n_clusters=i,
                init='k-means++',     # Choose the center of cluster based on k-means++
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_cluster[X_cluster.columns[X_cluster.columns != 'Product_ID']])  # Calculation
    distortions.append(km.inertia_)   # Get km.inertia_ by km.fit

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
```


    
![png](output_74_0.png)
    



```python
# Count the number of values in each cluster to check if the amount is enough
product_groupA = KMeans(n_clusters=3).fit_predict(X_cluster[X_cluster.columns[X_cluster.columns != 'Product_ID']])
product_groupA =pd.DataFrame(product_groupA)
product_groupA = product_groupA.rename(columns={0:"GroupA"})
product_groupA.value_counts()
```




    GroupA
    2         1635
    1         1255
    0          695
    dtype: int64




```python
# Add the column of clustering number to  the data frame
product_groupA.reset_index(drop=True, inplace=True)
X_cluster.reset_index(drop=True, inplace=True)

X_groupA = pd.concat([X_cluster,product_groupA], axis=1)
X_groupA.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>5882</th>
      <th>5883</th>
      <th>5884</th>
      <th>5885</th>
      <th>5886</th>
      <th>5887</th>
      <th>5888</th>
      <th>5889</th>
      <th>5890</th>
      <th>GroupA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P00000142</td>
      <td>2.528750</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.860785</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>...</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.893632</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P00000242</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>...</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>3.411828</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P00000342</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>...</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P00000442</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>8.437701</td>
      <td>-0.118953</td>
      <td>...</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P00000542</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>...</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5893 columns</p>
</div>




```python
# Separate data frame based on clustering group for the second clustering
X_groupA0 = X_groupA[X_groupA['GroupA']==0]
X_group0 = X_groupA0.drop('GroupA', axis=1)
X_groupA1 = X_groupA[X_groupA['GroupA']==1]
X_group1 = X_groupA1.drop('GroupA', axis=1)
X_groupA2 = X_groupA[X_groupA['GroupA']==2]
X_group2 = X_groupA2.drop('GroupA', axis=1)
```


```python
distortions = []

for i  in range(1,11):                # Calculate when the number of cluster is from 1 to 10 
    km = KMeans(n_clusters=i,
                init='k-means++',     # Choose the center of cluster based on k-means++
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_group0[X_group0.columns[X_group0.columns != 'Product_ID']])  # # Calculation
    distortions.append(km.inertia_)   # Get km.inertia_ by km.fit

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions') #Sum of Squared errors of prediction
plt.show()

# As the data is already clustered in the same group, there is no fundamental change in Sum of Squared errors of prediction
```


    
![png](output_78_0.png)
    



```python
# The second Kmeans for the group 0 from the first clustering.
product_groupB0 = KMeans(n_clusters=3).fit_predict(X_group0[X_group0.columns[X_group0.columns != 'Product_ID']])
product_groupB0 =pd.DataFrame(product_groupB0)
product_groupB0 = product_groupB0.rename(columns={0:"GroupB"})
# Count the number of values in each cluster to check if the amount is enough
print(product_groupB0.value_counts())
# Add the columns of the result of the second clustering. 
product_groupB0.reset_index(drop=True, inplace=True)
X_groupA0.reset_index(drop=True, inplace=True)
product_groupB0 = pd.concat([X_groupA0,product_groupB0], axis=1)
product_groupB0.head()
```

    GroupB
    2         258
    0         221
    1         216
    dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>5883</th>
      <th>5884</th>
      <th>5885</th>
      <th>5886</th>
      <th>5887</th>
      <th>5888</th>
      <th>5889</th>
      <th>5890</th>
      <th>GroupA</th>
      <th>GroupB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P00001442</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>8.785002</td>
      <td>-0.118169</td>
      <td>...</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>-0.118169</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P00003442</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>2.183120</td>
      <td>2.211697</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>...</td>
      <td>2.861349</td>
      <td>2.210744</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>-0.425965</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P00004142</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>...</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>-0.142737</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P00010842</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>...</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>3.290177</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>-0.351298</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P00014942</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>...</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>-0.123926</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5894 columns</p>
</div>




```python
distortions = []

for i  in range(1,11):                # Calculate when the number of cluster is from 1 to 10 
    km = KMeans(n_clusters=i,
                init='k-means++',     # Choose the center of cluster based on k-means++
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_group1[X_group1.columns[X_group1.columns != 'Product_ID']])      # Calculation
    distortions.append(km.inertia_)   # Get km.inertia_ by km.fit

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# As the data is already clustered in the same group, there is no fundamental change in Sum of Squared errors of prediction
```


    
![png](output_80_0.png)
    



```python
# The second Kmeans for the group 0 from the first clustering.
product_groupB1 = KMeans(n_clusters=3).fit_predict(X_group1[X_group1.columns[X_group1.columns != 'Product_ID']])
product_groupB1 =pd.DataFrame(product_groupB1)
product_groupB1 = product_groupB1.rename(columns={0:"GroupB"})
# Count the number of values in each cluster to check if the amount is enough
print(product_groupB1.value_counts())
# Add teh columns of the result of the second clustering. 
product_groupB1.reset_index(drop=True, inplace=True)
X_groupA1.reset_index(drop=True, inplace=True)
product_groupB1 = pd.concat([X_groupA1,product_groupB1], axis=1)
product_groupB1.head()
```

    GroupB
    1         649
    0         454
    2         152
    dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>5883</th>
      <th>5884</th>
      <th>5885</th>
      <th>5886</th>
      <th>5887</th>
      <th>5888</th>
      <th>5889</th>
      <th>5890</th>
      <th>GroupA</th>
      <th>GroupB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P00000142</td>
      <td>2.528750</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.860785</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>...</td>
      <td>-0.480398</td>
      <td>1.893632</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P00000242</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>...</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>3.411828</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P00000342</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>...</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P00000442</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>8.437701</td>
      <td>-0.118953</td>
      <td>...</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P00000542</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>...</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5894 columns</p>
</div>




```python
distortions = []

for i  in range(1,11):                # Calculate when the number of cluster is from 1 to 10 
    km = KMeans(n_clusters=i,
                init='k-means++',     # Choose the center of cluster based on k-means++
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_group2[X_group2.columns[X_group2.columns != 'Product_ID']])  # Calculation
    distortions.append(km.inertia_)   # Get km.inertia_ by km.fit

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# As the data is already clustered in the same group, there is no fundamental change in Sum of Squared errors of prediction
```


    
![png](output_82_0.png)
    



```python
# The second Kmeans for the group 0 from the first clustering.
product_groupB2 = KMeans(n_clusters=3).fit_predict(X_group2[X_group2.columns[X_group2.columns != 'Product_ID']])
product_groupB2 =pd.DataFrame(product_groupB2)
product_groupB2 = product_groupB2.rename(columns={0:"GroupB"})
# Count the number of values in each cluster to check if the amount is enough
print(product_groupB2.value_counts())
# Add teh columns of the result of the second clustering. 
product_groupB2.reset_index(drop=True, inplace=True)
X_groupA2.reset_index(drop=True, inplace=True)
product_groupB2 = pd.concat([X_groupA2,product_groupB2], axis=1)
```

    GroupB
    2         935
    0         516
    1         184
    dtype: int64



```python
product_category2 = pd.concat([product_groupB0,product_groupB1,product_groupB2], axis=0)
product_category2['Category']=((product_category2['GroupA'].astype(str))+(product_category2['GroupB'].astype(str)))

#Sort by product ID
product_category2 = product_category2.sort_values('Product_ID')
product_category2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_ID</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>5884</th>
      <th>5885</th>
      <th>5886</th>
      <th>5887</th>
      <th>5888</th>
      <th>5889</th>
      <th>5890</th>
      <th>GroupA</th>
      <th>GroupB</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P00000142</td>
      <td>2.528750</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.860785</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>...</td>
      <td>1.893632</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P00000242</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>...</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>3.411828</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P00000342</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>...</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P00000442</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>8.437701</td>
      <td>-0.118953</td>
      <td>...</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P00000542</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>...</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5895 columns</p>
</div>




```python
# Add the column of category (Combination of the first and the second clustering groups)
X_category = X

X_category =(X_category.set_axis(product_category2['Category'], axis='columns'))
X_category = X_category.T
X_category["Category"]=X_category.index
X_category.reset_index(drop=True, inplace=True)
    
X_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>5882</th>
      <th>5883</th>
      <th>5884</th>
      <th>5885</th>
      <th>5886</th>
      <th>5887</th>
      <th>5888</th>
      <th>5889</th>
      <th>5890</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.528750</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.860785</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>...</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>1.893632</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>-0.480398</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>5.831151</td>
      <td>...</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>-0.249369</td>
      <td>3.411828</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>...</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>-0.196176</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>8.437701</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>...</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>-0.118953</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>...</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>-0.153351</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5892 columns</p>
</div>




```python
# X_category_sum: The total amount of purchases for each product category
X_category_sum = X_category.groupby('Category').sum()
X_category_sum =X_category_sum.T
X_category_sum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>00</th>
      <th>01</th>
      <th>02</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-36.465039</td>
      <td>-27.437166</td>
      <td>-8.505863</td>
      <td>-92.935734</td>
      <td>-50.072651</td>
      <td>-20.122723</td>
      <td>-20.716411</td>
      <td>-11.309293</td>
      <td>-40.779975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.231138</td>
      <td>-26.048120</td>
      <td>12.688201</td>
      <td>8.105827</td>
      <td>-58.368340</td>
      <td>-20.122723</td>
      <td>-42.034589</td>
      <td>4.916714</td>
      <td>-44.243896</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-33.655337</td>
      <td>-29.391892</td>
      <td>-37.000442</td>
      <td>-51.400363</td>
      <td>-79.297466</td>
      <td>-18.958781</td>
      <td>-62.316718</td>
      <td>-11.309293</td>
      <td>-45.064330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-47.795902</td>
      <td>-12.346645</td>
      <td>-37.000442</td>
      <td>-81.455779</td>
      <td>-109.388991</td>
      <td>-20.122723</td>
      <td>-62.137709</td>
      <td>-11.309293</td>
      <td>-41.697051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-30.864670</td>
      <td>-25.851360</td>
      <td>-37.000442</td>
      <td>-41.433586</td>
      <td>-70.939927</td>
      <td>-10.572201</td>
      <td>260.861088</td>
      <td>10.481910</td>
      <td>2.307636</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create dependent variable of GenderAge
Y1 = pd.DataFrame(purchasing_history, columns = ["Gender","Age"])
Y1['GenderAge']= Y1['Gender']+Y1['Age']
Y1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>GenderAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>0-17</td>
      <td>F0-17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>55+</td>
      <td>M55+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>26-35</td>
      <td>M26-35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>46-50</td>
      <td>M46-50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>26-35</td>
      <td>M26-35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split data for training and testing
X_train_c,X_test_c,Y_train_c,Y_test_c = train_test_split(X_category_sum,Y1, test_size=0.3, shuffle=True, random_state=3)
```


```python
#Observe the accuracy rate, precision rate, and recall rate when the value of k is changed from 1 to 90
import matplotlib.pyplot as plt

accuracy  = []
precision = []
recall    = []

k_range = range(1,100)

for k in k_range:
    
    # Create a model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Model learning
    knn.fit(X_train_c,Y_train_c["GenderAge"])
    
    # Performance Evaluation
    Y_pred = knn.predict(X_test_c)
    accuracy.append(round(accuracy_score(Y_test_c["GenderAge"],Y_pred),3))
    precision.append(round(precision_score(Y_test_c["GenderAge"],Y_pred, average="macro"),3))
    recall.append(round(recall_score(Y_test_c["GenderAge"],Y_pred, average="macro"),3))
    
# Plotting
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.show()

# Export the result
max_accuracy = max(accuracy)
index        = accuracy.index(max_accuracy)
best_k_range = k_range[index]
print("When k="+str(best_k_range)+", Accuracy score becomes the maximum,"+str(max_accuracy))
model5_accuracy = str(max_accuracy)
```


    
![png](output_89_0.png)
    


    When k=32, Accuracy score becomes the maximum,0.257


2% of the accuracy score has been improved from the previous modelling 4, but it is not the highest from the other models.

## Model 6　Multi-Layer Perceptron Classifier
- Independent variable: Factors processed by standardization, multicollinearity and principle components analysis.
- Dependent variables: Gender and Age group
### Data
The same train and test set for modelling 1 and 2 are used.



Hyperparameter tuning by grid search
Check the best combination of hypermarameter by checking all possible combinations


```python
result = []
for hidden_layer_sizes in [10, 100, 1000]:
    for solver in ['sgd', 'adam', 'lbfgs']:
        for activation in ['identity', 'logistic', 'tanh', 'relu']:
            for learning_rate_init in [0.1, 0.01, 0.001]:
                clf = MLPClassifier(max_iter=10000,
                       hidden_layer_sizes=(hidden_layer_sizes,), 
                       activation=activation, solver=solver,
                       learning_rate_init=learning_rate_init)
                clf.fit(X_train, Y_train['GenderAge'].values.ravel())
                score = clf.score(X_train, Y_train['GenderAge'].values.ravel())
                result.append([hidden_layer_sizes, activation,
                       solver, learning_rate_init, score])
```

Make the data frame of the result of grid search. Sort by the highest score with train data. 


```python
results = pd.DataFrame(result)
results = results.rename(columns={0:"Hidden_layer_sizes",1:"Activation",2:"Solver",3:"The initial learning rate",4:"Score"})
results[results['Score']==1.000000]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hidden_layer_sizes</th>
      <th>Activation</th>
      <th>Solver</th>
      <th>The initial learning rate</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>identity</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10</td>
      <td>identity</td>
      <td>adam</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10</td>
      <td>identity</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>10</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>100</td>
      <td>identity</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>100</td>
      <td>logistic</td>
      <td>sgd</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>100</td>
      <td>logistic</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>100</td>
      <td>tanh</td>
      <td>sgd</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>100</td>
      <td>tanh</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>100</td>
      <td>relu</td>
      <td>sgd</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>100</td>
      <td>relu</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>100</td>
      <td>identity</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>100</td>
      <td>logistic</td>
      <td>adam</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>100</td>
      <td>logistic</td>
      <td>adam</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>100</td>
      <td>logistic</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>100</td>
      <td>tanh</td>
      <td>adam</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>100</td>
      <td>tanh</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>100</td>
      <td>relu</td>
      <td>adam</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>100</td>
      <td>relu</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>100</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>100</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>100</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>100</td>
      <td>logistic</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>100</td>
      <td>logistic</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>100</td>
      <td>logistic</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>100</td>
      <td>tanh</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>67</th>
      <td>100</td>
      <td>tanh</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>100</td>
      <td>tanh</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>100</td>
      <td>relu</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>100</td>
      <td>relu</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>100</td>
      <td>relu</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1000</td>
      <td>identity</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1000</td>
      <td>logistic</td>
      <td>sgd</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1000</td>
      <td>logistic</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1000</td>
      <td>tanh</td>
      <td>sgd</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1000</td>
      <td>tanh</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>1000</td>
      <td>relu</td>
      <td>sgd</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>1000</td>
      <td>relu</td>
      <td>sgd</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1000</td>
      <td>identity</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1000</td>
      <td>logistic</td>
      <td>adam</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1000</td>
      <td>logistic</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1000</td>
      <td>tanh</td>
      <td>adam</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1000</td>
      <td>tanh</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1000</td>
      <td>relu</td>
      <td>adam</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1000</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1000</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1000</td>
      <td>identity</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1000</td>
      <td>logistic</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1000</td>
      <td>logistic</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1000</td>
      <td>logistic</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>102</th>
      <td>1000</td>
      <td>tanh</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>103</th>
      <td>1000</td>
      <td>tanh</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>1000</td>
      <td>tanh</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>1000</td>
      <td>relu</td>
      <td>lbfgs</td>
      <td>0.100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>1000</td>
      <td>relu</td>
      <td>lbfgs</td>
      <td>0.010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>1000</td>
      <td>relu</td>
      <td>lbfgs</td>
      <td>0.001</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Try the combination of the hyperparameter suggested by grid search and choose the one that has the highest accuracy score with test data.


```python
# Define Neural Network model
clf = MLPClassifier(hidden_layer_sizes=100, activation='relu',
                    solver='sgd',learning_rate_init=0.100, max_iter=10000)
# Lerning model
clf.fit(X_train, Y_train['GenderAge'].values.ravel())

# Calculate prediction accuracy
print("Accuracy score (train):",clf.score(X_train, Y_train['GenderAge'].values.ravel()))
```

    Accuracy score (train): 1.0



```python
print("Accuracy score (test):", clf.score(X_test, Y_test['GenderAge'].values.ravel()))
model6_accuracy = str(clf.score(X_test, Y_test['GenderAge'].values.ravel()).round(3))
```

    Accuracy score (test): 0.3223981900452489


The accuracy score is the highest.

# Accuracy score of test data of each model
Model 6 has the highest accuracy score.


```python
print ("Model 1 :" +model1_accuracy)
print ("Model 2 :" +model2_accuracy)
print ("Model 3 :" +model3_accuracy)
print ("Model 4 :" +model4_accuracy)
print ("Model 5 :" +model5_accuracy)
print ("Model 6 :" +model6_accuracy)
```

    Model 1 :0.249
    Model 2 :0.269
    Model 3 :0.222
    Model 4 :0.231
    Model 5 :0.257
    Model 6 :0.322

