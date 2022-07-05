Test
====

.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)
    train_data.head()




.. raw:: html

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
          <th>age</th>
          <th>workclass</th>
          <th>fnlwgt</th>
          <th>education</th>
          <th>education-num</th>
          <th>marital-status</th>
          <th>occupation</th>
          <th>relationship</th>
          <th>race</th>
          <th>sex</th>
          <th>capital-gain</th>
          <th>capital-loss</th>
          <th>hours-per-week</th>
          <th>native-country</th>
          <th>class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>6118</th>
          <td>51</td>
          <td>Private</td>
          <td>39264</td>
          <td>Some-college</td>
          <td>10</td>
          <td>Married-civ-spouse</td>
          <td>Exec-managerial</td>
          <td>Wife</td>
          <td>White</td>
          <td>Female</td>
          <td>0</td>
          <td>0</td>
          <td>40</td>
          <td>United-States</td>
          <td>&gt;50K</td>
        </tr>
        <tr>
          <th>23204</th>
          <td>58</td>
          <td>Private</td>
          <td>51662</td>
          <td>10th</td>
          <td>6</td>
          <td>Married-civ-spouse</td>
          <td>Other-service</td>
          <td>Wife</td>
          <td>White</td>
          <td>Female</td>
          <td>0</td>
          <td>0</td>
          <td>8</td>
          <td>United-States</td>
          <td>&lt;=50K</td>
        </tr>
        <tr>
          <th>29590</th>
          <td>40</td>
          <td>Private</td>
          <td>326310</td>
          <td>Some-college</td>
          <td>10</td>
          <td>Married-civ-spouse</td>
          <td>Craft-repair</td>
          <td>Husband</td>
          <td>White</td>
          <td>Male</td>
          <td>0</td>
          <td>0</td>
          <td>44</td>
          <td>United-States</td>
          <td>&lt;=50K</td>
        </tr>
        <tr>
          <th>18116</th>
          <td>37</td>
          <td>Private</td>
          <td>222450</td>
          <td>HS-grad</td>
          <td>9</td>
          <td>Never-married</td>
          <td>Sales</td>
          <td>Not-in-family</td>
          <td>White</td>
          <td>Male</td>
          <td>0</td>
          <td>2339</td>
          <td>40</td>
          <td>El-Salvador</td>
          <td>&lt;=50K</td>
        </tr>
        <tr>
          <th>33964</th>
          <td>62</td>
          <td>Private</td>
          <td>109190</td>
          <td>Bachelors</td>
          <td>13</td>
          <td>Married-civ-spouse</td>
          <td>Exec-managerial</td>
          <td>Husband</td>
          <td>White</td>
          <td>Male</td>
          <td>15024</td>
          <td>0</td>
          <td>40</td>
          <td>United-States</td>
          <td>&gt;50K</td>
        </tr>
      </tbody>
    </table>
    </div>



