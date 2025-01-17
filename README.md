
## Final Project Submission

Please fill out:
* Student name: Devin Belden
* Student pace: full time
* Scheduled project review date/time: TBD
* Instructor name: James Irving, Ph.D
* Blog post URL: TBD

# Business Case

For this project, we attempt to answer four questions concerning the Northwind database, trying to find sources of statistically significant impacts on sales. By finding such impacts, we can focus our business efforts on a few key areas that will lead to growth, and potentially learn which facets of the business on which we should not focus, thereby applying a two-pronged approach to resource management.

To answer all of the following questions, we'll first need to import relevant packages, as well as initialize a SQLite cursor.


```python
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
conn = sqlite3.connect('Northwind_small.sqlite')
cur = conn.cursor()
```

## Relationship 1: Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount?

### Null Hypothesis ($H_0$): There is no statistically significant effect of any discount level on the quantity of a product in an order.
### Alternative Hypothesis ($H_1$): There is a statistically significant effect of discount level on quantity of a product in an order.

The phrasing of the question dictates a two-tailed test, as it leaves room for a discount level to *decrease* the quantity of product in an order. There may seem to be no logical reason to order less of a product if a discount is offered, but we'll make room for this circumstance nonetheless.

First, we'll execute a SQL query to pull relevant data.


```python
cur.execute("""select discount, quantity, productid
               from orderdetail
               order by 1""")
df = pd.DataFrame(cur.fetchall(), columns=['discount','quantity','productid'])
df
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
      <th>discount</th>
      <th>quantity</th>
      <th>productid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>12</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>10</td>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>5</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>9</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>40</td>
      <td>51</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>0.25</td>
      <td>4</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2151</th>
      <td>0.25</td>
      <td>20</td>
      <td>54</td>
    </tr>
    <tr>
      <th>2152</th>
      <td>0.25</td>
      <td>20</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2153</th>
      <td>0.25</td>
      <td>20</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2154</th>
      <td>0.25</td>
      <td>10</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>2155 rows × 3 columns</p>
</div>



Let's take a look at the quantity of product sold at each discount level.


```python
for i in list(df['discount'].unique()):
    print(i, len(df[df['discount'] == i]))
```

    0.0 1317
    0.01 1
    0.02 2
    0.03 3
    0.04 1
    0.05 185
    0.06 1
    0.1 173
    0.15 157
    0.2 161
    0.25 154
    

Just by visual inspection, the sample sizes for 1-4%, as well as 6%, are rather small for meaningful statistical analysis. Nevertheless, we'll inspect every discount level just to be sure.

Next, we'll visually illustrate the existence of any impact on product quantity in an order based purely on whether a discount was given. Additionally, we'll test for normality in order to determine which test to use to compare the samples.


```python
df['discounted'] = np.where(df['discount']==0.00,0,1)
df.head()
grp0 = df.groupby('discounted').get_group(0)['quantity']
grp1 = df.groupby('discounted').get_group(1)['quantity']

plt.figure(figsize=(8,5))
plt.bar(x='Full Price', height=grp0.mean(), yerr=stats.sem(grp0))

plt.bar(x='Discounted', height=grp1.mean(), yerr=stats.sem(grp1))
plt.ylabel("Average Quantity per Order")
plt.title("The Effect of Discounts on Order Quantity")
```




    Text(0.5, 1.0, 'The Effect of Discounts on Order Quantity')




![png](output_9_1.png)


Looks like there might be some significant difference between order quantities based on whether or not a discount was offered. Let's check the distributions for normality and run the appropriate hypothesis test.


```python
stats.normaltest(grp0), stats.normaltest(grp1)
```




    (NormaltestResult(statistic=544.5770045551502, pvalue=5.579637380545965e-119),
     NormaltestResult(statistic=261.5280122997891, pvalue=1.6214878452828687e-57))



Since the distributions are not normal, we should use a non-parametric test such as Mann-Whitney.


```python
stats.mannwhitneyu(grp0, grp1, alternative='two-sided')
```




    MannwhitneyuResult(statistic=461541.0, pvalue=1.3258763653999732e-10)



Interesting to note that running a two-tailed test as opposed to a single tailed made no difference in the end.

A P-value this low means we've achieved at least some statistical significance. Let's run a Tukey test to see which discount level matters most.


```python
import statsmodels.api as sms
data = df['quantity'].values
labels = df['discount'].values
model = sms.stats.multicomp.pairwise_tukeyhsd(data, labels)
model.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>  <th>p-adj</th>   <th>lower</th>   <th>upper</th>  <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-19.7153</td>   <td>0.9</td>  <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-19.7153</td>   <td>0.9</td>   <td>-62.593</td> <td>23.1625</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-20.0486</td>  <td>0.725</td> <td>-55.0714</td> <td>14.9742</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-20.7153</td>   <td>0.9</td>  <td>-81.3306</td> <td>39.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>   <td>6.2955</td>  <td>0.0011</td>  <td>1.5381</td>  <td>11.053</td>   <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-19.7153</td>   <td>0.9</td>  <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>   <td>3.5217</td>  <td>0.4269</td>  <td>-1.3783</td> <td>8.4217</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>   <td>6.6669</td>  <td>0.0014</td>   <td>1.551</td>  <td>11.7828</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>   <td>5.3096</td>  <td>0.0303</td>  <td>0.2508</td>  <td>10.3684</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>    <td>6.525</td>  <td>0.0023</td>  <td>1.3647</td>  <td>11.6852</td>  <td>True</td> 
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>     <td>0.0</td>     <td>0.9</td>  <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>   <td>-0.3333</td>   <td>0.9</td>  <td>-70.2993</td> <td>69.6326</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>-1.0</td>     <td>0.9</td>  <td>-86.6905</td> <td>84.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>   <td>26.0108</td>   <td>0.9</td>   <td>-34.745</td> <td>86.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>     <td>0.0</td>     <td>0.9</td>  <td>-85.6905</td> <td>85.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>   <td>23.237</td>    <td>0.9</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>   <td>26.3822</td>   <td>0.9</td>  <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>   <td>25.0248</td>   <td>0.9</td>  <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>   <td>26.2403</td>   <td>0.9</td>  <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>   <td>-0.3333</td>   <td>0.9</td>  <td>-55.6463</td> <td>54.9796</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>-1.0</td>     <td>0.9</td>  <td>-75.2101</td> <td>73.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>   <td>26.0108</td> <td>0.6622</td> <td>-17.0654</td> <td>69.087</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>     <td>0.0</td>     <td>0.9</td>  <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>   <td>23.237</td>  <td>0.7914</td> <td>-19.8552</td> <td>66.3292</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>   <td>26.3822</td> <td>0.6461</td> <td>-16.7351</td> <td>69.4994</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>   <td>25.0248</td> <td>0.7089</td> <td>-18.0857</td> <td>68.1354</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>   <td>26.2403</td> <td>0.6528</td> <td>-16.8823</td> <td>69.3628</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>   <td>-0.6667</td>   <td>0.9</td>  <td>-70.6326</td> <td>69.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>   <td>26.3441</td> <td>0.3639</td>  <td>-8.9214</td> <td>61.6096</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>   <td>0.3333</td>    <td>0.9</td>  <td>-69.6326</td> <td>70.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>   <td>23.5703</td> <td>0.5338</td> <td>-11.7147</td> <td>58.8553</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>   <td>26.7155</td> <td>0.3436</td>  <td>-8.6001</td> <td>62.0311</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>   <td>25.3582</td>  <td>0.428</td>  <td>-9.9492</td> <td>60.6656</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>   <td>26.5736</td> <td>0.3525</td>  <td>-8.7485</td> <td>61.8957</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>   <td>27.0108</td>   <td>0.9</td>   <td>-33.745</td> <td>87.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>     <td>1.0</td>     <td>0.9</td>  <td>-84.6905</td> <td>86.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>   <td>24.237</td>    <td>0.9</td>  <td>-36.5302</td> <td>85.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>   <td>27.3822</td>   <td>0.9</td>  <td>-33.4028</td> <td>88.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>   <td>26.0248</td>   <td>0.9</td>  <td>-34.7554</td> <td>86.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>   <td>27.2403</td>   <td>0.9</td>  <td>-33.5485</td> <td>88.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-26.0108</td>   <td>0.9</td>  <td>-86.7667</td> <td>34.745</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>   <td>-2.7738</td>   <td>0.9</td>   <td>-9.1822</td> <td>3.6346</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>   <td>0.3714</td>    <td>0.9</td>   <td>-6.2036</td> <td>6.9463</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>   <td>-0.986</td>    <td>0.9</td>   <td>-7.5166</td> <td>5.5447</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>   <td>0.2294</td>    <td>0.9</td>   <td>-6.3801</td>  <td>6.839</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>   <td>23.237</td>    <td>0.9</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>   <td>26.3822</td>   <td>0.9</td>  <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>   <td>25.0248</td>   <td>0.9</td>  <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>   <td>26.2403</td>   <td>0.9</td>  <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>3.1452</td>    <td>0.9</td>   <td>-3.5337</td>  <td>9.824</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>   <td>1.7879</td>    <td>0.9</td>   <td>-4.8474</td> <td>8.4231</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>   <td>3.0033</td>    <td>0.9</td>   <td>-3.7096</td> <td>9.7161</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>-1.3573</td>   <td>0.9</td>   <td>-8.1536</td> <td>5.4389</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>   <td>-0.1419</td>   <td>0.9</td>   <td>-7.014</td>  <td>6.7302</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>   <td>1.2154</td>    <td>0.9</td>   <td>-5.6143</td> <td>8.0451</td>   <td>False</td>
</tr>
</table>




```python
grp5 = df.groupby('discount').get_group(0.05)['quantity']
grp15 = df.groupby('discount').get_group(0.15)['quantity']
grp20 = df.groupby('discount').get_group(0.2)['quantity']
grp25 = df.groupby('discount').get_group(0.25)['quantity']

plt.figure(figsize=(8,5))
plt.bar(x='5%', height=grp5.mean(), yerr=stats.sem(grp5))
plt.bar(x='15%', height=grp15.mean(), yerr=stats.sem(grp15))
plt.bar(x='20%', height=grp20.mean(), yerr=stats.sem(grp20))
plt.bar(x='25%', height=grp25.mean(), yerr=stats.sem(grp25))

plt.ylabel("Average Quantity per Order")
plt.title("Comparison of Discount Levels")
```




    Text(0.5, 1.0, 'Comparison of Discount Levels')




![png](output_16_1.png)


Looks like discount levels of 5%, 15%, 20%, and 25% all lead to statistically significant increases in amount of product sold in a given order. However, none of these discount levels, when compared to each other, lead to statistically significant increases, as shown in the above graph. In essence, we cannot say that a customer will order more product when given a 25% discount than when given a mere 5% off. We can see this visually as well:


```python
model.plot_simultaneous();
```


![png](output_18_0.png)


As seen above, the confidence intervals for our selected discount levels eclipse each others' means but not the 0% discount level, whereas the confidence interval for, say, a 10% discount level eclipses the mean of the 0% level. Thus, we can reject the null hypothesis and say that the selected discount levels provide a statistically significant increase in product quantity per order.

### Conclusion: Discount levels of 5%, 15%, 20%, and 25% all result in statistically significant (p < 0.05) increases in quantity of product per order.

## Relationship 2: Does the quantity of product sold fluctuate with the time of year? If so, when are more products being sold?

### $H_0$: There is no statistically significant effect of time of year on total quantity of product sold.
### $H_1$: Time of year has a statistically significant impact on total quantity of product sold.

For this test, we'll split the time of year into quarters. As usual, let's start by pulling the relevant data in an SQL query.


```python
cur.execute("""select o.shipcountry, o.orderdate, sum(od.quantity) from 'order' o
               join orderdetail od
               on o.id = od.orderid
               group by 2""")
df = pd.DataFrame(cur.fetchall(), columns=['shipcountry','orderdate','quantity'])
df
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
      <th>shipcountry</th>
      <th>orderdate</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>2012-07-04</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Germany</td>
      <td>2012-07-05</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>2012-07-08</td>
      <td>101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belgium</td>
      <td>2012-07-09</td>
      <td>105</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2012-07-10</td>
      <td>102</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>Italy</td>
      <td>2014-04-30</td>
      <td>151</td>
    </tr>
    <tr>
      <th>476</th>
      <td>USA</td>
      <td>2014-05-01</td>
      <td>277</td>
    </tr>
    <tr>
      <th>477</th>
      <td>Germany</td>
      <td>2014-05-04</td>
      <td>101</td>
    </tr>
    <tr>
      <th>478</th>
      <td>Germany</td>
      <td>2014-05-05</td>
      <td>365</td>
    </tr>
    <tr>
      <th>479</th>
      <td>Denmark</td>
      <td>2014-05-06</td>
      <td>178</td>
    </tr>
  </tbody>
</table>
<p>480 rows × 3 columns</p>
</div>



We'll need to massage this data into an appropriate format, including adding a column for the quarter of the year. 


```python
df['orderdate'] = pd.to_datetime(df['orderdate'], format='%Y/%m/%d')
df['quarter'] = df['orderdate'].dt.to_period("q")
df['quarter'] = df.apply(lambda row: int(str(row['quarter'])[-1]), axis=1)
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
      <th>shipcountry</th>
      <th>orderdate</th>
      <th>quantity</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>2012-07-04</td>
      <td>27</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Germany</td>
      <td>2012-07-05</td>
      <td>49</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>2012-07-08</td>
      <td>101</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belgium</td>
      <td>2012-07-09</td>
      <td>105</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2012-07-10</td>
      <td>102</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
grp0 = df.groupby('quarter').get_group(1)['quantity']
grp1 = df.groupby('quarter').get_group(2)['quantity']
grp2 = df.groupby('quarter').get_group(3)['quantity']
grp3 = df.groupby('quarter').get_group(4)['quantity']

plt.figure(figsize=(8,5))
plt.bar(x='Q1', height=grp0.mean(), yerr=stats.sem(grp0))
plt.bar(x='Q2', height=grp1.mean(), yerr=stats.sem(grp1))
plt.bar(x='Q3', height=grp2.mean(), yerr=stats.sem(grp2))
plt.bar(x='Q4', height=grp3.mean(), yerr=stats.sem(grp3))
plt.title("Quantity Sold by Quarter")
plt.xlabel("Quarter of Year")
plt.ylabel("Average Quantity per Order")
```




    Text(0, 0.5, 'Average Quantity per Order')




![png](output_25_1.png)


Just for fun, we'll test for normality of the distributions.


```python
for grp in [grp0,grp1,grp2,grp3]:
    print(stats.normaltest(grp))
```

    NormaltestResult(statistic=71.88070533047213, pvalue=2.462085666549949e-16)
    NormaltestResult(statistic=63.04152441886395, pvalue=2.0450640156162815e-14)
    NormaltestResult(statistic=48.25959334243682, pvalue=3.31560248332186e-11)
    NormaltestResult(statistic=13.21563374774304, pvalue=0.0013497756657326514)
    

As expected, these distributions are not normally distributed, but since our sample sizes are over 15, we can still use ANOVA.


```python
from scipy.stats import f_oneway

f_oneway(grp0,grp1,grp2,grp3)
```




    F_onewayResult(statistic=10.936050410502693, pvalue=5.879748582008128e-07)



Looks like we've got a low enough P-value to confidently reject the null hypothesis. Once again, we can run Tukey test to see which quarters see a statistically significant change in total quantity of product sold.


```python
data = df['quantity'].values
labels = df['quarter'].values
model = sms.stats.multicomp.pairwise_tukeyhsd(data, labels)
model.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>  <th>p-adj</th>   <th>lower</th>    <th>upper</th>  <th>reject</th>
</tr>
<tr>
     <td>1</td>      <td>2</td>    <td>-8.1064</td> <td>0.8837</td> <td>-37.3175</td>  <td>21.1048</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>3</td>   <td>-53.6079</td>  <td>0.001</td> <td>-80.1857</td>  <td>-27.03</td>   <td>True</td> 
</tr>
<tr>
     <td>1</td>      <td>4</td>   <td>-34.7625</td> <td>0.0042</td> <td>-61.1895</td>  <td>-8.3356</td>  <td>True</td> 
</tr>
<tr>
     <td>2</td>      <td>3</td>   <td>-45.5015</td>  <td>0.001</td> <td>-74.6656</td> <td>-16.3374</td>  <td>True</td> 
</tr>
<tr>
     <td>2</td>      <td>4</td>   <td>-26.6562</td> <td>0.0849</td> <td>-55.6829</td>  <td>2.3705</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>4</td>    <td>18.8453</td>  <td>0.255</td>  <td>-7.5296</td>  <td>45.2202</td>  <td>False</td>
</tr>
</table>




```python
model.plot_simultaneous();
```


![png](output_32_0.png)


Looks like we can confidently reject the null hypothesis and say that there is a statistically significant difference in total sales depending on the time of year.

### Conclusion: There is a statistically significant (p < 0.05) difference in total sales depending on the time of year.

### Relationship 3: Is there a statistically significant difference in sales between employees throughout the company, depending on their age? If so, which employees are selling more?

#### $H_0$: There is no statistically significant effect of an employee's age on their selling ability.

#### $H_1$: There is a statistically significant effect of an employee's age on their selling ability.

Once again, we'll pull relevant data using SQL.


```python
cur.execute("""select distinct o.id, e.birthdate, od.unitprice, od.quantity
               from 'order' o
               join employee e on o.employeeid = e.id
               join orderdetail od on o.id = od.orderid
               order by 2""")
revenue_by_birthdate = pd.DataFrame(cur.fetchall(), columns=['Order ID','Birthdate','Unit Price','Quantity'])

revenue_by_birthdate['Total Revenue by item'] = revenue_by_birthdate.apply(lambda row: 
                                                                         row['Unit Price']*row['Quantity'], axis=1)
revenue_by_birthdate
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
      <th>Order ID</th>
      <th>Birthdate</th>
      <th>Unit Price</th>
      <th>Quantity</th>
      <th>Total Revenue by item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10250</td>
      <td>1969-09-19</td>
      <td>7.7</td>
      <td>10</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10250</td>
      <td>1969-09-19</td>
      <td>42.4</td>
      <td>35</td>
      <td>1484.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10250</td>
      <td>1969-09-19</td>
      <td>16.8</td>
      <td>15</td>
      <td>252.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10252</td>
      <td>1969-09-19</td>
      <td>64.8</td>
      <td>40</td>
      <td>2592.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10252</td>
      <td>1969-09-19</td>
      <td>2.0</td>
      <td>25</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2149</th>
      <td>11022</td>
      <td>1998-01-27</td>
      <td>9.2</td>
      <td>35</td>
      <td>322.0</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>11022</td>
      <td>1998-01-27</td>
      <td>36.0</td>
      <td>30</td>
      <td>1080.0</td>
    </tr>
    <tr>
      <th>2151</th>
      <td>11058</td>
      <td>1998-01-27</td>
      <td>10.0</td>
      <td>3</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2152</th>
      <td>11058</td>
      <td>1998-01-27</td>
      <td>34.0</td>
      <td>21</td>
      <td>714.0</td>
    </tr>
    <tr>
      <th>2153</th>
      <td>11058</td>
      <td>1998-01-27</td>
      <td>28.5</td>
      <td>4</td>
      <td>114.0</td>
    </tr>
  </tbody>
</table>
<p>2154 rows × 5 columns</p>
</div>



We could have just as easily grouped this by employee ID or any other column descriptive of the employees themselves, but grouping by birthdate (and therefore age) adds another dimension to the statistical inference.


```python
for date in revenue_by_birthdate['Birthdate'].unique():
    revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == date].hist(column='Total Revenue by item')
    plt.title("Birthdate: "+ str(date))
    plt.xlabel('Total Revenue')
    plt.ylabel('Amount Sold')
```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)



![png](output_38_4.png)



![png](output_38_5.png)



![png](output_38_6.png)



![png](output_38_7.png)



![png](output_38_8.png)


Below is a graph comparing the average sales 


```python
grp0 = revenue_by_birthdate.groupby('Birthdate').get_group('1969-09-19')['Total Revenue by item']
grp1 = revenue_by_birthdate.groupby('Birthdate').get_group('1980-12-08')['Total Revenue by item']
grp2 = revenue_by_birthdate.groupby('Birthdate').get_group('1984-02-19')['Total Revenue by item']
grp3 = revenue_by_birthdate.groupby('Birthdate').get_group('1987-03-04')['Total Revenue by item']
grp4 = revenue_by_birthdate.groupby('Birthdate').get_group('1990-01-09')['Total Revenue by item']
grp5 = revenue_by_birthdate.groupby('Birthdate').get_group('1992-05-29')['Total Revenue by item']
grp6 = revenue_by_birthdate.groupby('Birthdate').get_group('1995-07-02')['Total Revenue by item']
grp7 = revenue_by_birthdate.groupby('Birthdate').get_group('1995-08-30')['Total Revenue by item']
grp8 = revenue_by_birthdate.groupby('Birthdate').get_group('1998-01-27')['Total Revenue by item']

plt.figure(figsize=(10,3))
plt.bar(x='1969-09-19', height=grp0.mean(), yerr=stats.sem(grp0))
plt.bar(x='1980-12-08', height=grp1.mean(), yerr=stats.sem(grp1))
plt.bar(x='1984-02-19', height=grp2.mean(), yerr=stats.sem(grp2))
plt.bar(x='1987-03-04', height=grp3.mean(), yerr=stats.sem(grp3))
plt.bar(x='1990-01-09', height=grp4.mean(), yerr=stats.sem(grp4))
plt.bar(x='1992-05-29', height=grp5.mean(), yerr=stats.sem(grp5))
plt.bar(x='1995-07-02', height=grp6.mean(), yerr=stats.sem(grp6))
plt.bar(x='1995-08-30', height=grp7.mean(), yerr=stats.sem(grp7))
plt.bar(x='1998-01-27', height=grp8.mean(), yerr=stats.sem(grp8))

plt.title("Average Revenue of Each Salesperson by Birthdate")
plt.xlabel("Birthdate")
plt.xticks(rotation='45')
plt.ylabel("Average Revenue Sold")
```




    Text(0, 0.5, 'Average Revenue Sold')




![png](output_40_1.png)


Next, we'll break up the data into individual dataframes of each employee's total revenue sold. Then we'll run the non-parametric Kruskal-Wallis H-test to check for similarity between datasets.

(Note: if the code in the following cell looks abysmal, that's because it is.)


```python
a = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1969-09-19']
a = a['Total Revenue by item']
b = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1980-12-08']
b = b['Total Revenue by item']
c = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1984-02-19']
c = c['Total Revenue by item']
d = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1987-03-04']
d = d['Total Revenue by item']
e = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1990-01-09']
e = e['Total Revenue by item']
f = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1992-05-29']
f = f['Total Revenue by item']
g = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1995-07-02']
g = g['Total Revenue by item']
h = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1995-08-30']
h = h['Total Revenue by item']
i = revenue_by_birthdate[revenue_by_birthdate['Birthdate'] == '1998-01-27']
i = i['Total Revenue by item']
```


```python
from scipy.stats import kruskal

kruskal(a,b,c,d,e,f,g,h,i)
```




    KruskalResult(statistic=10.766600688693485, pvalue=0.2152778357561379)



A rather high P-value, which tells us we cannot confidently reject the null hypothesis.

### Conclusion: An employee's birthdate has no statistically significant (p > 0.05) effect on their total revenue sold.

### Relationship 4: Is there a statistically significant difference in average discount level per order between both US and UK locations? If so, which office is giving more discounts?

Since we know that discount level has a statistically significant effect on the quantity in an order, why don't we find out where most of the discounts are coming from?

#### $H_0$: There is no statistically significant difference in average discount amount per order between the US and UK offices.

#### $H_1$: There is a statistically significant difference in average discount amount per order between the US and UK offices.

Since we have no idea which office offers more discounts, it makes sense to make this a two-tailed test, as will become relevant later.


```python
cur.execute("""select distinct o.id, e.country, avg(od.discount)
               from 'order' o
               join employee e on e.id = o.employeeid
               join orderdetail od on od.orderid = o.id
               group by 1""")
discount_by_country = pd.DataFrame(cur.fetchall(), columns=['orderID','country of sales rep','average discount in order'])
discount_by_country
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
      <th>orderID</th>
      <th>country of sales rep</th>
      <th>average discount in order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248</td>
      <td>UK</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10249</td>
      <td>UK</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10250</td>
      <td>USA</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10251</td>
      <td>USA</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10252</td>
      <td>USA</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>825</th>
      <td>11073</td>
      <td>USA</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>826</th>
      <td>11074</td>
      <td>UK</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>827</th>
      <td>11075</td>
      <td>USA</td>
      <td>0.150000</td>
    </tr>
    <tr>
      <th>828</th>
      <td>11076</td>
      <td>USA</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>829</th>
      <td>11077</td>
      <td>USA</td>
      <td>0.027600</td>
    </tr>
  </tbody>
</table>
<p>830 rows × 3 columns</p>
</div>




```python
grp0 = discount_by_country.groupby('country of sales rep').get_group('UK')['average discount in order']
grp1 = discount_by_country.groupby('country of sales rep').get_group('USA')['average discount in order']
```


```python
plt.figure(figsize=(8,5))
plt.bar(x='UK', height=grp0.mean(), yerr=stats.sem(grp0))
plt.bar(x='USA', height=grp1.mean(), yerr=stats.sem(grp1))
plt.xlabel("Country")
plt.ylabel("Average Discount Level per Order")
plt.title("Average Discount Level by Country")
```




    Text(0.5, 1.0, 'Average Discount Level by Country')




![png](output_49_1.png)


It looks like there could be a statistically significant difference between the two. Let's probe further, using normality test and parametric vs. non-parametric tests to check our hypothesis.


```python
stats.normaltest(grp0), stats.normaltest(grp1)
```




    (NormaltestResult(statistic=29.21057587088821, pvalue=4.5394584260173466e-07),
     NormaltestResult(statistic=114.84021802148685, pvalue=1.155482856407473e-25))




```python
# what happens if we run anova?
f_oneway(grp0, grp1)
```




    F_onewayResult(statistic=2.2234335387227073, pvalue=0.13631131652445494)




```python
#ruh roh. what happens if we run mann whitney?
stats.mannwhitneyu(grp0,grp1, alternative='two-sided')
```




    MannwhitneyuResult(statistic=72740.0, pvalue=0.08325052765573286)



A P-value slightly too high to reject the null hypothesis. We cannot confidently say that the UK gives out more or fewer discounts than the US office. It's worth noting, especially considering the Mann-Whitney test's P-value, that specifying a two-tailed test caused us to fail to reject the null hypothesis, whereas specifying a single-tailed test would result in a P-value of 0.04, grounds for rejecting the null hypothesis.

### Conclusion: There is no statistically significant (p > 0.05) difference in average discount level given between the UK and US offices.

# Summary of Conclusions

* Discount levels of 5%, 15%, 20%, and 25% all result in higher order volumes.
* A discount of 5% is just as effective at moving product as a 25% discount.
* Q1 and Q2 sales volume is higher than Q3 and Q4. 

Given the above three points, there is motivation to increase sales efforts during Q3 and Q4, and a technique to increase these sales is to offer products at 5% discount.
