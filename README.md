@[TOC](machine learning)
# data access
## formats
1. Unformated text
2. PDF
	have some structure need to use tool
3. HTML documents (web pages)
	include explicit makeup up, better to recognize components than HTML
4. XML and  json document( API )
	good for representing hierarchical structure, have tag
5. ==CSV data file==(spread sheet)
	a (relational) table
	use pandas to provides r/w CSV data in panda

# data pre-processing
## clean data
### dropping NAN value
df.dropna(axis=1, how='all') 
axis = 1 for column
axis = 0 for row

### dropping columns
df.drop(drop_list, inlace=True, axis=1)

### dropping rows on a condition
df.drop(df[boolean conditions].index, inlace=True)
### dropping duplicate Rows
df.drop_duplicates()
for all the value exactly same, drop the last defaultly.

## format data
1. change data type on read
	dtype param in read_csv
2. in dataframe

## manipulate data
1. merge
	1. concatenate two dataset having different columns by using pd.concat
	2. have columns need to be merged by using pd.merge
	3. patch miss data in calling object by using combine_first
2. apply function
	to entire dataset or on the level of columns
3. pivot table
	summary table, introduce new columns from calculations
	e.g.
	![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427205257641.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427205257187.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

4. change the index of a df
	df['columns_name'].is_unique check uniqueness
	df.set_index set the index
	
5. groupby
	df.droupby splits the data into different groups depending on a variable of your choice

6. sort
	df.sort_values()

# data visualisation
## 3 mains purpose and some basics
accuracy, story, knowledge
Aims to create a visualization that are accurate, tell a good story, and provide real knowledge to the audience

 basics:
 - Chart vs Graphs
 	Graph: rely on X or Y or both axe
 	Charts: not restricted by X/Y axies
 - colors
 - keep scales consistent
 - legend and sources

## preparing the data

## paradigm（范式）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427211916939.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)
- scatter plot graph show relationship
- line graph show value "over time" or a continuous interval
- pie chart usually show the percentage but bar graph can do the same job, maybe ok when showing two variable
- histograms are useful for viewing the distribution of data points, the use of discretization helps to see the "bigger picture"


# REST Services
## Architectural constrains of REST
1. client-server
2. uniform interface
3. statelessness
	All calls from clients are independent
	key notion :_separation of client application state and RESTful resource state_
	server does not keep the application state on behalf of a client
4. caching
	reason: stateless make interactions 'chattier'
	only generate data traffic when needed, other times use cache
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427231037168.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)
5. layered system
6. code on demand

_Design satisfies the first five, API is 'RESTFUL'_

## The Richardson Maturity Model
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427232428934.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

# classification(分类)

## (true or false)positive & negative
true positive(真正例): model correctly predicts the positive class
true negative(真负例):  model correctly predicts the negative class
false positive:  model incorrectly predicts the positive class
false negative:  model incorrectly predicts the negative class

negative class contain the element that ==should not== belong to that class

*for example:*

>pred=[0, 1, 1, 2, 1, 1]
>true=[0, 2, 1, 2, 1, 1]
class-----tp----tn----fp----fn
--0-- ----- 1 --- 4----0-----0
--1-- ----- 3 ----4----0----1
--2-- ----- 1 ----4----1-----0

## k nearest neighbor(近邻演算法)
### principle
**non-parametric(无母数统计)**:  no explicit assumptions about the functional form of how the prediction is make
**instance-based(基于实数)**: chooses to memorize the training instances which are subsequently used as 'knowledge' for prediction
**Supervised learning(监督式学习)**

### algorithm
need:
1.four sets(x_train, y_train, x_test, y_test)
2.number of nearest number k to look at
3.optional weighting function

### choose k in k-Nearest Neighbors
The training error rate and the validation error rate are two parameters to access K-value![在这里插入图片描述](https://img-blog.csdnimg.cn/20200409002752221.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)
Rule of thumb is ==k < sqrt(n)==, where n is the number of training examples
**overfit**: if k too big, models will be too complex
**underfit**if k too small, models will be too simple

### Complexity
**expensive at the test time**: O(kdN) need to compute the distance to all N training example
**storage requirement**: must store all training data, which increase exponentially with dimension

### When to consider
– Instance map to points in Rn
– **Less than 20 attributes** per instance 
– Lots of training data

### Advantage
– **Training is very fast**
– Learn complex target functions 
– Do not lose information

### disadvantage
– **Slow at query**
– Easily fooled by **irrelevant attributes**(smooth b having k nearest neighbors vote)
– The Bigger the dataset the more impact on performance


## Decision Tree
### Advantages
- handling of categorical variables
- handling of **missing values and unkown labels**
- detection of **nonlinear relationships**
- visualization and interpretation in decision trees

### When to consider
- instances describable by attribute-value pairs
- target function is discrete valued
- noisy training data
- missing attribute values

_e.g
medical diagnosis_

### Introduction
Each leaf node assigns a classification, determined by majority vote of training example reaching that leaf.
Each internal node is a question on features, tests an attribute
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020042914112785.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)

==smaller the better==

### ID3 algorithm
*The core algorithm for building decision trees is called ID3. Developed by J. R. Quinlan, this algorithm employs a top-down, greedy search through the space of possible branches with no backtracking. ID3 uses Entropy and Information Gain to construct a decision tree.*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429171105649.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

### greedy algorithm
recursive:
select the "best" variable, and generate child nodes: one for each possible value.
Partition samples using the possible values, and assign these subsets of samples to the child nodes.
Repeat for each child node until all samples associated with a node that are either all positive or all negative.

### variable selection
the best variable for partition is the most informative variable, which have ==highest information gain==. So we need to understand ==Entropy== which present the level of impurity.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429154636514.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

Entropy:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429154636876.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

information gain:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429161507683.jpg#pic_center)
E(X,A) need to be smaller to gain higher information gain.

[further information about information gain](https://medium.com/@rishabhjain_22692/decision-trees-it-begins-here-93ff54ef134)

### Occam's Razor
"if two theotries explain the facts equally well, then the simpler theory is to be preferred"

Arguments in favor: fewer short hypotheses than long hypotheses which is unlikely to be coincidence.
Argument opposed: there are many ways to define small sets of hypotheses

### How to deal with unknown attribute values
1. if node n tests A, assign most common value of A among other example sorted to node n
2. Assign most common value of A among other examples with same target value
3. Assign probability pi to each possible value vi of A


### advoid overfitting
1. stop growing the tree when the error doesn't drop by more than a threshold with any new cut.
2. Prue a large tree from leaves to root.


# Regression
*a linear model is a sum of weighted variables that predict a target output value given an input data instance.*

## correlation coefficient
-1<< r << 1
1 indicate a strong positive relationship
0 indicates no relationship at all

## Least square error
*use to estimate model parameters*
*we need to find the minimized squared error*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429181646528.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)

## Gradient Descnt
*a method of updating model paramameters to reduce the least square error*
*As a process move from left highest mountain to the right bottom*

learning rate(size of steps):
A smaller learning rate could get you closer to the minima but more time.
A larger learning rate converges sooner but could overshoot the minima

[to get full kowledge](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)

## linear regression
use line function to explain data
### formular
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429171737784.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)

==model  parameters==: slop and intercept

### conditions
- outcome variable must be continuous
- Little or no multicollinearity between the features
- Normal distribution of error terms
- minimum outliers **(remove them)**
## Multiple linear regression
*used to explain a dependent variable using more than one independent variable*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429183429653.jpg)
### conditions
- **no major correlation** between the independent variables
- the information on the multiple variables can be used to create an accurate prediction on the level of effect they have on the outcome variable.

### selection of features
==P value== for each term test the null hypothesis that the ==correlation coeffiecient is equal to zero==. A low p-value(< 0.05) indicates that you can reject the null hypothesis which means that **low p value is likely to be meaningful feature**.

#### method1: backward elimination
1. set a significance level for which data can stay in the model.
2. fit the full model with all possible predictors
3. if the highest P-value is greater that prediction, then **remove it**.  Go back to the step 2. Else stop elimination.


# Clustering
## feature
* unsupervised learning
* requires data, but no lablels
* detect patterns

## kinds
- Flat algorithms
	usually start with a random partitioning
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429200941944.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

- hierarchical algorithms![在这里插入图片描述](https://img-blog.csdnimg.cn/20200429200941937.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
- hard clustering
	each example belongs to exactly one cluster
- soft clustering
	an example can belong to more than one cluster

## K-means
1. start with some initial cluster centers
2. iterate:
	for each points get distance to each cluster center
	assign/cluster each example to closet center
	Recalculate and change centers as the mean of the points in the cluster
3. Stop when no points' assignments change

### properties
guaranteed to converge in a finite number of iterations
1. assign data points to closest cluster center O(KN) time
2. change the cluster center to average of its assigned points O(N)

### seed choice
Result can vary drastically based on random seed selection

Common heuristics:
- try out multiple starting points
- initialize with the results of another clustering method
- furthest centers heuristic(chose the point is furthest from any previous centers)

### K-mean++
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430102511809.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
### Limitation
- number of cluster is difficult to determine
- does not do well with irregular to complex cluster
- has a problem with data containing outliers

## What is a good clustring
- intra-class similarity is high
- inter-class similarity is low
- intra-cluster cohesion(compactness)
	 use ==sum of squared error== to measure
- inter-cluster speparation(isolation)

## Agglomerative clustering algorithm(Hierarchical algorithms)
1. compute the proximity matrix
2. let each data point be a cluster
3. repeat: merge the two closest cluster and update matrix
4. until only a single cluster remain

*example(single linkage)*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430180251326.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430180252487.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

### Linkage criteria
methods to determine the distance:
1. two most similar parts(single linkage)
2. two least similar bits(complete linkage)
3. center of the cluster(mean or average linkage)

### choose the number of clusters
using a ==Dendrogram==
Dendrogram:
- tree-like diagram that records the sequences of merges or splits
- when two cluster are merges, join them in this dendrogram and the height of the join will be the distance
- set a threshold distance and draw a horizontal line to cut the tallest vertical line
- the number of clusters will be the number of vertical lines which are being intersected by the line drawn using the threshold

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430180922200.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

### Advantages
- easy to implement
- no need to decide the number of cluster beforehand


### Disadvantages
- not suitable for large datasets
- sensitive to outliers
- initial seeds have strong impacts of final results
- linkage criteria and distance measure are selected most of the time arbitrary

# Recommender systems
## The long tail
==RS good at== recommending widely unknown items the user might like
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430115052455.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70#pic_center)

## collaborative filtering(CF)
### Basic idea
user give ratings to catalog item and customers who had similar tasts in the past, will have similar tastes in the future

### Approaches
input: only a matrix of given user-item ratings
output: a numerical prediction indicating to what degree the current user will like to dislike a certain item

### User-based nearest-neighbor CF
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430120232306.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430120232571.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
#### similarity measure: **Pearson correlation**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430134706370.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

#### prediction: 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430134716479.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)

#### improving the prediction funtion
1. use more weight(not all neighbor ratings might be equally "variable")
2. linearly reducing the weight when the number of co-rated items is low
3. Case amplification: give more weight to "very similar" neighbors.
4. neighborhood selection: use similarity threshold or fix number of neighbors.

### Item-based CF
#### Basic idea
use the similarity between items to make prediction
item-based filtering does not solve the scalability itself

#### preprocessing
calculate all pair-wise item silimarities in advance.

Memory requirements
- up to N^2 pair-wise similarity to be memorized(N = number of items) in theory
- significantly lower in practice.
- further reduction: 1. minimum threshold for co-ratings 2.limit neighborhood size.

### Rating
implicit ratings: collected by the application in which the recommender system is embedded(e.g. shopping history), no require efforts from the side of user. But it could be interpreted.

explicit: from the actual ratings given by user.

implicit ratings can be used in addiction to explicit one cuz question of correcness of interpretation.

### limitation
 - cold start
 1. new items need to get enough ratings to be recommended. (content-based)
 2. new user must provide user's preference.(hybrid approach)
 - popularity bias: unique tastes
 - trends to recommend popular items cuz items from the tail don't get so much data

## Content-based recommendation
use information about available items as "content" and user profile as their preference. Learn user preferences and recommend items that are "similar" to the user preference.

### simple Approach
- Given a set of documents D already rated by user
- Find the n nearest neighbors of an not-yet-seem item I in D
	user similarity measures
- take these neighbors to predict a rating for i.


### Advantages
- good to model short-term interest/ follows-up stories
- can use when CF get cold start

### disadvantages
- keywords may not be sufficient to judge quality/relevance of documents
- ramped-up phase required
- overspecializarion

# Evaluation
## Accuracy
## precision & Recall
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430193408675.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200430193347198.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h4eWNocmlzdGluYQ==,size_16,color_FFFFFF,t_70)
## F1 score
F1-score=2*(Precision*Recall)/(precision+recall)

## Cross-validatiobn
*is a resampling procedure used to evaluate machine learning models on a limited data sample*

### Stratification
*is a technique where we rearange the data in a way that each fold has a good representation of the whole dataset*

each fold to have at least m instances of each class

# Deep learning


# scikit-learn
### sklearn.metric
precision_score = tp/tp + fp
recall_score = tp/tp + fn

