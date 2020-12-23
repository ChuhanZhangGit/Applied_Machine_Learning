import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix


fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

fruits.head()

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   



## plotting a scatter matrix
from matplotlib import cm

# create training and test split
# train_test_split shuffle data and split to 75% to train and 25% to test.
# X is attributes or flavors
# y is label
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

## plotting a 3D scatter plot
#from mpl_toolkits.mplot3d import Axes3D
# K - Nearest Neighbor , k refer to the number of nearest neighbor, when k > 1, classify to majority neighbor.
# Instance base learning, 
# 1. Find similar instance, 2. Get the labels for x-NN 3. predict the label for x_test by combining the label y_NN

##
# A distance metric, Euclidean(linear distance), more general Minkowski with p =2
# how many neighbors to look? 
# weights on different neighbor
# how to aggregate the classes of neighbor


#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
#ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
#ax.set_xlabel('width')
#ax.set_ylabel('height')
#ax.set_zlabel('color_score')
#plt.show()


# Create classifier object

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

# Train the classifier using training data. Update state of knn of variable
knn.fit(X_train, y_train)


# Estimate the accuracy of the classifier
knn.score(X_test, y_test)

# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
lookup_fruit_name[fruit_prediction[0]]


# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
lookup_fruit_name[fruit_prediction[0]]

from adspy_shared_utilities import plot_fruit_knn

plot_fruit_knn(X_train, y_train, 5, 'uniform')   # we choose 5 nearest neighbors


k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);