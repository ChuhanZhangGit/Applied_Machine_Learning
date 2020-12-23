import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()


# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])



def answer_one():
    # Your code here
    df = pd.DataFrame(data= np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']) )
    return df


def answer_two():
    cancerdf = answer_one()
    #print((cancerdf.target == 1).sum())
    
    series = cancerdf['target'].value_counts()
    series = series.rename({0.0: 'malignant', 1.0:'benign'})

    return series



def answer_three():
    cancerdf = answer_one()
    # Your code here
    y = cancerdf['target']
    X = cancerdf.drop(columns='target')
    
    return X, y



from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    # Your code here
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    # Your code here
    knn= KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return knn


def answer_six():
    # Your code here    
    cancerdf = answer_one()
    knn = answer_five()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    pred = knn.predict(means)
    return pred



def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    pred = knn.predict(X_test)
       # Your code here
    
    return pred



def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    # Your code here
    
    return knn.score(X_test, y_test)


import matplotlib.pyplot as plt

def accuracy_plot():

 

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                       ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()
    
    
accuracy_plot()



