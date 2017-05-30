
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''

import numpy as np
import os
import csv

import matplotlib.pyplot as plt
from sklearn import svm, neighbors

from sklearn.cross_validation import cross_val_score

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9708651, 'Christopher', 'O\'Rafferty'), (9400001, 'Moira', 'Quinn'), (7226209, 'Maurice', 'Cafun') ]
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''

    # Declare Numpy Arrays
    X = np.empty(shape=[0,1],dtype=float)
    y = np.empty(shape=[0],dtype=bool)    
    
    # Read in Data with CSV Reader
    with open(dataset_path, newline='') as csvfile:
        # Go through each row and build y numpy array
        patients = csv.reader(csvfile,delimiter=',')
        for row in patients:
            numcols=len(row[2:])
            y = np.append(y, [int(row[1] == 'M')], axis=0)

    # Create X numpy array
    X = np.genfromtxt(dataset_path,delimiter=',',usecols=range(2,numcols+2))

    return X,y

    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
      
    max_tests = 60
    k_range = range(1, max_tests)
    
    k_scores = []
    best_k = 1
    
    for k in k_range:
        clf = neighbors.KNeighborsClassifier(k)
        score = cross_val_score(clf, X_training, y_training, scoring="accuracy", cv = 10).mean()
        if k > 1 and score >= k_scores[best_k-1]:
            best_k = k
            
        k_scores.append(score)
                    
    """
    plt.plot(k_range, k_scores)
    plt.xlabel("k for KNN")
    plt.ylabel("accuracy")
    plt.show()
    """
    
    clf = neighbors.KNeighborsClassifier(best_k)
    clf.fit(X_training, y_training)
    return clf
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    max_tests = 60
    p_range = range(0, max_tests)
    
    best_p_score = 0
    best_p_index = 0
    
    int count = 0
    
    for p in p_range:
        clf = svm.LinearSVC(C = p)
        score = cross_val_score(clf, X_training, y_training, scoring="accuracy", cv = 10).mean()
        if p > 1 and score >= p_scores[best_p_index - 1]:
            best_p = p
            
        p_scores.append(score)
        count += 1
                    
    print (best_p)
        
    plt.plot(p_range, p_scores)
    plt.xlabel("penalty for SVC")
    plt.ylabel("accuracy")
    plt.show()
    
    
    clf = svm.LinearSVC(C = p_range[best_p_index])
    clf.fit(X_training, y_training)
    return clf
    
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def random_permutation(X, y):    
    n = X.shape[0]
    p = np.random.permutation(n)
    
    X, y = X[p], y[p]
    
    return X, y

    
def find_file():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    rel_path = "medical_records.data"
    return os.path.join(script_dir, rel_path)
    

if __name__ == "__main__":
    pass
    # call your functions here

    # my_team
    print(my_team())
    
    # prepare_dataset    
    X, y = prepare_dataset(find_file())   
    X, y = random_permutation(X, y)
    
    #knn_clf = build_NN_classifier(X, y)
    svm_clf = build_SVM_classifier(X, y)
    
    knn_score = cross_val_score(knn_clf, X, y, cv = 10).mean()   
    print (knn_score)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    


