
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''

import numpy as np
import os
import csv

from sklearn import svm, neighbors

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
      
    clf = neighbors.KNeighborsClassifier()
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
    
    clf = svm.LinearSVC()    
    clf.fit(X_training, y_training)
    
    return clf
    
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def split_training_data(X, y):    
    n = X.shape[0]
    splitPoint = int(n*0.9)
    
    XTrain, XTest = X[:splitPoint], X[splitPoint:]
    yTrain, yTest = y[:splitPoint], y[splitPoint:]
    
    #np.set_printoptions(threshold=np.inf)
    
    svc = build_SVM_classifier(XTrain, yTrain)   
    print("SVC", svc.score(XTest, yTest)) 
        
    knn = build_NN_classifier(XTrain, yTrain)
    print("KNN", knn.score(XTest, yTest))
    

def random_permutation(X, y):    
    n = X.shape[0]
    p = np.random.permutation(n)
    
    X, y = X[p], y[p]
    
    return X, y


if __name__ == "__main__":
    pass
    # call your functions here

    # my_team
    print(my_team())
    
    # prepare_dataset
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    rel_path = "medical_records.data"
    abs_file_path = os.path.join(script_dir, rel_path)
    
    X, y = prepare_dataset(abs_file_path)
    X, y = random_permutation(X, y)
    split_training_data(X, y)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    


