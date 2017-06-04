
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import cross_val_score

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
    
    clfG = GaussianNB()
    clfM = MultinomialNB()
    clfB = BernoulliNB()
    
    G_scores = []
    M_scores = []
    B_scores = []

    iterations = 20
    
    for i in range(iterations):
        X, y = random_permutation(X_training, y_training)
        G_scores.append(cross_val_score(clfG, X, y, scoring="accuracy", cv = 10).mean())
        M_scores.append(cross_val_score(clfM, X, y, scoring="accuracy", cv = 10).mean())
        B_scores.append(cross_val_score(clfB, X, y, scoring="accuracy", cv = 10).mean())
        
    
    
    plt.plot(range(iterations), G_scores, 'r', range(iterations), M_scores, 'g', range(iterations), B_scores, 'b')
    plt.xlabel("k for KNN")
    plt.ylabel("accuracy")
    plt.show()
    
    G_mean = np.mean(G_scores)
    M_mean = np.mean(M_scores)
    B_mean = np.mean(B_scores)
    
    # return the best classifier
    if G_mean > max(M_mean, B_mean): 
        clfG.fit(X_training, y_training)    
        return clfG
    
    if M_mean > max(G_mean, B_mean):
        clfM.fit(X_training, y_training)    
        return clfM
    
    clfB.fit(X_training, y_training)    
    return clfB
    
    
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
    
    maximum_range = 100
    
    depth_range = range(2, maximum_range)
    
    max_scores = []
    
    best_max = 2
    max_score = 0;
    
    for m in depth_range:
        clf = DecisionTreeClassifier(max_depth=m)

        score = cross_val_score(clf, X_training, y_training, scoring="accuracy", cv = 10).mean()

        if m > 1 and score >= max_score:
            best_max = m
            max_score = score


        max_scores.append(score)
   

    plt.plot(depth_range, max_scores, 'r')
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.show()
    

    clf = DecisionTreeClassifier(max_depth=best_max)
    clf = clf.fit(X_training, y_training)
    

    
    return clf
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training, return_best_k = False):
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
    
    # Finds the best K value for the KNN classifier based on the inputted data
    for k in k_range:
        clf = neighbors.KNeighborsClassifier(k)
        score = cross_val_score(clf, X_training, y_training, scoring="accuracy", cv = 10).mean()
        if k > 1 and score >= k_scores[best_k-1]:
            best_k = k
            
        k_scores.append(score)
                    
    """
    plt.plot(k_range, k_scores)
    plt.xlabel("Value of k")
    plt.ylabel("Cross valued score")
    plt.show()
    """
    
    if return_best_k:
        return best_k
    
    clf = neighbors.KNeighborsClassifier(best_k)
    clf.fit(X_training, y_training)
    return clf
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training, return_best_C = False):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    print ("--Starting SVC--")
    
    num_tests = 200
    multiplier = 0.5
    c_range = range(1, num_tests)
    
    c_scores = []
    best_c = 1
    
    progress_counter = 20
    
    # Finds the best C value for the SVC based on the inputted data
    for c in c_range:
        clf = svm.LinearSVC(C=float(c * multiplier))
        score = cross_val_score(clf, X_training, y_training, scoring="accuracy", cv = 10).mean()
        c_scores.append(score)
        if score >= c_scores[best_c-1]:
            best_c = c            
        
        if c % progress_counter == 0:
            print (str((c / num_tests) * 100) + "% complete.",)
    
   
    print ("--Finished SVC--")

    """
    plt.plot(np.arange(multiplier, num_tests * multiplier, multiplier), c_scores)
    plt.xlabel("Value of C")
    plt.ylabel("Cross valued score")
    plt.show()
    """
    
    if return_best_C:
        return best_c * multiplier
    
    # Note the value generated from this clf may not be the best value found above, due to random generation
    clf = svm.LinearSVC(C=best_c * multiplier)
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
    

def do_svm(X, y):
    svm_clf = build_SVM_classifier(X, y)
    svm_score = cross_val_score(svm_clf, X, y, scoring="accuracy", cv = 10).mean()   
    print (svm_score)
    return svm_clf



def do_knn(X, y):
    knn_clf = build_NN_classifier(X, y)
    knn_score = cross_val_score(knn_clf, X, y, scoring="accuracy", cv = 10).mean()  
    print (knn_score)
    return knn_clf

    

def do_nb(X, y):
    nb_clf = build_NB_classifier(X, y)
    nb_score = cross_val_score(nb_clf, X, y, scoring="accuracy", cv = 10).mean()  
    print (nb_score)
    return nb_clf

def do_dt(X, y):    
    dt_clf = build_DT_classifier(X, y)
    dt_score = cross_val_score(dt_clf, X, y, scoring="accuracy", cv = 10).mean()  
    print (dt_score)
    return dt_clf


def average_best_k(X, y):
    iterations = 100
    best_ks = []
    
    for i in range(iterations):
        X, y = random_permutation(X, y)
        best_ks.append(build_NN_classifier(X, y, True))
        
    plt.plot(range(iterations), best_ks)
    plt.xlabel("Iteration (100 total)")
    plt.ylabel("Best k for this permutation")
    plt.show()
    
    print ("The average best k value is:", np.mean(best_ks))
    
    
def average_best_C(X, y):
    iterations = 50
    best_Cs = []
    
    for i in range(iterations):
        X, y = random_permutation(X, y)
        best_Cs.append(build_SVM_classifier(X, y, True))
        
    plt.plot(range(iterations), best_Cs)
    plt.xlabel("Iteration (50 total)")
    plt.ylabel("Best C for this permutation")
    plt.show()
    
    print ("The average best C value is:", np.mean(best_Cs))
        


    
if __name__ == "__main__":
    pass
    # call your functions here

    # my_team
    print(my_team())
    
    # prepare_dataset    
    X, y = prepare_dataset(find_file())   
    X, y = random_permutation(X, y)
    
    #do_svm(X, y)
    #do_knn(X, y)
    #do_nb(X, y)
    do_dt(X, y)

    
    #average_best_k(X, y)
    #average_best_C(X, y)

    


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    


