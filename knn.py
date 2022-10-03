#-------------------------------------------------------------------------
# AUTHOR: Henry Hu
# FILENAME: knn.py
# SPECIFICATION: Completing the KNN program to read binary.csv and output the LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


num_correct = 0
num_incorrect = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    X = []
    Y = []
    test_sample = []
    ground_truth = 1.0


    #add the training features to the 2D array X and remove the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    #transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning messages

    for a, sample in enumerate(db):
        # leave one out
        if a != i:
            # add to the current to training set.
            X.append([float(sample[0]), float(sample[1])])
            classifier = 1.0
            if sample[2] == '+':
                classifier = 2.0
            Y.append(classifier)
        else:
            test_sample = [float(sample[0]), float(sample[1])]
            if sample[2] == '+':
                ground_truth = 2.0


    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]

    class_predicted = clf.predict([test_sample])[0]
    print('Test #',i, 'predicts ', class_predicted, ' with groundtruth as = ', ground_truth)
    if class_predicted == ground_truth:
        num_correct += 1
    else:
        num_incorrect += 1

    #compare the prediction with the true label of the test instance to start calculating the error rate.
print('The error rate is =', num_correct/float(num_incorrect+num_correct))
#print the error rate











