#-------------------------------------------------------------------------
# AUTHOR: Henry Hu
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# These datasets contain the following features:Age,Spectacle Prescription,Astigmatism,Tear Production Rate, Recommended Lenses
value_map = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3, "Myope": 1, "Hypermetrope": 2, "Yes": 1, "No": 2,
             "Normal": 1, "Reduced": 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    lowest_accuracy = 1

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

    for training_instance in dbTraining:
        sample = []
        for i, value in enumerate(training_instance):
            if i != 4:
                sample.append(value_map.get(value))
            else:
                Y.append(value_map.get(value))
        X.append(sample)

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # transform the features of the test instances to numbers following the same strategy done during training,
        # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label

        dbTest = []
        ground_truth = -1
        num_correct = 0.0
        num_incorrect = 0.0
        # reading the testing data in a csv file
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for c, row in enumerate(reader):
                if c > 0:  # skipping the header
                    dbTest.append(row)

        for data in dbTest:
            for a, val in enumerate(data):
                if a != 4:
                    data[a] = value_map.get(val)
                else:
                    ground_truth = value_map.get(data[a])
                    del data[-1]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            result = clf.predict([data])[0]
            if result != ground_truth:
                num_incorrect += 1
            else:
                num_correct += 1

        accuracy = num_correct/float(num_correct + num_incorrect)
        print('Test #', i, ' Accuracy is ', accuracy)
        #find the lowest accuracy of this model during the 10 runs (training and test set)
        if lowest_accuracy > accuracy:
            lowest_accuracy = accuracy

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print('The lowest accuracy from testing', ds, 'is : ', lowest_accuracy)



