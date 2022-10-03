#-------------------------------------------------------------------------
# AUTHOR: Henry Hu
# FILENAME: naive_bayes.py
# SPECIFICATION: Complete the naive_bayes to read weather_trainings.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

training_set = []

# reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            training_set.append(row)

value_map = {"D1": 1, "D2": 2, "D3": 3, "D4": 4, "D5": 5, "D6": 6, "D7": 7, "D8": 8, "D9": 9, "D10": 10, "D11": 11,
             "D12": 12, "D13": 13, "D14": 14, "Sunny": 1, "Overcast": 2, "Rain": 3, "Hot": 1, "Mild": 2, "Cool": 3,
             "High": 1, "Normal": 2, "Weak": 1, "Strong": 2, "No": 1, "Yes": 2}

X = []
Y = []

for data in training_set:
    sample = []
    for i, value in enumerate(data):
        if i != 5:
            sample.append(value_map.get(value))
        else:
            Y.append(value_map.get(value))
    X.append(sample)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]


# reading the test data in a csv file
test_set = []

# reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            test_set.append(row)

A = []
new_map_counter = len(value_map) + 1

for data in test_set:
    sample = []
    for i, value in enumerate(data):
        # Handle new data not found
        temp = value_map.get(value)
        if temp is not None:
            value = temp
        else:
            value = new_map_counter
            new_map_counter += 1
        if i != 5:
            sample.append(value)
    A.append(sample)

for i in range(len(A)):
    temp_str = (str(test_set[i][0]).ljust(15) +str(test_set[i][1]).ljust(15) +str(test_set[i][2]).ljust(15)
                +str(test_set[i][3]).ljust(15) +str(test_set[i][4]).ljust(15))

    temp_calc = str(clf.predict_proba([A[i]])[0]).split(" ")
    play_tenis = 'No'
    if temp_calc[0] < temp_calc[1]:
        play_tenis = 'Yes'

    print(temp_str, play_tenis, temp_calc[0], temp_calc[1])