import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn

#imports the data 
import csv
file = open('spy.csv')
type = file
csvreader = csv.reader(file)
header = []
header = next(csvreader)
header

rows = []
for row in csvreader:
    rows.append(row)
rows.pop()

file.close()


numbers = []


#Takes out the unessecary string values to convert them later
for i in range(len(rows)):
    rows[i][0] = rows[i][0].strip().strip(r"'\"")
    rows[i][1] = rows[i][1].strip('\\\"')
    rows[i][2] = rows[i][2].strip('\\\"')
    if i != len(rows) - 1:
        rows[i].pop()
    
#Puts the data into a new array in order to manipulate
for item in rows:
    numbers.append(item)

#takes out the date leaving with closing price and volume
#then converts this array to a floating point array
for i in range(len(rows)):
    numbers[i].pop(0)
    j = 0
    while j < 2:
        numbers[i][j] = float(numbers[i][j])
        j+=1
    print(numbers[i][1])
#changes to a numpy array then plots it
numbers = np.array(numbers)
#pyplot.scatter(numbers[:,0],numbers[:,1])
#pyplot.grid(True)
#pyplot.show()

#centers the data at 0
mean = np.mean(numbers, axis= 0)
mean_numbers = numbers - mean
print("Mean ", mean[1])
pyplot.scatter(mean_numbers[:,0],mean_numbers[:,1])
pyplot.grid(True)
pyplot.show()
