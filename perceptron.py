import pandas as pd
import numpy as np
import random

def calculated_output(theta,weights,u,v,w,x):
    sum = weights[0]*u + weights[1]*v + weights[2]*w + weights[3]*x + weights[4]
    if sum >= theta:
        return 1
    else: return 0

NO_OF_INSTANCES = 80.0
LEARNING_RATE = 0.1
MAX_ITER = 80.0
global_error = 1.0

theta = 0
df = pd.read_csv('iris.csv')

#Iris-setosa are marked 0 class label
df['E'] = df['E'].replace('Iris-setosa',0)

#Iris-versicolor are marked 1 class label
df['E'] = df['E'].replace('Iris-versicolor',1)

a = []
b = []
c = []
d = []
e = []

weights = []

for i in range(40):
    a.append(df.iloc[i]['A'])
    b.append(df.iloc[i]['B'])
    c.append(df.iloc[i]['C'])
    d.append(df.iloc[i]['D'])
    e.append(df.iloc[i]['E'])

for i in range(40,80):
    a.append(df.iloc[i]['A'])
    b.append(df.iloc[i]['B'])
    c.append(df.iloc[i]['C'])
    d.append(df.iloc[i]['D'])
    e.append(df.iloc[i]['E'])

weights.append(random.uniform(0, 1)) #weight for variable a
weights.append(random.uniform(0, 1)) #weight for variable b
weights.append(random.uniform(0, 1)) #weight for variable c
weights.append(random.uniform(0, 1)) #weight for variable d
weights.append(random.uniform(0, 1)) # bias

print weights

iteration = 0
while global_error != 0 and iteration<=MAX_ITER:
    iteration = iteration+1
    global_error = 0.0

    for i in range(80):
        output = calculated_output(theta,weights,a[i],b[i],c[i],d[i])
        local_error = e[i] - output
        weights[0] += LEARNING_RATE * local_error * a[i]
        weights[1] += LEARNING_RATE * local_error * b[i]
        weights[2] += LEARNING_RATE * local_error * c[i]
        weights[3] += LEARNING_RATE * local_error * d[i]
        weights[0] += LEARNING_RATE * local_error
        global_error += (local_error*local_error)

    print "Iteration:",iteration,"RMSE:",(global_error/NO_OF_INSTANCES)**(1/2)


#print "Final weights", weights

print "\nPrediction:(0)",calculated_output(theta,weights,5.1,3.5,1.4,0.2)
print "\nPrediction:(1)",calculated_output(theta,weights,5.5,2.6,4.4,1.2)
print "\nPrediction:(0)",calculated_output(theta,weights,4.9,3.0,1.4,0.2)
print "\nPrediction:(1)",calculated_output(theta,weights,6.1,3.0,4.6,1.4)
print "\nPrediction:(0)",calculated_output(theta,weights,4.7,3.2,1.3,0.2)

    






