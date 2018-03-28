import numpy as np
from numpy import gefromtxt
from sklearn import linear_model

file=genfromtxt("iris.data",delimiter=",",dtype="str")
dic={}
for val in file:
count=0
if val[4] not in dic:
dic[val[4]]=count
count +=1
for val in file:
 val[4]=dict[val[4]]

trainingSet = file[:130] 
testingset =file[130:] 

trainingX =trainingSet[:,[0,1,2,3]]
trainingX = trainingX.astype(float)
trainingY =trainingSet[:,[4]]

testingX =testingSet[:,[0,1,2,3]]
testingX = testingX.astype(float)
testingY =testingSet[:,[4]]

lr=linear_model.LogisticRegression()
lr.fit(trainingX,trainingY)

print(testingX[12])
print("Predicted value is "+str(lr.predict([testingX[12]])))
print("Real value is"+str(testingY[12]))

lr.score(testingX,testingY)*100