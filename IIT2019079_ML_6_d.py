
import numpy as np
import pandas as pd
import math

def calculate_error(Y, Y_pred):
    error = 0
    for i in range(len(Y)):
        error += abs(Y[i] - Y_pred[i]) / Y[i]
    error = error / len(Y)
    return error * 100

def kernel(X, xi, Hyper_Parameter_Tau):
    return np.exp(-np.sum((xi - X) ** 2, axis = 1) / (2 * Hyper_Parameter_Tau * Hyper_Parameter_Tau))

def LocallyWeightedLR(X, xi, Y, Hyper_Parameter_Tau):
	TransposeX = np.transpose(X)
	W = kernel(X, xi, Hyper_Parameter_Tau)
	XTransposeW = TransposeX * W
	XTransposeWX = np.matmul(XTransposeW, X)
	InverseXTransposeWX = np.linalg.pinv(XTransposeWX)
	InverseXTransposeWXXTransposeW = np.matmul(InverseXTransposeWX, XTransposeW)
	InverseXTransposeWXXTransposeWY = np.matmul(InverseXTransposeWXXTransposeW, Y)
	InverseXTransposeWXXTransposeWYTranspose = np.transpose(InverseXTransposeWXXTransposeWY)
	return InverseXTransposeWXXTransposeWYTranspose.dot(xi)

input_data = pd.read_csv('Housing Price data set.csv', usecols = ["price", "lotsize", "bedrooms", "bathrms"])
Floor_Area_Train_Set = input_data['lotsize']
No_Of_Bedrooms_Train_Set = input_data['bedrooms']
No_Of_Bathrooms_Train_Set = input_data['bathrms']

Y = input_data['price']
Y = np.array(Y)
Y = Y.reshape(Y.shape[0], 1)

# Performing feature scanning on Floor_Area_Train_Set
Floor_Area_Train_Set_Mean = np.mean(Floor_Area_Train_Set)
Floor_Area_Train_Set_Max = max(Floor_Area_Train_Set)
Floor_Area_Train_Set_Min = min(Floor_Area_Train_Set)
Floor_Area_Train_Set_Scaled = []
for i in Floor_Area_Train_Set:
	Floor_Area_Train_Set_Scaled.append((i - Floor_Area_Train_Set_Mean) / (Floor_Area_Train_Set_Max - Floor_Area_Train_Set_Min))

X = []
for i in range(len(Floor_Area_Train_Set)):
	X.append([1, Floor_Area_Train_Set_Scaled[i], No_Of_Bedrooms_Train_Set[i], No_Of_Bathrooms_Train_Set[i]])
X = np.array(X)

Hyper_Parameter_Tau = 0.00005
print("Using Locally Weighted Linear Regression for Tau = " + str(Hyper_Parameter_Tau))
pred = []
for i in range(X.shape[0]):
	y_pred = LocallyWeightedLR(X, X[i], Y, Hyper_Parameter_Tau)
	pred.append(y_pred)
print("Mean absolute percentage error is : " + str(calculate_error(Y,pred)))
print()

Price = input_data['price']

#segmenting the features
Features_Train_Set = []
for i in range(383):
	Features_Train_Set.append([1, Floor_Area_Train_Set_Scaled[i], No_Of_Bedrooms_Train_Set[i], No_Of_Bathrooms_Train_Set[i]])
Price_Train_Set = Price[:383]
PriceTest = []
Features_Test = []
for i in range(383, len(Price)):
	Features_Test.append([1, Floor_Area_Train_Set_Scaled[i], No_Of_Bedrooms_Train_Set[i], No_Of_Bathrooms_Train_Set[i]])
	PriceTest.append(Price[i])
m = len(Features_Train_Set)

# Function to calculate Slope to find coefficients
def Slope(Coeff, Features_Train_Set, Price_Train_Set, ind):
	Error = 0
	for i in range(len(Features_Train_Set)):
		itr = 0
		for j in range(len(Coeff)):
			itr = itr + Coeff[j] * Features_Train_Set[i][j]
		Error += (itr - Price_Train_Set[i]) * Features_Train_Set[i][ind]
	return Error

# Using scaled batch gradient with regularisation
print("Using scaled batch gradient with regularisation")
LearningRate = 0.005
Lambda_Parameter = -45
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)
for epochs in range(5000):
	Temp_Coeff = Coeff.copy()
	for j in range(len(Coeff)):
		if (j == 0):
			Temp_Coeff[j] = Temp_Coeff[j] - ((LearningRate / m) * (Slope(Coeff, Features_Train_Set, Price_Train_Set, j)))	
		else:
			Temp_Coeff[j] = (1 - LearningRate * Lambda_Parameter / m) * Temp_Coeff[j] - ((LearningRate / m) * (Slope(Coeff, Features_Train_Set, Price_Train_Set, j)))
	Coeff = Temp_Coeff.copy()
print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * Features_Test[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

def SlopeStoch(Coeff,Features_Train_Set,ActualVal,ind):
	itr = 0
	for j in range(len(Coeff)):
		itr = itr + Coeff[j]*Features_Train_Set[j]
	return (itr - ActualVal) * Features_Train_Set[ind]

# Using Scaled Stochastic gradient with regularisation.
print("Using Stochastic gradient with regularisation")

# I tried with different values of tau but found this to be the best.
LearningRate = 0.004
Lambda_Parameter = 142000
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)

for iter in range(10):
	for i in range(len(Price_Train_Set)):
		Temp_Coeff = Coeff.copy()
		for j in range(4):
			if j == 0:
				Temp_Coeff[j] = Temp_Coeff[j] - (LearningRate * (SlopeStoch(Coeff, Features_Train_Set[i], Price_Train_Set[i], j)))
			else:
				Temp_Coeff[j] = (1 - LearningRate * Lambda_Parameter / m) * Temp_Coeff[j] - (LearningRate * (SlopeStoch(Coeff, Features_Train_Set[i], Price_Train_Set[i], j)))
		Coeff = Temp_Coeff.copy()

print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * Features_Test[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Minibatch gradient with regularisation for batch size = 20
print("Using Scaled Minibatch gradient with regularisation for batch size = 20")

Batch_Size = 20;
LearningRate = 0.002
Lambda_Parameter = -372
Coeff = [0, 0, 0, 0]
No_Of_Batches = math.ceil(len(Price_Train_Set) / Batch_Size)
equallyDiv = False
if (len(Price_Train_Set) % Batch_Size == 0):
	equallyDiv = True;

for epoch in range(30):
	for batch in range(No_Of_Batches):
		Sum = [0, 0, 0, 0]
		for j in range(len(Coeff)):
			for i in range(Batch_Size):
				if (batch * Batch_Size + i == len(Features_Train_Set)):
					break
				Predicted_Value = 0.0
				for wj in range(len(Coeff)):
					Predicted_Value += Coeff[wj] * Features_Train_Set[batch * Batch_Size + i][wj]
				Predicted_Value -= Price_Train_Set[batch * Batch_Size + i]
				Predicted_Value *= Features_Train_Set[batch * Batch_Size + i][j]
				Sum[j] += Predicted_Value;

		if (not equallyDiv and batch == No_Of_Batches - 1):
			for j in range(len(Sum)):
				if j == 0:
					Coeff[j] -= (Sum[j] / (len(Price_Train_Set) % Batch_Size)) * LearningRate
				else:
					Coeff[j] = (1 - LearningRate * Lambda_Parameter / m) * Coeff[j] - (Sum[j] / (len(Price_Train_Set) % Batch_Size)) * LearningRate
		else:
			for j in range(len(Sum)):
				if j == 0:
					Coeff[j] -= (Sum[j] / Batch_Size) * LearningRate
				else:
					Coeff[j] = (1 - LearningRate * Lambda_Parameter / m) * Coeff[j] - (Sum[j] / Batch_Size) * LearningRate
print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * Features_Test[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()