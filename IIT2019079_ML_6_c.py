
import numpy as np
import pandas as pd
import math

#Reading input file
input_data = pd.read_csv("Housing Price data set.csv")


Price = input_data['price']
Train_Set_Floor_Area = input_data['lotsize']
Train_Set_No_Of_Bedrooms = input_data['bedrooms']
Train_Set_No_Of_Bathrooms = input_data['bathrms']

#feature scaling on Train_Set_Floor_Area
Train_Set_Floor_Area_Mean = np.mean(Train_Set_Floor_Area)
Train_Set_Floor_Area_Max = max(Train_Set_Floor_Area)
Train_Set_Floor_Area_Min = min(Train_Set_Floor_Area)
Train_Set_Floor_Area_Scaled = []

for i in Train_Set_Floor_Area:
	Train_Set_Floor_Area_Scaled.append((i - Train_Set_Floor_Area_Mean) / (Train_Set_Floor_Area_Max - Train_Set_Floor_Area_Min))

#segmenting the features

Features_Train_Set = []
for i in range(383):
	Features_Train_Set.append([1, Train_Set_Floor_Area_Scaled[i], Train_Set_No_Of_Bedrooms[i], Train_Set_No_Of_Bathrooms[i]])
Price_Train_Set = Price[:383]

#Test examples
PriceTest = []
Features_Test_Set = []
for i in range(383, len(Price)):
	Features_Test_Set.append([1, Train_Set_Floor_Area_Scaled[i], Train_Set_No_Of_Bedrooms[i], Train_Set_No_Of_Bathrooms[i]])
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

# Using scaled batch gradient without regularisation

print("Using scaled batch gradient without regularisation")
Learning_Rate = 0.001
m = len(Features_Train_Set)

Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)
for i in range(5000):
	TempCoeff = Coeff.copy()
	for j in range(len(Coeff)):
		TempCoeff[j] = TempCoeff[j] - ((Learning_Rate / m) * (Slope(Coeff, Features_Train_Set, Price_Train_Set, j)))
	Coeff = TempCoeff.copy()
print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.

Error = 0
for i in range(len(Features_Test_Set)):
	prediction = 0
	for j in range(len(Coeff)):
	  	prediction = prediction + Coeff[j] * Features_Test_Set[i][j]
	Error += abs(prediction - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test_Set)) * 90
print("Mean absolute percentage error is : " + str(Error))
print()

# Using scaled batch gradient with regularisation

print("Using scaled batch gradient with regularisation")
Learning_Rate = 0.001
Lambda_Parameter = -49
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)
for epochs in range(5000):
	TempCoeff = Coeff.copy()
	for j in range(len(Coeff)):
		if (j == 0):
			TempCoeff[j] = TempCoeff[j] - ((Learning_Rate / m) * (Slope(Coeff, Features_Train_Set, Price_Train_Set, j)))	
		else:
			TempCoeff[j] = (1 - Learning_Rate * Lambda_Parameter / m) * TempCoeff[j] - ((Learning_Rate / m) * (Slope(Coeff, Features_Train_Set, Price_Train_Set, j)))
	Coeff = TempCoeff.copy()
print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test_Set)):
	prediction = 0
	for j in range(len(Coeff)):
	  	prediction = prediction + Coeff[j] * Features_Test_Set[i][j]
	Error += abs(prediction - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test_Set)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

def SlopeStoch(Coeff,Features_Train_Set,ActualVal,ind):
	itr = 0
	for j in range(len(Coeff)):
		itr = itr + Coeff[j]*Features_Train_Set[j]
	return (itr - ActualVal) * Features_Train_Set[ind]

# Using Scaled Stochastic gradient without regularisation.
print("Using Stochastic gradient without regularisation")

Learning_Rate = 0.005
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)

for iter in range(10):
	for i in range(len(Price_Train_Set)):
		TempCoeff = Coeff.copy()
		for j in range(4):
			TempCoeff[j] = TempCoeff[j] - (Learning_Rate * (SlopeStoch(Coeff, Features_Train_Set[i], Price_Train_Set[i], j)))
		Coeff = TempCoeff.copy()

print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test_Set)):
	prediction = 0
	for j in range(len(Coeff)):
	  	prediction = prediction + Coeff[j] * Features_Test_Set[i][j]
	Error += abs(prediction - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test_Set)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Stochastic gradient with regularisation.
print("Using Stochastic gradient with regularisation")

Learning_Rate = 0.005
Lambda_Parameter = 142000
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)

for iter in range(10):
	for i in range(len(Price_Train_Set)):
		TempCoeff = Coeff.copy()
		for j in range(4):
			if j == 0:
				TempCoeff[j] = TempCoeff[j] - (Learning_Rate * (SlopeStoch(Coeff, Features_Train_Set[i], Price_Train_Set[i], j)))
			else:
				TempCoeff[j] = (1 - Learning_Rate * Lambda_Parameter) * TempCoeff[j] - (Learning_Rate * (SlopeStoch(Coeff, Features_Train_Set[i], Price_Train_Set[i], j)))
		Coeff = TempCoeff.copy()

print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test_Set)):
	prediction = 0
	for j in range(len(Coeff)):
	  	prediction = prediction + Coeff[j] * Features_Test_Set[i][j]
	Error += abs(prediction - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test_Set)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Minibatch gradient without regularisation for batch size = 20
print("Using Scaled Minibatch gradient without regularisation for batch size = 20")

Batch_Size = 20;
Learning_Rate = 0.005
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
				predicted_Value = 0.0
				for wj in range(len(Coeff)):
					predicted_Value += Coeff[wj] * Features_Train_Set[batch * Batch_Size + i][wj]
				predicted_Value -= Price_Train_Set[batch * Batch_Size + i]
				predicted_Value *= Features_Train_Set[batch * Batch_Size + i][j]
				Sum[j] += predicted_Value;

		if (not equallyDiv and batch == No_Of_Batches - 1):
			for j in range(len(Sum)):
				Coeff[j] -= (Sum[j] / (len(Price_Train_Set) % Batch_Size)) * Learning_Rate
		else:
			for j in range(len(Sum)):
				Coeff[j] -= (Sum[j] / Batch_Size) * Learning_Rate
print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test_Set)):
	prediction = 0
	for j in range(len(Coeff)):
	  	prediction = prediction + Coeff[j] * Features_Test_Set[i][j]
	Error += abs(prediction - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test_Set)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Minibatch gradient with regularisation for batch size = 20
print("Using Scaled Minibatch gradient with regularisation for batch size = 20")

Batch_Size = 20;
Learning_Rate = 0.002
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
				predicted_Value = 0.0
				for wj in range(len(Coeff)):
					predicted_Value += Coeff[wj] * Features_Train_Set[batch * Batch_Size + i][wj]
				predicted_Value -= Price_Train_Set[batch * Batch_Size + i]
				predicted_Value *= Features_Train_Set[batch * Batch_Size + i][j]
				Sum[j] += predicted_Value;

		if (not equallyDiv and batch == No_Of_Batches - 1):
			for j in range(len(Sum)):
				if j == 0:
					Coeff[j] -= (Sum[j] / (len(Price_Train_Set) % Batch_Size)) * Learning_Rate
				else:
					Coeff[j] = (1 - Learning_Rate * Lambda_Parameter / m) * Coeff[j] - (Sum[j] / (len(Price_Train_Set) % Batch_Size)) * Learning_Rate
		else:
			for j in range(len(Sum)):
				if j == 0:
					Coeff[j] -= (Sum[j] / Batch_Size) * Learning_Rate
				else:
					Coeff[j] = (1 - Learning_Rate * Lambda_Parameter / m) * Coeff[j] - (Sum[j] / Batch_Size) * Learning_Rate
print("Final coefficients are:")
print(Coeff)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_Test_Set)):
	prediction = 0
	for j in range(len(Coeff)):
	  	prediction = prediction + Coeff[j] * Features_Test_Set[i][j]
	Error += abs(prediction - PriceTest[i]) / PriceTest[i]
Error = (Error / len(Features_Test_Set)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()