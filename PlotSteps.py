import matplotlib.pyplot as plt
import numpy as np

#ladda in datafilen 
data = np.load('steps.npy')

#ange över hur många värden datan medelvärdesbildas
averageOver = 100


plotData = np.zeros(round(len(data)/averageOver))

for i in range(round(len(data)/averageOver)):
	plotData[i] = np.average(data[i*averageOver:i*averageOver+(averageOver-1)])
	
plt.plot(np.arange(len(plotData)),plotData) # TODO: Fixa så grafen är över hela intervallet
plt.show()
