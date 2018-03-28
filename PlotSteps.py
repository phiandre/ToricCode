import matplotlib.pyplot as plt
import numpy as np

data = np.load('steps.npy')

averageOver = 100
plotData = np.zeros(round(len(data)/averageOver))

for i in range(round(len(data)/averageOver)):
	plotData[i] = np.average(data[i*averageOver:i*averageOver+(averageOver-1)])
	
plt.plot(np.arange(len(plotData)),plotData)
plt.show()
