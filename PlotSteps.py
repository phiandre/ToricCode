import matplotlib.pyplot as plt
import numpy as np

data = np.load('steps.npy')

plotData = np.zeros(round(len(data)/100))

for i in range(round(len(data)/100)):
	plotData[i] = np.average(data[i*100:i*100+99])
	
plt.plot(np.arange(len(plotData)),plotData)
plt.show()
