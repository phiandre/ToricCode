import matplotlib.pyplot as plt
import numpy as np

data = np.load('PlotThis.npy')


	
plt.plot(5+2*np.arange(len(data)),data) # TODO: Fixa så grafen är över hela intervallet
plt.show()
