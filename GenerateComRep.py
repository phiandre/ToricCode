import numpy as np
import random
import pickle

size=10
numerror=4 #måste vara jämnt antal fel!
iterations=10000
computerRepresentation=np.zeros((size,size,iterations))
for iter in range(iterations):
	for i in range(numerror):
		x=0
		while x==0:
			r=random.randint(0,(size-1))
			c=random.randint(0,(size-1))
			if computerRepresentation[r,c,iter]==0:
				computerRepresentation[r,c,iter]=1
				x=1

np.save('ComputerData',computerRepresentation)