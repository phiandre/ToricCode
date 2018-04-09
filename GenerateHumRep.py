import numpy as np
import random
import pickle

size=10
numerrors=4 #måste vara jämnt antal fel!
humanRepresentation=np.zeros((size,size))
print(humanRepresentation)
for i in range(numerror):
	x=0
	while x=0:
		r=random.randint(0,(size-1))
		c=random.randint(0,(size-1))
		if humanRepresentation[r,c]==0:
			humanRepresentation[r,c]=1
			x=1

#for i in range(numerrors):
#	np.random

#np.random.choice(humanRepresentation)