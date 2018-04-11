import numpy as np
import random
import pickle


size=10 #storlek på gitter
numerror=4 #antal fel, måste vara jämnt antal fel!
iterations=10000 #antal datapunkter som genereras
computerRepresentation=np.zeros((size,size,iterations))

for iter in range(iterations):
	for i in range(numerror):
		x=0
		while x==0:
			r=random.randint(0,(size-1)) #slumpa radindex
			c=random.randint(0,(size-1)) #slumpa kolumnindex
			
			#Kolla så att ett fel inte redan finns i plaketten, annars
			#slumpas indexen om. 
			if computerRepresentation[r,c,iter]==0: 
				computerRepresentation[r,c,iter]=1
				x=1

np.save('ComputerData',computerRepresentation)
