import numpy as np

A = np.zeros((25,25))
A[0,:] = 1
A[2,:] = 2
A[3,:] = 4
A[:,3] = 5

newA = np.zeros((5,5))
for i in range(5):
	for j in range(5):
		B = A[5*i:(5*i+5),5*j:(5*j+5)]
		numErrors = len(np.transpose(np.nonzero(B)))
		if numErrors%2!=0:
			newA[i,j] = 1

print(newA)

