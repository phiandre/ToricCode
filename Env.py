
import math
import numpy as np

class Env:

	"""
	Konstruktor för klass Env.
		@param
			state: tar in initial statematris i numpy.
	"""
	def __init__(self, state):

		self.state = state
		self.length = state.shape[0]

		self.updateErrors()


	"""
	Hittar alla fel och uppdaterar matrisen errors där felens
	koordinater finns uppradade.
	"""
	def updateErrors(self):

		temp = np.zeros([self.length * self.length, 2], dtype=np.int8)
		x = 0

		for i in range(self.length):
			for j in range(self.length):
				if self.state[i,j] == 1:
					temp[x,0:2] = [i, j]
					x += 1

		self.errors = temp[0:x,0:2]

	"""
	Flyttar errors, och släcker ut som två errors möter varandra.
		@param
	Action: "u - 0" "d - 1" "l - 2" "r - 3"
	"""
	def moveError(self, action, errorIndex):

		firstPos = self.errors[errorIndex, :]
		secondPos = self.getPos(action, firstPos)
		print(secondPos)
		self.state[firstPos] = 0
		print(self.state)
		print(secondPos)
		self.state[secondPos] = 1 - self.state[secondPos]
		print(self.state)

		self.updateErrors()

	def getPos(self, action, position):

		nextPos = np.array(position, copy=True)
		print(nextPos)
		if action == 0:
			nextPos[0] -= 1
			if nextPos[0] == 0:
				nextPos[0] = self.length - 1
		if action == 1:
			nextPos[0] += 1
			if nextPos[0] == self.length - 1:
				nextPos[0] = 0
		if action == 2:
			nextPos[1] -= 1
			if nextPos[1] == 0:
				nextPos[1] = self.length - 1
		if action == 3:
			nextPos[1] += 1
			if nextPos[1] == self.length - 1:
				nextPos[1] = 0

		return nextPos

if __name__ == '__main__':

	S = np.zeros([3,3])
	S[0,0] = 1
	S[1,1] = 1
	S[2,1] = 1
	env = Env(S)

	print(env.state)

	env.moveError(1,1)

	print(env.state)



