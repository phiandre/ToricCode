
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

		for i in range(self.length):		#Kollar igenom tillståndet och sparar felen i en array
			for j in range(self.length):
				if self.state[i,j] == 1:
					temp[x,0:2] = [i, j]
					x += 1

		self.errors = temp[0:x,0:2]			# Arrayen innehåller positionen för alla fel


	"""
	Flyttar errors, och släcker ut som två errors möter varandra.
		@param
	Action: "u - 0" "d - 1" "l - 2" "r - 3"
	"""
	def moveError(self, action, errorIndex):   			# Tar in ett fel och vilken action den ska ta

		firstPos = self.errors[errorIndex, :]			# Positionen för felet som skall flyttas
		secondPos = self.getPos(action, firstPos)		# Nya positionen för felet givet action och position
		print("2ndPos")
		print(secondPos)
		self.state[firstPos] = 0
		print("secondPos")
		print(secondPos)
		if self.state[secondPos[0],secondPos[1]] == 0:
			self.state[secondPos] = 1
		else:
			self.state[secondPos] = 0

		print("newstate")
		print(self.state)


		self.updateErrors()									# Kolla igenom igen vart fel finns

	def getPos(self, action, position):					# Input: vilken action, felets position

		nextPos = np.array(position, copy=True)
		print("nextpos")
		print(nextPos)
		if action == 0:									# Beroende på action väljs steg
			nextPos[0] -= 1
			if nextPos[0] == 0:
				nextPos[0] = self.length			#Flyttar genom väggen
		if action == 1:
			nextPos[0] += 1
			if nextPos[0] == self.length:
				nextPos[0] = 0
		if action == 2:
			nextPos[1] -= 1
			if nextPos[1] == 0:
				nextPos[1] = self.length
		if action == 3:
			nextPos[1] += 1
			if nextPos[1] == self.length:
				nextPos[1] = 0
		print("newpos")
		print(nextPos)
		return nextPos									# Output: nya positionen för felet

if __name__ == '__main__':

	S = np.zeros([3,3])
	S[0,0] = 1
	S[1,1] = 1
	S[2,1] = 1
	env = Env(S)
	print("Start state:")
	print(env.state)

	env.moveError(1,1)
	print("final state")
	print(env.state)




