
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

	def getErrors(self):					# Skapar funktion så vi kan hämta felen
		return self.errors

	"""
	Flyttar errors, och släcker ut som två errors möter varandra.
		@param
	Action: "u - 0" "d - 1" "l - 2" "r - 3"
	"""
	def moveError(self, action, errorIndex):            # Tar in ett fel och vilken action den ska ta

		firstPos = self.errors[errorIndex,:]            # Positionen för felet som skall flyttas
		secondPos = self.getPos(action, firstPos)       # Nya positionen för felet givet action och position

        # Uppdatera den gamla positionen
		self.state[firstPos[0],firstPos[1]] = 0
        # Uppdatera den nya positionen
		if self.state[secondPos[0], secondPos[1]] == 0:
			self.state[secondPos[0],secondPos[1]] = 1
		else:
			self.state[secondPos] = 0

		self.updateErrors()     						# Kolla igenom igen vart fel finns

	"""
	Returnerar positionen efter att ha rört sig i en viss riktning.
		@param
			action: den associerade riktningsrörelsen.
			position: positionen vi står vid innan vi flyttar oss.
		@return
			numpy: koordinater för nya positionen.
	"""
	def getPos(self, action, position):

		# Kopiera tidigare position så vi får ny pekare
		nextPos = np.array(position, copy=True)

		# Beroende på action väljs steg
		if action == 0:
			if nextPos[0] == 0:
				nextPos[0] = self.length - 1
			else:
				nextPos[0] -= 1
		if action == 1:
			if nextPos[0] == self.length - 1:
				nextPos[0] = 0
			else:
				nextPos[0] += 1
		if action == 2:
			if nextPos[1] == 0:
				nextPos[1] = self.length - 1
			else:
				nextPos[1] -= 1
		if action == 3:
			if nextPos[1] == self.length - 1:
				nextPos[1] = 0
			else:
				nextPos[1] += 1

		# Returnera nya positionen för felet
		return nextPos

	def getArray(self,errorIndex):
		a = np.zeros([self.length,self.length])
		a[errorIndex] = 1
		return a

"""
Mainmetod för att testa ovanstående klass.
"""
if __name__ == '__main__':

	# Här testas i princip ovanstående klass!
	S = np.zeros([3,3])
	S[0,0] = 1
	S[1,1] = 1
	S[2,2] = 1
	env = Env(S)
	print("Start state:")
	print(env.state)

	env.moveError(3,2)
	print("final state")
	print(env.state)




			
