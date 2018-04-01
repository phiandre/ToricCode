
import math
import numpy as np

class Env:

	"""""""""""""""""""""""""""""""""""""""""""""""""""
	Konstruktor för klass Env.
		@param
			state: tar in initial statematris i numpy.
	"""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, state):
		# Spara viktiga matriser och variabler
		self.state = state
		self.length = state.shape[0]
		
		# Uppdatera platser där fel finns
		self.updateErrors()

	"""""""""""""""""""""""""""""""""""""""""""""""""""
	Hittar alla fel och uppdaterar matrisen errors där
	felens koordinater finns uppradade.
	"""""""""""""""""""""""""""""""""""""""""""""""""""
	def updateErrors(self):
		# Skapa tillfällig matris
		temp = np.zeros([self.length * self.length, 2], dtype=np.int8)
		x = 0
		#Kollar igenom tillståndet och sparar felen i en array
		for i in range(self.length):
			for j in range(self.length):
				if self.state[i, j] == 1:
					temp[x, 0:2] = [i, j]
					x += 1
		# Arrayen innehåller positionen för alla fel
		self.errors = temp[0:x, 0:2]

	"""""""""""""""""""""""""""""""""""""""
	Returnerar matris innehållande felen.
		@return
			numpy: matris innehållande fel.
	"""""""""""""""""""""""""""""""""""""""
	def getErrors(self):					
		return self.errors

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Flyttar errors, och släcker ut som två errors möter varandra.
	Actions följer: [u = 0, d = 1, l = 2, r = 3]
		@param
			action: rörelse som vi vill utföra.
			errorIndex: index till fel som vi vill flytta.
		@return
			int: reward, 10 för att ta bort, -1 för ingen skillnad.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def moveError(self, action, errorIndex):

		# Kolla antal errors innan
		amountErrors = len(self.errors)
		# Positionen för felet som skall flyttas
		firstPos = self.errors[errorIndex, :]
		# Nya positionen för felet givet action och position
		secondPos = self.getPos(action, firstPos)

		#  Uppdatera den gamla positionen
		self.state[firstPos[0], firstPos[1]] = 0
		# Uppdatera den nya positionen
		if self.state[secondPos[0], secondPos[1]] == 0:
			self.state[secondPos[0], secondPos[1]] = 1
		else:
			self.state[secondPos[0],secondPos[1]] = 0

		# Kolla igenom igen vart fel finns
		self.updateErrors()

		if amountErrors > len(self.errors):
			return 10

		return -1
		
	def centralize(self, error):
		# state är matrisen som karaktäriserar tillståndet
		# error är koordinaterna för felet
		state_=np.concatenate((self.state[:,error[1]:],self.state[:,0:error[1]]),1)
		state_=np.concatenate((state_[error[0]:,:],state_[0:error[0],:]),0)
		rowmid=int(np.ceil(self.state.shape[0]/2))
		colmid=int(np.ceil(self.state.shape[1]/2))
		state_=np.concatenate((state_[:,colmid:],state_[:,0:colmid]),1)
		state_=np.concatenate((state_[rowmid:,:],state_[0:rowmid,:]),0)
		return state_
	
	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Returnerar positionen efter att ha rört sig i en viss riktning.
		@param
			action: den associerade riktningsrörelsen.
			position: positionen vi står vid innan vi flyttar oss.
		@return
			numpy: koordinater för nya positionen.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Returnerar
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def getObservation(self):
		if len(self.errors)==0:
			return 'terminal'
		else:
			numerror=self.errors.shape[0]
			observation=np.zeros((self.length,self.length,numerror))
			for i in range(numerror):
				observation[:,:,i]=self.centralize(self.errors[i,:])

			return observation
	
"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod för att testa ovanstående klass.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':

	# Här testas i princip ovanstående klass!
	S = np.zeros([3, 3])
	S[0, 0] = 1
	S[1, 1] = 1
	S[2, 1] = 1
	S[0, 2] = 1
	env = Env(S)
	print("Start state:")
	print(env.state)

	env.moveError(1, 1)
	print("final state")
	print(env.state)
	env.getObservation()



			
