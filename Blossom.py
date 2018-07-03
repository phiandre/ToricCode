from os import system
from Env import Env
import numpy as np
import os.path

class Blossom:
	
	"""
	The constructor creates a graph representation of an observation
			@param
				obs: an env.getObservation() of a state
	"""
	def __init__(self, state):
		self.state = np.copy(state)
		self.inputFile = 'Blossom\\state_graph.txt'
		self.outputFile = 'Blossom\\result.txt'
		
		
		self.distances = dict()
		self.edgeList = list()
		self.errorIndex = dict()
		
		self.createGraph(state)
		
		
	def createEuclidianGraph(self, state):
		obs = Env(state).getObservation()
		originalErrorIndex = self.getErrorIndices(state)
		amountOfErrors = self.getAmountOfErrors(obs[:,:,0])
		amountOfEdges = 2*(np.sum(i for i in range(0,amountOfErrors)))
		self.edgeList.clear()
		self.distances.clear()
		self.edgeList.append(str(amountOfErrors) + " " + str(amountOfEdges))
		
	
		for i in range(obs.shape[2]):
			state = obs[:,:,i]
			#print("state:\n", state)
			self.errorIndex[i+1] = originalErrorIndex[i]
			errors = self.getErrorIndices(state)
			
			currentError = np.array((int(np.floor(obs.shape[0]/2)), int(np.floor(obs.shape[0]/2)))) #index of center position
			
			for error in errors:
				errorNumber = int(state[error[0],error[1]])
				centerNumber = int(state[currentError[0],currentError[1]])
				if errorNumber == centerNumber:
					continue
				dist = self.getEuclidianDistance(currentError, error)
				
				#print("centerNumber-1: ", centerNumber-1)
				#print("errorNumber-1: ", errorNumber-1)
				#print("dist: ", dist)
				self.distances[str(centerNumber-1) +", " +str(errorNumber-1)] = dist
				self.edgeList.append(str(centerNumber-1) + " " + str(errorNumber-1) + " " + str(dist))
		self.createGraphAsTxt(self.edgeList)
		self.computeMWPM()
			
			
		
	
	def createGraph(self, state):
		obs = Env(state).getObservation()
		originalErrorIndex = self.getErrorIndices(state)
		amountOfErrors = self.getAmountOfErrors(obs[:,:,0])
		amountOfEdges = 2*(np.sum(i for i in range(0,amountOfErrors)))
		self.edgeList.clear()
		self.distances.clear()
		self.edgeList.append(str(amountOfErrors) + " " + str(amountOfEdges))
		
	
		for i in range(obs.shape[2]):
			state = obs[:,:,i]
			#print("state:\n", state)
			self.errorIndex[i+1] = originalErrorIndex[i]
			errors = self.getErrorIndices(state)
			
			currentError = np.array((int(np.floor(obs.shape[0]/2)), int(np.floor(obs.shape[0]/2)))) #index of center position
			
			for error in errors:
				errorNumber = int(state[error[0],error[1]])
				centerNumber = int(state[currentError[0],currentError[1]])
				if errorNumber == centerNumber:
					continue
				dist = self.getDistance(currentError, error)
				
				#print("centerNumber-1: ", centerNumber-1)
				#print("errorNumber-1: ", errorNumber-1)
				#print("dist: ", dist)
				self.distances[str(centerNumber-1) +", " +str(errorNumber-1)] = dist
				self.edgeList.append(str(centerNumber-1) + " " + str(errorNumber-1) + " " + str(dist))
		self.createGraphAsTxt(self.edgeList)
		self.computeMWPM()
		
		"""
		for i in range(obs.shape[2]):
			state = obs[:,:,i]
			#state = self.labelState(s,obs.shape[0])
			print("labeled state:\n", state)
			self.edgeList.clear()
			self.distances.clear()
			self.edgeList.append(str(amountOfErrors) + " " + str(amountOfEdges))
			
			env = Env(state)
			observation = env.getObservation()
			for j in range(observation.shape[2]):
				newState = observation[:,:,j]
				errors = self.getErrorIndices(newState)
				currentError = np.array((int(np.floor(obs.shape[0]/2)), int(np.floor(obs.shape[0]/2)))) #index of center position
				for error in errors:
					errorNumber = int(newState[error[0],error[1]])
					centerNumber = int(newState[currentError[0],currentError[1]])
					if errorNumber == centerNumber:
						continue
					dist = self.getDistance(currentError, error)
					
					#print("centerNumber-1: ", centerNumber-1)
					#print("errorNumber-1: ", errorNumber-1)
					#print("dist: ", dist)
					self.distances[str(centerNumber-1) +", " +str(errorNumber-1)] = dist
					self.edgeList.append(str(centerNumber-1) + " " + str(errorNumber-1) + " " + str(dist))
					
					
					if errorNumber <= (i+1):
						continue
					else:
						dist = self.getDistance(currentError, error)
						self.distances[str(i) +", " +str(errorNumber-1)] = dist
						edgeList.append(str(i) + " " + str(errorNumber-1) + " " + str(dist))
					
			self.createGraphAsTxt(self.edgeList)
			self.computeMWPM()
			self.readResult()
		"""
	
		
	"""
	Returns the amount of errors present i.e. the amount of nodes in the graph.
	This is required by the blossom implementation.
		@param
			state: a matrix representation of a state
	"""
	def getAmountOfErrors(self, state):
		return len(self.getErrorIndices(state))
		
	"""
	Returns indices of the errors in the state
		@param
			state: a matrix representation of a state
	"""
	def getErrorIndices(self, state):
		return np.transpose(np.nonzero(state))
	
	"""
	Calculates the distance between two errors, which corresponds to the weights
	in the graph
		@param
			index1: index of error 1 
			index2: index of error 2
		
		@return
			int: distance between error 1 and error 2
	"""
	def getDistance(self, index1, index2):
		x1 = index1[0]
		y1 = index1[1]
		x2 = index2[0]
		y2 = index2[1]
		
		xdist = np.abs(x1-x2)
		ydist = np.abs(y1-y2)
		
		return xdist + ydist
	
	def getEuclidianDistance(self, index1, index2):
		x1 = index1[0]
		y1 = index1[1]
		x2 = index2[0]
		y2 = index2[1]
		
		xdist = np.abs(x1-x2)
		ydist = np.abs(y1-y2)
		
		return xdist**2 + ydist**2
		
	"""
	Save the graph representation of the state as a txt-file, suitable for the Blossom algorithm file.
		@param
			edgeList: a list of all edges with corresponding weights, which are to be
					added to the txt file
	"""
	def createGraphAsTxt(self, edgeList):
		if os.path.isfile(self.inputFile):
			open(self.inputFile, 'w').close()
			
		with open(self.inputFile, 'a') as f:
			for l in edgeList:
				f.write(l+"\n")
	
		
	"""
		Execute C++ implementation of the Blossom algorithm to compute a MWPM 
	"""
	def computeMWPM(self):
		system("Blossom\\blossom5.exe -e " + str(self.inputFile) + " -w " + str(self.outputFile) +" -V")
	
	"""
	The resulting txt-file is returned as a list of tuples where each tuple contains
	the matched nodes and the distance between them
			
		@return
			tuple: The MWPM represented as a list of tuples [(node1, node2, distance)]
	"""
	def readResult(self):
		l = list()
		with open(self.outputFile, 'r') as f:
			next(f)
			for line in f:
				first_node = line.split(' ')[0]
				second_node = line.split(' ')[1].strip('\n')
				#print("first_node: ", first_node)
				#print("second_node: ", second_node)
				matching_node_1 = self.errorIndex[int(first_node)+1]
				matching_node_2 = self.errorIndex[int(second_node)+1]
				
				l.append( (int(first_node)+1, matching_node_1, int(second_node)+1, matching_node_2, int(self.distances[first_node+ ", "+  second_node])))

		
		return l		
		
if __name__ == '__main__':
	A = np.zeros((5,5))
	A[1,1] = 1
	A[1,3] = 1
	A[2,2] = 1
	A[2,4] = 1
	A[3,1] = 1
	A[4,2] = 1
	
	
	Blossom(A)
