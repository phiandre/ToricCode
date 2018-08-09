from os import system
from BlossomEnv import Env
import numpy as np
import os.path
from random import shuffle
from itertools import groupby
import time

class Blossom:
	
	"""
	The constructor creates a graph representation of an observation
			@param
				obs: an env.getObservation() of a state
	"""
	def __init__(self, state):
		self.state = state
		self.inputFile = 'MAC_blossom/state_graph.txt'
		self.outputFile = 'MAC_blossom/result.txt'
		
		
		self.distances = dict()
		self.edgeList = list()
		self.errorIndex = dict()
		self.cost = 0
		
		self.createGraph(state)
		self.readResult()
		
		
		
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
			
			
	def labelState(self, s, size):
		state = s
		label = 1
		for j in range(size):
			for k in range(size):
				if state[j,k] == 1:
					state[j,k] = label
					label +=1
		return state
	
	def createGraph(self, state):
		#print("State\n", state)
		obs = Env(state).getObservation()
		originalErrorIndex = self.getErrorIndices(state)
		#print("state\n", obs[0][0,:,:])
		amountOfErrors = self.getAmountOfErrors(obs[:,:,0])
		#print("amountofErrors: ", amountOfErrors)
		amountOfEdges = 2*(np.sum(i for i in range(0,amountOfErrors)))
		#print("amountOfEdges: ", amountOfEdges)
		self.edgeList.clear()
		self.distances.clear()
		#self.edgeList.append(str(amountOfErrors) + " " + str(amountOfEdges))
		
	
		for i in range(obs.shape[2]):
			state = obs[:,:,i]
			#print("state before\n", state_)
			#state = self.labelState(np.copy(state_), state_.shape[0])
			#print("state:\n", state)
			self.errorIndex[i+1] = originalErrorIndex[i]
			errors = self.getErrorIndices(state)
			#print("errors", errors)
			currentError = np.array((int(np.floor(obs.shape[0]/2)), int(np.floor(obs.shape[0]/2)))) #index of center position
			#print("currentError", currentError)
			for error in errors:
				errorNumber = int(state[error[0],error[1]])
				centerNumber = int(state[currentError[0],currentError[1]])

				#print("errorNumber: ", errorNumber)
				#print("centerNumber: ", centerNumber)
				if errorNumber == centerNumber:
					continue
				dist = self.getDistance(currentError, error)
				
				#print("centerNumber-1: ", centerNumber-1)
				#print("errorNumber-1: ", errorNumber-1)
				#print("dist: ", dist)
				self.distances[str(centerNumber-1) +", " +str(errorNumber-1)] = dist
				self.edgeList.append(str(centerNumber-1) + " " + str(errorNumber-1) + " " + str(dist))
				
		#shuffle(self.edgeList)
		self.edgeList.insert(0, str(amountOfErrors) + " " + str(amountOfEdges))

		#print("edgeList:", self.edgeList)
		#print("distances: ", self.distances)
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
		print("In here")
		system("MAC_blossom/blossom5 -e " + str(self.inputFile) + " -w " + str(self.outputFile) +" -V")
	
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
				
				
				#The following may be appended if needed
				
				#matching_node_1 = self.errorIndex[int(first_node)+1]
				#matching_node_2 = self.errorIndex[int(second_node)+1]
				#dist = int(self.distances[first_node+ ", "+  second_node])
				self.cost += int(self.distances[first_node+ ", "+  second_node])
				l.append( (int(first_node), int(second_node)))

		
		return l		
		
	def getCost(self):
		return self.cost
		
if __name__ == '__main__':
	comRep = np.load('ToricCodeComputer.npy')
	start = time.time()
	for i in range(comRep.shape[2]):
		tmp = time.time()
		state = np.copy(comRep[:,:,i])
		Blossom(state)
		tmp_fin = time.time()
		print("Time elapsed since last completion:", tmp_fin-tmp)
		print("Average: ", (tmp_fin-start)/(i+1))
	finish = time.time()
	print("Total time elapsed: ", finish-start)
	print("Time per state: ", (finish-start)/100)

