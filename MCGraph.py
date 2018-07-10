import numpy as np
from Env import Env
from itertools import groupby
import collections
from Blossom import Blossom
import time

class MCGraph:
	
	def __init__(self, state):
		self.state = np.copy(state)
		self.distances = dict()
		self.edgeList = list()
		self.errorIndex = dict()
		
		numMatchings = self.doublefactorial(self.getAmountOfErrors(self.state)-1)
		
		self.matchings = list()
		self.allMWPM = list()
		self.createGraph(self.state)
		
		
		
		
		while len(self.matchings) < numMatchings:
			self.createInitialMatching()
			
		
		matches = [k for k,v in groupby(sorted(self.matchings))]
		
		#print(matches)
		
		B = Blossom(state)
		B.readResult()
		minSteps = B.getCost()
		#print("minSteps: ", minSteps)
		for l in matches:
			tot = -sum(i for i in range(0,self.getAmountOfErrors(self.state)))
			for tup in l:
				tot += sum(tup)
			if tot == minSteps:
				self.allMWPM.append(l)
				
		
		"""
		for i in range(30):
			B = Blossom(state)
			self.allMWPM.append(B.readResult())
		
		
		MWPM = [k for k,v in groupby(sorted(self.allMWPM))]
		"""
		
		
		
	
	def createGraph(self, state):
		obs = Env(state).getObservation()
		originalErrorIndex = self.getErrorIndices(state)
		self.edgeList.clear()
		self.distances.clear()
		
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
				
	def getDistance(self, index1, index2):
		x1 = index1[0]
		y1 = index1[1]
		x2 = index2[0]
		y2 = index2[1]
		
		xdist = np.abs(x1-x2)
		ydist = np.abs(y1-y2)
		
		return xdist + ydist
		
		
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
		
	
	def createInitialMatching(self):
		tmp = list()
		tmp2 = list()
		acceptNewMatch = True
		
		for i in range( (self.getAmountOfErrors(self.state))):
			tmp.append(i)
			#dist = self.distances[str(2*i)+", "+str(2*i+1)]
			#tmp.append((2*i, 2*i+1, dist))
			
		while len(tmp) != 0:
			index1 = np.random.randint(0,len(tmp))
			match1 = tmp.pop(index1)
			index2 = np.random.randint(0,len(tmp))
			match2 = tmp.pop(index2)
			dist = self.distances[str(match1) +", " +str(match2)]
			if match1 > match2:
				swap_tmp = match1
				match1 = match2
				match2 = swap_tmp
			tmp2.append((match1, match2, dist))
		
		for l in self.matchings:
			counter = 0
			for el in tmp2:
				#print("l: ", l)
				#print("el: ", el)
				#print("el in l: ", el in l)
				if el in l:
					counter +=1
			#print(counter)
			if counter == len(tmp2):
				#print("New matching NOT accepted")
				acceptNewMatch = False
				
		if acceptNewMatch:
			#print("New matching accepted")
			#print("Adding new match: ", tmp2)
			#print("Matchings before: ", [k for k,v in groupby(sorted(self.matchings))])
			self.matchings.append(tmp2)
			#print("Matchings after: ", [k for k,v in groupby(sorted(self.matchings))])
			
		
		
	"""
	def computeMatching(self):
		previousMatching = self.matchings[len(self.matchings)-1].copy()
		
		
		#print(previousMatching)
		
		firstIndex = np.random.randint(0,int((self.getAmountOfErrors(self.state)/2)))
		secondIndex = np.random.randint(0,int((self.getAmountOfErrors(self.state)/2)))
		while firstIndex == secondIndex:
			secondIndex = np.random.randint(0,int((self.getAmountOfErrors(self.state)/2)))
		#print("first:",firstIndex)
		#print("second:",secondIndex)
		
		tup1 = previousMatching[firstIndex]
		tup2 = previousMatching[secondIndex]
		
		change1 = tup1[0]
		change1_ = tup2[1]
		
		old_dist = tup1[2]+tup2[2]
		dist1 = self.distances[str(change1) +", " +str(change1_)]
		
		change2 = tup1[1]
		change2_ = tup2[0]
		
		dist2 = self.distances[str(change2) +", " +str(change2_)]
		
		new_dist = dist1+dist2
	
		#print("new_dist: ", new_dist)
		#print("old_dist: ", old_dist)
		
		if new_dist <= old_dist:
			acceptNewMatch = True
			if change1 > change1_:
				tmp = change1
				change1 = change1_
				change1_ = tmp
			if change2 > change2_:
				tmp = change2
				change2 = change2_
				change2_ = tmp
			previousMatching[firstIndex] = (change1, change1_, dist1)
			previousMatching[secondIndex] = (change2, change2_, dist2)
			
			for l in self.matchings:
				counter = 0
				for el in previousMatching:
					if el in l:
						counter +=1
				if counter == len(previousMatching):
					#print("New matching NOT accepted in computeMatching()")
					acceptNewMatch = False
			
			if acceptNewMatch:
				self.matchings.append(previousMatching)
			#print("Matching complete")
		else:
			p = np.random.rand()
			p_threshold = 0.1
			if p<p_threshold:
				acceptNewMatch = True
				if change1 > change1_:
					tmp = change1
					change1 = change1_
					change1_ = tmp
				if change2 > change2_:
					tmp = change2
					change2 = change2_
					change2_ = tmp
				previousMatching[firstIndex] = (change1, change1_, dist1)
				previousMatching[secondIndex] = (change2, change2_, dist2)
				
				for l in self.matchings:
					counter = 0
					for el in previousMatching:
						if el in l:
							counter +=1
					if counter == len(previousMatching):
						#print("New matching NOT accepted in computeMatching()")
						acceptNewMatch = False
				
				if acceptNewMatch:
					self.matchings.append(previousMatching)
				#print(previousMatching)
				#self.matchings.insert(0,previousMatching)
				#print("Matching complete")
				self.matchings.append(previousMatching)	 
	"""
		
	def getMWPM(self):
		return self.allMWPM
		
	def doublefactorial(self, n):
		if n <= 0:
			return 1
		else:
			return n * self.doublefactorial(n-2)
		
if __name__ == '__main__':
	A = np.zeros((5,5))
	A[0,0] = 1
	A[0,1] = 2
	A[0,4] = 3
	A[1,1] = 4
	A[1,2] = 5
	A[2,0] = 6
	A[3,3] = 7
	A[3,4] = 8
	A[4,1] = 9
	A[4,2] = 10
	print(A)
	
	start = time.time()
	G = MCGraph(A)
	end = time.time()
	print("Elapsed time: ", end-start)
	#for key in G.distances.keys():
		#print("Key: ", key, " Value: ", G.distances[key])
	
