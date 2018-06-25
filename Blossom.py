import numpy as np
from Env import Env
from queue import Queue

class Pair:
	def __init__(self, a, b):
		self.nodeName = a
		self.nodeDistance = b
		
	
	def toString(self):
		return "<Node: " + str(self.nodeName)+ ", Dist: " + str(self.nodeDistance) + ">"
	
	def getNodeName(self):
		return self.nodeName
		
	def getNodeDistance(self):
		return self.nodeDistance	



class Node:
	def __init__(self, data):
		self.left = None
		self.right = None
		self.data = data



	
class Tree:
	
	def __init__(self):
		self.root = None
	
	def getRoot(self):
		return self.root
	
	def add(self, data):
		if(self.root == None):
			self.root = Node(data)
		else:
			self._add(data, self.root)
	
	def _add(self, data, node):
		if(data < node.data):
			if(node.left != None):
				self._add(data,node.left)
			else:
				node.left = Node(data)
		else:
			if(node.right != None):
				self._add(data,node.right)
			else:
				node.right = Node(data)
				
	def find(self, data):
		if (self.root != None):
			return self._find(data, self.root)
		else:
			return None
	
	def _find(self, data, node):
		if (data == node.data):
			return node
		elif (data < node.data and node.left != None):
			self._find(data, node.left)
		elif (data > node.data and node.right != None):
			self._find(data, node.right)
	
	def printTree(self):
		if(self.root != None):
			self._printTree(self.root)
	
	def _printTree(self, node):
		if(node != None):
			self._printTree(node.left)
			print( str(node.data) + ' ')
			self._printTree(node.right)
		
	

class Graph:
	
	def __init__(self):
		A = np.zeros((5,5))
		A[0,1] = 1
		A[1,1] = 1
		A[0,3] = 1
		A[0,4] = 1
		
		self.errors = self.getErrorIndices(A)
		A = self.changeErrorRep(A)
		
		env = Env(A)
		observation = env.getObservation()
		
		self.mapRep = dict()
	
		for i in range(observation.shape[2]):
			self.getErrorDistance(observation[:,:,i])
			
		
			


	def getErrorIndices(self, state):
		return np.transpose(np.nonzero(state))
	
		
	def changeErrorRep(self, state):
		n = 1
		for i in self.errors:
			state[i[0],i[1]] = n
			n += 1
		return state
	
	def getErrorDistance(self, state):
		errorIndices = self.getErrorIndices(state)
		errorList = list()
		
		middleIndex = int(np.floor(state.shape[0]/2))
		centerError = int(state[middleIndex,middleIndex])
		for error in errorIndices:
			xdist = np.abs(error[0] - middleIndex)
			ydist = np.abs(error[1] - middleIndex)
			
			errorDistance = np.abs(xdist+ydist)
			errorLabel = int(state[error[0],error[1]])
			
			if errorDistance > 0:
				errorList.append(Pair(errorLabel,errorDistance))
			
		self.mapRep[centerError] = errorList
	
	def printGraph(self):
		for key in self.mapRep.keys():
			neighbours = self.mapRep[key]
			l = list()
			for element in neighbours:
				l.append(element.toString())
			
			print(key, l)
	
	def getKeys(self):
		return self.mapRep.keys()
		
	def getNeighbours(self,key):
		return self.mapRep[key]
		
		
	
class Blossom:
	
	def __init__(self):
		g = Graph()
		self.MWPM(g)
		#self.treeTest()
	
	def treeTest(self):
		T = Tree()
		T.add(1)
		T.add(2)
		T.add(3)
		T.add(5)
		T.add(4)
		T.add(6)
		T.add(9)
		T.printTree()
	
	def MWPM(self, graph):
		G = graph
		Q = Queue()
		M = dict()
		F = set()
		for key in G.getKeys():
			F.add(key)
			
		
		while len(F) != 0:
			r = F.pop()
			Q.put(r)
			T = Tree()
			T.add(r)
			while not Q.empty():
				v = Q.get()
				neighbours = graph.getNeighbours(v)
				for neighbour in neighbours:
					node = neighbour.getNodeName()
					if (T.find(node) == None) and (node not in M):
						T.add(node)
						T.add(M(node))
						Q.put(M(node))
					elif (T.find(node) != None ):
						
						
					
			
			
			
		
		
		
	
		
if __name__ == '__main__':
	Blossom()
