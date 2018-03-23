import numpy as np
import random


class Generate:
	@staticmethod
	def generateData(size, numFlips):
		humanRepresentation, computerRepresentation = intialize(size)
		
		
		
		for iteration in range(0,numFlips):
			rowIndex, columnIndex = flipRandomIndex(2*size)
			while (humanRepresentation[(rowIndex, columnIndex)]== -1):
				rowIndex, columnIndex = flipRandomIndex(2*size)
			#print("human: " + str(rowIndex) +"," + str(columnIndex))
			humanRepresentation[(rowIndex, columnIndex)] = -1*humanRepresentation[(rowIndex, columnIndex)];
			
			#print("computer: " + str(computerRowIndex) +"," +str(computerColumnIndex))
			
			computerRepresentation = updateComputerRepresentation(rowIndex, columnIndex, size, computerRepresentation)
			
				
		print(humanRepresentation)
		print(computerRepresentation)	
		#humanRepresentation[]
		
		
		return humanRepresentation, computerRepresentation
		
	def intialize(size):
		humanRepresentation = np.zeros((2*size,2*size))
		for i in range(0,2*size):
			if i%2==0:
				for j in range(0,2*size):
					if j%2==1:
						humanRepresentation[i,j] = 1
					
			else:
				for j in range(0,2*size):
					if j%2==0:
						humanRepresentation[i,j] = 1
		computerRepresentation = np.ones((size,size))
		return humanRepresentation, computerRepresentation
		

			

	def updateComputerRepresentation(humanRowIndex, humanColumnIndex, size, rep):
		computerRepresentation = rep
		rowIndex = int(np.floor(humanRowIndex / 2))
		columnIndex = int(np.floor(humanColumnIndex / 2))
		
		print(rowIndex)
		print(columnIndex)
		if humanRowIndex == 0:
			computerRepresentation[(rowIndex, columnIndex)] = (-1)*computerRepresentation[(rowIndex, columnIndex)]
			computerRepresentation[(size-1, columnIndex)] = (-1)*computerRepresentation[(size-1, columnIndex)]
		elif humanColumnIndex == 0:
			computerRepresentation[(rowIndex, columnIndex)] = (-1)*computerRepresentation[(rowIndex, columnIndex)]
			computerRepresentation[(rowIndex, size-1)] = (-1)*computerRepresentation[(rowIndex, size-1)]
		elif humanRowIndex % 2 == 1:
			computerRepresentation[(rowIndex,columnIndex)] = (-1)*computerRepresentation[(rowIndex,columnIndex)]
			computerRepresentation[(rowIndex,columnIndex-1)] = (-1)*computerRepresentation[(rowIndex,columnIndex-1)]
		else:
			computerRepresentation[(rowIndex,columnIndex)] = (-1)*computerRepresentation[(rowIndex,columnIndex)]
			computerRepresentation[(rowIndex-1,columnIndex)] = (-1)*computerRepresentation[(rowIndex-1,columnIndex)]
			
		return computerRepresentation
		
	
	
	def flipRandomIndex(size):
		rowIndex = random.randint(0,size-1)
		columnIndex = random.randint(0,size-1)
		
		while rowIndex % 2 == 1 and columnIndex % 2 == 1:
			columnIndex = random.randint(0,size-1)
		while rowIndex % 2 == 0 and columnIndex % 2 == 0:
			columnIndex = random.randint(0,size-1)
		return rowIndex , columnIndex


if __name__ == '__main__':
	size = 5
	numFlips = 6
	Generate.generateData(size,numFlips)
	


