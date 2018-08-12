import numpy as np
import random
import os
import time
import numbers
from stop_watch import StopWatch
class MineSweeperEnv:
	def __init__(self, mines = 10,height = 9,width = 9, god = False, human = False):
		self.God = god
		self.mines = mines
		self.height = height
		self.width = width
		whiteBlock = chr(9608)
		#self.view = np.zeros((self.height,self.width))*(-1)
		self.view = []
		self.firstRow = [str(colNum) for colNum in range(1,width)] + [str(width)+')']
		for i in range(self.height):
			tmpRow = [str(i+1)+')']
			for j in range(self.width):
				tmpRow.append(whiteBlock*2)
			self.view.append(tmpRow)
		
		self.firstClick = True
		
		
		
	def startGame(self,roww,col):
		whiteBlock = chr(9608)
		self.firstClick = False
		self.mineSet = set()
		self.grid = np.zeros((self.height,self.width))
		#self.view = np.zeros((self.height,self.width))*(-1)
		
		self.view = []
		for i in range(self.height):
			tmpRow = [str(i+1)+')']
			for j in range(self.width):
				tmpRow.append(whiteBlock*2)
			self.view.append(tmpRow)
		
		self.score = 0
		row_range = np.linspace(0,self.height,num=self.height,endpoint = False)
		column_range = np.linspace(0,self.width,num=self.width,endpoint = False)
		
		corners_rows = [0, 0, self.height, self.height]
		corner_columns = [0, self.width, 0, self.width]
		self.visible = set()
		
		self.vacant = width * height - mines
		
		
		added = 0
		while added != self.mines:
			row = int(np.random.choice(row_range))
			column = int(np.random.choice(column_range))
			if (self.grid[row,column] != 1) and not (row == roww and col == column):
				self.grid[row,column] = 1
				self.mineSet.add((row,column))
				added += 1
		self.updateView()
	
	def updateView(self):
		os.system('CLS')
		if self.God and not self.firstClick:
			print('Grid:\n',self.grid)
		
		print('\nView:\n'+ '   ' + ') '.join(element for element in self.firstRow) + '\n' +'\n\n'.join(' '.join(element for element in row) for row in self. view))
	
	
	def click(self,row,col,proper = True):

		if self.firstClick:
			self.startGame((row-1),(col-1))
		if (row,col) in self.visible:
			self.updateView()
			return True, self.score
		
		if (col > self.width) or (col<0) or (row > self.height) or (row <0) :
			self.updateView()
			print('Out of bounds!')
			return True, self.score

		
		
		gridCol = col - 1
		gridRow = row - 1
		
		if (gridRow,gridCol) in self.mineSet:
			#self.God = True
			self.updateView()
			print("\nYou're dead!")
			return False, self.score
		self.visible.add((row,col))
		
		windowRowStart = gridRow
		windowRowEnd = gridRow
		
		windowColStart = gridCol
		windowColEnd =gridCol
		
		while (windowRowStart > 0) and (abs(windowRowStart-gridRow)<1):
			windowRowStart -= 1
		
		while (windowRowEnd < self.height) and (abs(windowRowEnd-gridRow)<2):
			windowRowEnd += 1
		
		while (windowColStart > 0) and (abs(windowColStart-gridCol)<1):
			
			windowColStart -= 1
		
		while (windowColEnd < self.height) and (abs(windowColEnd-gridCol)<2):
			windowColEnd += 1
		
		window = self.grid[windowRowStart:windowRowEnd,windowColStart:windowColEnd]
		neighborMines = np.count_nonzero(window)
		self.view[gridRow][col] = ' ' + str(neighborMines)
		
		
		if neighborMines == 0:
			
			xs = windowRowStart
			ys = windowColStart
			
			x=xs
			y=ys
			
			while x <= windowRowEnd-1:
				y=ys
				x += 1
				while y <= windowColEnd-1:
					y += 1
					if not ((x,y)in self.visible):
						self.click(x,y,proper = False)
		self.score = len(self.visible)
		if self.score == self.vacant:
			#self.God = True
			self.updateView()
			print('You won!')
			return False, self.score
		
		if proper:
			self.updateView()
		return True, self.score
					
					
				
			
		
		
		
		
if __name__ == '__main__':
	playing = True
	while playing:
		os.system('CLS')
		alive = True
		height = 9
		width = 9
		mines = 10
		stage = MineSweeperEnv(mines = mines, height = height, width = width, god = False)
		vacant = height * width - mines
		filename = 'HighScore/HighScore.npy'
		high_score = int(np.load(filename))
		SW = StopWatch()
		started = False
		invalid = False
		while alive:
			xChosen = False
			yChosen = False
			while not xChosen:
				if not started:
					stage.updateView()
					if invalid:
						print('Invalid')
					invalid = True
				x = input("Choose row:\n")
				xChosen = x.isdigit()
				if xChosen:
					x=int(x)
				else:
					stage.updateView()
					print('Invalid')
			stage.updateView()
			while not yChosen:
				y = input("Choose column:\n")
				yChosen = y.isdigit()
				if yChosen:
					y=int(y)
				else:
					stage.updateView()
					print('Invalid')
			
			if not started:
				started = True
				SW.start()
			alive, score = stage.click(x,y)
		
		timer = int(SW.stop())
		if vacant ==score:
			if timer < high_score:
				print("NEW HIGH SCORE!\n")
				high_score = timer
				np.save(filename, high_score)
			print('Finishing time: ' + str(timer) + ' seconds')
			print('High Score: ' + str(high_score) + ' seconds')
		input('Enter anything to continue!\n')
		os.system('CLS')
		answered = False
		while not answered:
			answer = input("Wan't to play again? (y/n)\n")
			if answer == 'n':
				os.system('CLS')
				print('Good Bye!')
				playing = False
				answered = True
			elif answer == 'y':
				answered = True
			elif answer != 'y':
				print('Invalid answer!')