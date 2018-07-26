import multiprocess as mp
import numpy as np
def changer(v):
	print('Initializing changer',v)
	a=1/v
		
class MainClass:
	def __init__(self):
		self.run()
	
	def run(self):
		m = mp.Manager()
		variable = m.list()
		variable.append(np.ones(1000000))
		
		p1 = mp.Process(target = changer, args=(0,))
		p2 = mp.Process(target = changer, args=(0,))
		
		p1.start()
		p2.start()
		while 1==1:
			a=1
if __name__ == '__main__':
	MainClass()