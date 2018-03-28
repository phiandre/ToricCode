import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
import time


"""
size=5
flips=3

humRep,comRep = Generate.generateData(size,flips)
# humRep visar alla spin - ej tillgänglig för RL-hjärnan
# comRep visar var fel finns i torusen

E=Env(comRep)
# E är vår environment

observation=E.getObservation()
# observation är den observationstensor som skickas till RL-hjärnan
"""


"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	
	flip = np.arange(5)
	size=5
	actions=4

	rl = RLsys(4, size)
	
	humRep=np.load('ToricCodeHuman.npy')
	comRep=np.load('ToricCodeComputer.npy')
	print(comRep[:,:,3])
	
	
	for i in range(comRep.shape[2]):
		state=comRep[:,:,i]
		env = Env(state)
		
		
		while len(env.getErrors()) > 0:
			print('Bana nummer ' + str(i))
			print(state)
			observation = env.getObservation()
			a, e = rl.choose_action(observation)
			r = env.moveError(a, e)
			new_observation = env.getObservation()
			
			rl.learn(observation[:,:,e], a, r, new_observation)
	for i in range(10)
		while len(env.getErrors()) > 0:
			print('Bana nummer ' + str(i))
			print(state)
			observation = env.getObservation()
			a, e = rl.choose_action(observation)
			r = env.moveError(a, e)
			new_observation = env.getObservation()
			time.sleep(2)
				
