import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate


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
	
	humRep, state = Generate.generateData(size, 1)
	env = Env(state)

	while len(env.getErrors()) > 0:
		print(state)
		observation = env.getObservation()
		a, e = rl.choose_action(observation)
		r = env.moveError(a, e)
			
