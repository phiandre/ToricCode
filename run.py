import numpy as np
from Env import Env
from GenerateToricData import Generate

size=5
flips=3

humRep,comRep = Generate.generateData(size,flips)
# humRep visar alla spin - ej tillgänglig för RL-hjärnan
# comRep visar var fel finns i torusen

E=Env(comRep)
# E är vår environment

observation=E.getObservation()
# observation är den observationstensor som skickas till RL-hjärnan

