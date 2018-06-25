import numpy as np
from EnvAlt import Env

tuple = (6,1)
rowSteps = -1
colSteps = -2
comRep=np.load('ToricCodeComputerTest.npy')
humRep=np.load('ToricCodeHumanTest.npy')
state = comRep[:,:,0]
humState = humRep[:,:,0]
env = Env(state, humState)
env.cancelTuple(tuple,rowSteps,colSteps)
