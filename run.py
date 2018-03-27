import numpy as np
from Env import Env
from GenerateToricData import Generate



humRep,comRep = Generate.generateData(5,3)
E=Env(comRep)

print(humRep)
print(comRep)


errors=E.getErrors()


def centralize(state,error):
	# state är matrisen som karaktäriserar tillståndet
	# error är koordinaterna för felet
	state_=np.concatenate((state[:,error[1]:],state[:,0:error[1]]),1)
	state_=np.concatenate((state_[error[0]:,:],state_[0:error[0],:]),0)
	rowmid=int(np.ceil(state.shape[0]/2))
	colmid=int(np.ceil(state.shape[1]/2))
	state_=np.concatenate((state_[:,colmid:],state_[:,0:colmid]),1)
	state_=np.concatenate((state_[rowmid:,:],state_[0:rowmid,:]),0)
	return state_

