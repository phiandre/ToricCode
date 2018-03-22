



class Env:

	"""
	Konstruktor för klass Env.
		@param
			state: tar in initial statematris.
			action_space: mängden av tillåtna actions.
	"""
	def __init__(self, state, action_space):

		
		self.state = state
		self.length = len(state)


	def updateErrors(self):


		self.errors
