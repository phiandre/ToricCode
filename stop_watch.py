import time

class StopWatch:
	def __init__(self):
		self.Start = time.time()
		self.Stop = time.time()
		self.IntervalStart = time.time()
		self.IntervalStop = time.time()
		self.Intervals =  [0]
		self.n = 0
		self.passedTime = 0
		self.intervalTime = 0
		self.currentInterval = 0
		self.stopped = False
		
	def start(self):
		self.Start = time.time()
		self.Stop = time.time()
		self.IntervalStart = time.time()
		self.IntervalStop = time.time()
		self.Intervals =  [0]
		self.n = 0
		self.passedTime = 0
		self.intervalTime = 0
		self.currentInterval = 0
		self.stopped = False
		
	def stop(self):
		if not self.stopped:
			self.Stop = time.time()
			self.passedTime = self.Stop-self.Start
			self.intervalTime = self.Stop - self.IntervalStart
			self.Intervals[self.currentInterval] = self.intervalTime
			self.stopped = True
			self.IntervalStop = time.time()
			return self.passedTime
		else:
			print("Can't stop, since already stopped or paused!")
			print("Returning time passed before stopping occured.")
			return self.passedTime
		
	def interval(self):
		if not self.stopped:
			self.IntervalStop = time.time()
			intervalTime = self.IntervalStop - self.IntervalStart
			self.IntervalStart = time.time()
			self.Intervals[self.currentInterval] = intervalTime
			self.Intervals.append(0)
			self.currentInterval +=1
			self.n += 1
			return intervalTime
		else:
			print('Returning time of interval before stopping occured.')
			print('New interval will begin when unpaused.')
			intervalTime = self.Intervals[self.currentInterval]
			self.Intervals.append(0)
			self.currentInterval += 1
			self.n += 1
			self.intervalTime = 0
			return intervalTime
		
	def average(self):
		return sum(self.Intervals)/self.n
	
	def pause(self):
		if not self.stopped:
			self.passedTime = time.time() - self.Start
			self.intervalTime = time.time()- self.IntervalStart
			self.stopped = True
			return self.passedTime
		else:
			print("Can't pause, since already stopped or paused!")
			print("Returning time passed before stopping occured.")
			return self.passedTime
		
	
	def unpause(self):
		self.Start = time.time() - self.passedTime
		self.IntervalStart = time.time() - self.intervalTime
		self.stopped = False
	def totalTime(self):
		return sum(self.Intervals)