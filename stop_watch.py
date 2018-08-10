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
		
	def stop(self):
		self.Stop = time.time()
		self.passedTime = self.Stop-self.Start
		self.intervalTime = self.Stop - self.IntervalStart
		self.Intervals[self.currentInterval] = self.intervalTime
		return self.passedTime
		
	def interval(self):
		self.IntervalStop = time.time()
		intervalTime = self.IntervalStop - self.IntervalStart
		self.IntervalStart = time.time()
		self.Intervals[self.currentInterval] = intervalTime
		self.Intervals.append(0)
		self.currentInterval +=1
		self.n += 1
		return intervalTime
		
	def average(self):
		return sum(self.Intervals)/self.n
	
	def pause(self):
		self.passedTime = time.time() - self.Start
		self.intervalTime = time.time()- self.IntervalStart
		return self.passedTime
	
	def unpause(self):
		self.Start = time.time() - self.passedTime
		self.IntervalStart = time.time() - self.intervalTime
	def totalTime(self):
		return sum(self.Intervals)