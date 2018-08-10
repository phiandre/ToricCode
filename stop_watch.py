import time

class StopWatch:
	def __init__(self):
		self.Start = time.time()
		self.Stop = time.time()
		self.IntervalStart = time.time()
		self.IntervalStop = time.time()
		self.Intervals =  []
		self.n = 0
	def start(self):
		self.Start = time.time()
		self.Stop = time.time()
		self.IntervalStart = time.time()
		self.IntervalStop = time.time()
		self.Intervals =  []
		self.n = 0
	def stop(self):
		self.Stop = time.time()
		return self.Stop-self.Start
	def interval(self):
		self.IntervalStop = time.time()
		intervalTime = self.IntervalStop - self.IntervalStart
		self.IntervalStart = time.time()
		self.Intervals.append(intervalTime)
		self.n += 1
		return intervalTime
	def average(self):
		return sum(self.Intervals)/self.n
	def totalTime(self):
		return sum(self.Intervals)