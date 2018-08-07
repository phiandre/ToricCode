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
	def stop(self):
		self.Stop = time.time()
		return self.Stop-self.Start
	def interval(self):
		self.IntervalStop = time.time()
		intervalTime = self.IntervalStop - self.IntevalStart
		self.IntervalStart = time.time()
		self.Intevals.append(intervalTime)
		self.n += 1
		return intervalTime
	def average(self)
		return sum(self.Intervals)/self.n