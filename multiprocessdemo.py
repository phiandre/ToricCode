import pathos as pt
import time

def calc_square(numbers):
	for n in numbers:
		print('Square: ' + str(n*n))

def calc_cube(numbers):
	for n in numbers:
		print('Cube: ' + str(n*n*n))

if __name__ == '__main__':
	arr = [1,2,3,4]
	p1 = pt.helpers.mp.Process(target=calc_square, args=([2],))
	p2 = pt.helpers.mp.Process(target=calc_cube, args=([2],))

	p1.start()
	p2.start()