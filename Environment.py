"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import random
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import os.path


UNIT = 40   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width
sleepTime = 0 # time in between steps
numberOfErrors = 4 # must be even!


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.numErrors = numberOfErrors
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

        # creates a new filename each time we run the code
        tmp = list('numSteps1.txt')
        i = 1
        while os.path.isfile("".join(tmp)):
            i += 1
            tmp[8] = str(i)

        self.filename="".join(tmp)


    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = [20, 20]


        #create errors
        self.errors = []
        self.errorSquares= []
        for i in range(0,self.numErrors):
            center = origin + [random.randint(0,MAZE_W-1)*UNIT,random.randint(0,MAZE_H-1)*UNIT]
            while any((center == x) for x in self.errors): #if error already exists
                center = origin + [random.randint(0,MAZE_W-1)*UNIT,random.randint(0,MAZE_H-1)*UNIT]

            print("center: ", center)
            self.errors.append(center)
            self.errorSquares.append(self.canvas.create_rectangle(
             self.errors[i][0] - 15, self.errors[i][1] - 15,
             self.errors[i][0] + 15, self.errors[i][1] + 15,
             fill='blue'))

        # pack all
        self.canvas.pack()

    def reset(self):
        self.numSteps = 0

        self.update()
        time.sleep(sleepTime)
  
        origin = [20, 20]

        for i in range(0,self.numErrors):
            self.canvas.delete(self.errorSquares[i])
        self.errors = []

        self.numErrors = numberOfErrors

        self.errorSquares= []
        for i in range(0,self.numErrors):
            center = [origin[0] + random.randint(0,MAZE_W-1)*UNIT, origin[1] + random.randint(0,MAZE_H-1)*UNIT]
            #print("center: ",center)

            
            while any((center == x) for x in self.errors): #if error already exists
                center = [origin[0] + random.randint(0,MAZE_W-1)*UNIT, origin[1] + random.randint(0,MAZE_H-1)*UNIT]
            self.errors.append(center)
            self.errorSquares.append(self.canvas.create_rectangle(
             self.errors[i][0] - 15, self.errors[i][1] - 15,
             self.errors[i][0] + 15, self.errors[i][1] + 15,
             fill='blue'))


        return self.errors

    def getCenter(self,edges):
        return [(edges[2]+edges[0])/2, (edges[3]+edges[1])/2]

    def step(self, observation, action, errorIndex):
        #print("self.errors ",self.errors)
        self.errors = observation
        self.errors_ = self.errors[:]
        self.numSteps +=  1
        #print("hej: ",self.errorSquares[errorIndex])
        s = self.canvas.coords(self.errorSquares[errorIndex])
        base_action = [0, 0]
        #print("action: ",action)
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
            else:
                base_action[1] +=(MAZE_H - 1) * UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
            else:
                base_action[1] -=(MAZE_H - 1) * UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
            else:
                base_action[0] -= (MAZE_H - 1) * UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
            else:
                base_action[0] += (MAZE_W - 1) * UNIT

        self.canvas.move(self.errorSquares[errorIndex], base_action[0], base_action[1])  # move agent

        p_ = self.getCenter(self.canvas.coords(self.errorSquares[errorIndex])) # next state


        # reward function
        done = False
        if any((p_ == x) for x in self.errors_):
            
            #print("errorIndex: ", errorIndex)
            del self.errors_[errorIndex]
            self.canvas.delete(self.errorSquares[errorIndex])
            del self.errorSquares[errorIndex]

            errorIndex2=self.errors_.index(p_)   #removes error on same place as s_, note that s_ is still represented as s in the errors-list
            del self.errors_[errorIndex2]
            self.canvas.delete(self.errorSquares[errorIndex2])
            del self.errorSquares[errorIndex2]
            self.numErrors-=2
            
            reward = 1
            if len(self.errors_)==0: #if we are finished
                with open(self.filename , "a+")as f:
                    f.write(str(self.numSteps)+"\n")
                done = True
                self.numSteps=0

        else:
            self.errors_[errorIndex]=p_
            reward = -1




        return self.errors_, reward, done


    def render(self):
        time.sleep(sleepTime)
        self.update()


def update():
    for t in range(10):
        s = env.reset()

        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
