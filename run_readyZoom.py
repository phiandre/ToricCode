import numpy as np
from RL import RLsys
from Env import Env
from GenerateToricData import Generate
from BlossomEnv import Env as BEnv
from keras.models import load_model
import time
import os.path
import pickle
import math
from Blossom import Blossom
import time


class MainClass:
    def __init__(self):

        # Alla booleans
        self.loadNetwork = True  # train an existing network
        self.gsRGrowth = np.load("Tweaks/GSgrowth.npy")
        self.windowSize = 5
        self.windowSize2 = 5
        self.checkGS = True

        # Epsilon decay parameters

        self.epsilonDecay = np.load("Tweaks/epsilonDecay.npy")
        if self.epsilonDecay:
            self.alpha = np.load("Tweaks/alpha.npy")  # flyttar "änden" på epsilon-kurvan
            self.k = np.load("Tweaks/k.npy")  # flyttar "mitten" på epsilon-kurvan

        self.networkName = 'Networks/trainedNetwork33.h5'
        self.networkNameZoom = 'Networks/Network100kMEM.h5'

        self.saveRate = 100  # how often the network is saved

        # creates a new filename for numSteps each time we run the code
        self.getFilename()

        self.avgTol = 1000  # Den mängd datapunkter som average tas över
        self.fR = np.load("Tweaks/correctGsR.npy")  # asymptotic Ground State reward

        self.run()

    # Om man vill ha en tanh-kurveökning av Ground State Reward väljs parametrar här




    def getFilename(self):
        tmp = list('Steps/numSteps1.npy')
        self.filename = "Steps/" + 'numSteps1.npy'
        self.static_element = 1

        while os.path.isfile("".join(tmp)):
            self.static_element += 1
            tmp[14] = str(self.static_element)
            self.filename = "".join(tmp)

    def rotateHumanRep(self, humanRep, j):
        tmp = np.concatenate([humanRep, humanRep[:, 0:1]], axis=1)
        tmp1 = np.concatenate([tmp, tmp[0:1, :]])
        humanRep = np.rot90(tmp1, j)
        state = humanRep[0:(humanRep.shape[0] - 1), 0:(humanRep.shape[1] - 1)]
        return state

    def labelState(self, s, size):
        state = s
        label = 1
        for j in range(size):
            for k in range(size):
                if state[j, k] == 1:
                    state[j, k] = label
                    label += 1
        return state

    def run(self):
        actions = 4
        comRep = np.load('ToricCodeComputerTest.npy')
        humRep = np.load('ToricCodeHumanTest.npy')
        size = comRep.shape[0]
        segmentSize = 3
        rl = RLsys(actions, size, windowSize = self.windowSize)
        rl2 = RLsys(actions, int(size / segmentSize), windowSize = self.windowSize)
        if self.loadNetwork:
            importNetwork1 = load_model(self.networkName)
            importNetwork2 = load_model(self.networkNameZoom)
            rl.qnet.network = importNetwork1
            print(rl.qnet.network.summary())

            rl2.qnet.network = importNetwork2
        rl.epsilon = 0
        rl.gamma = 0.6
        rl2.epsilon = 0
        steps = np.zeros(comRep.shape[2] * 4)

        averager = np.zeros(comRep.shape[
                                2] * 4)  # Används till att räkna ut hur sannolikt algoritmen återvänder till rätt grundtillstånd

        n = 0


        trainingIteration = 0
        if self.gsRGrowth:
            A = np.load("Tweaks/AGS.npy")
            B = np.load("Tweaks/BGS.npy")
            w = np.load("Tweaks/wGS.npy")
            b = np.load("Tweaks/bGS.npy")

        incorrectGsR = np.load("Tweaks/incorrectGsR.npy")
        stepR = np.load("Tweaks/stepR.npy")

        GScounter = 0
        Bcounter = 0



        for i in range(comRep.shape[2]):
            for j in range(4):
                state = np.copy(comRep[:, :, i])
                state = np.rot90(state, j)

                humanRep = humRep[:, :, i]
                humanRep = self.rotateHumanRep(humanRep, j)

                env = Env(state, humanRep, checkGroundState=self.checkGS, segmentSize=segmentSize, windowSize= self.windowSize)
                env.incorrectGsR = incorrectGsR
                env.stepR = stepR
                numSteps = 0

                if self.epsilonDecay:
                    rl.epsilon = ((self.k + trainingIteration + 12000) / self.k) ** (self.alpha)
                if self.gsRGrowth:
                    env.correctGsR = A * np.tanh(w * (trainingIteration + b)) + B
                else:
                    env.correctGsR = self.fR
                r = 0
                alone = False
                env.elimminationR = 10

                if np.count_nonzero(state) > 0:
                    state_ = np.copy(state)
                    state_ = self.labelState(state_, size)
                    BlossomObject = Blossom(state_)
                    MWPM = BlossomObject.readResult()
                    Benv = BEnv(state_, humanRep, checkGroundState=True)
                    minSteps = 0
                    for element in MWPM:
                        error1 = element[0] + 1
                        error2 = element[1] + 1
                        blossomReward, steps = Benv.blossomCancel(error1, error2)
                        minSteps += steps

                while (not alone) and (len(env.getErrors()) > 0):
                    numSteps = numSteps + 1
                    observation, x, indexVector = env.getObservation()
                    a, e = rl.choose_action(observation, indexVector)
                    r = env.moveError(a, e)
                    new_observation, alone, newIndexVector = env.getObservation()
                    #print('State: \n', env.state)


                   # if (numSteps % 20 == 0):
                        #print("Errors remaining: ", len(env.getErrors()))
                    if numSteps > 1000:
                        print('State: \n', env.state)
                print("I am zooming out...")
                zoomedOutState = env.zoomOut()
                zoomedOutEnv = Env(zoomedOutState, windowSize = self.windowSize2)
                zoomedOutEnv.elimminationR = -1
                # print("Zoomed out state \n", zoomedOutEnv.state)

                while len(env.getErrors()) > 0:
                    # print("zoomedOutEnv\n", zoomedOutEnv.state)
                    # print("env\n", env.state)
                    # print("len", len(env.getErrors()))
                    observation, x, indexVector = zoomedOutEnv.getObservation()

                    a, e = rl2.choose_action(observation, indexVector)
                    index = zoomedOutEnv.getErrors()[e, :]

                    nextPos = zoomedOutEnv.getPos(a, index)

                    annihilation = (zoomedOutEnv.state[nextPos[0], nextPos[1]] == 1)
                    r2 = zoomedOutEnv.moveError(a, e)
                    np.set_printoptions(linewidth=np.inf)

                    longStep, r = env.longMove(a, index)


                    numSteps += longStep
                    if annihilation:
                        pairSteps, r = env.pairErrors(nextPos)
                        numSteps += pairSteps
                    new_observation, alone, newindexVector = zoomedOutEnv.getObservation()


                    #if (numSteps % 20 == 0):
                        #print("Errors remaining: ", len(env.getErrors()))

                print("Steps taken at iteration " + str(trainingIteration) + ": ", numSteps)
                if r == env.correctGsR:
                    GScounter += 1

                if blossomReward == Benv.correctGsR:
                    Bcounter += 1
                print("Probability of correct GS so far: ", GScounter/(trainingIteration+1))
                print("MWPM: ", Bcounter/(trainingIteration+1))


                """
                if self.checkGS:
                    if r != 0:
                        if r == env.correctGsR:
                            averager[n] = 1
                        n += 1

                    if n < self.avgTol:
                        average = np.sum(averager)/n
                    else:
                        average = np.sum(averager[(n-self.avgTol):n])/self.avgTol



                if self.checkGS:
                    if n<self.avgTol:
                        print("Probability of correct GS last " + str(n) + ": " + str(average*100) + " %")
                    else:
                        print("Probability of correct GS last " + str(self.avgTol) + ": " + str(average*100) + " %")
                    steps[trainingIteration] = numSteps
                """

                if ((trainingIteration + 1) % self.saveRate == 0):
                    tmp = list('Networks/trainedNetwork1.h5')
                    tmp[23] = str(self.static_element)
                    filename = "".join(tmp)

                    # np.save(self.filename,steps[0:(trainingIteration+1)])

                    rl.qnet.network.save(filename)

                trainingIteration = trainingIteration + 1


"""""""""""""""""""""""""""""""""""""""""""""
Mainmetod, här körs själva simuleringen.
"""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':

    MainClass()


