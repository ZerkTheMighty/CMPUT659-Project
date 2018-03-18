import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from keras.initializers import RandomUniform, lecun_uniform

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j] == obj).all():
                return i,j

#Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4,4,4))
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])

    return state

def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((4,4,4))

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if (new_loc != wall):
        if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
            state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1

    return state

def getLoc(state, level):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return 0

def dispGrid(state):
    grid = np.zeros((4,4), dtype=str)
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '

    if player_loc:
        grid[player_loc] = 'P' #player
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit

    return grid

model = Sequential()
#init_weights = RandomUniform(minval=0.0, maxval=0.00001, seed=None)
init_weights = lecun_uniform()
model.add(Dense(164, init=init_weights, input_shape=(64,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init=init_weights))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(4, init=init_weights))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

#sgd = SGD(lr=0.1, momentum=0.0, decay=0.5, nesterov=False)
rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


#EASY GRIDWORLD VERSION
epochs = 200
gamma = 1 #since it may take several moves to goal, making gamma high
epsilon = 1
for i in range(epochs):

    state = initGrid()
    status = 1
    #while game still in progress
    while(status == 1):
        #random.seed(i)
        #np.random.seed(i)
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,64), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)
        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,64), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,4))
        y[:] = qval[:]
        if reward == 0: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output
        print("Game #: %s" % (i,))
        model.fit(state.reshape(1,64), y, batch_size=1, nb_epoch=1, verbose=1)
        state = new_state
        if reward != 0:
            status = 0
    if epsilon > 0.1:
        epsilon -= (1/epochs)


#HARDER GRIDWORLD VERSION
# model.compile(loss='mse', optimizer=rms)#reset weights of neural network
# epochs = 1000
# gamma = 0.9
# epsilon = 1
# batchSize = 40
# buffer = 80
# replay = []
# #stores tuples of (S, A, R, S')
# h = 0
# for i in range(epochs):
#
#     state = initGrid()
#     #state = initGridPlayer() #using the harder state initialization function
#     status = 1
#     #while game still in progress
#     while(status == 1):
#         #We are in state S
#         #Let's run our Q function on S to get Q values for all possible actions
#         qval = model.predict(state.reshape(1,64), batch_size=1)
#         if (random.random() < epsilon): #choose random action
#             action = np.random.randint(0,4)
#         else: #choose best action from Q(s,a) values
#             action = (np.argmax(qval))
#         #Take action, observe new state S'
#         new_state = makeMove(state, action)
#         #Observe reward
#         reward = getReward(new_state)
#
#         #Experience replay storage
#         if (len(replay) < buffer): #if buffer not filled, add to it
#             replay.append((state, action, reward, new_state))
#         else: #if buffer full, overwrite old values
#             if (h < (buffer-1)):
#                 h += 1
#             else:
#                 h = 0
#             replay[h] = (state, action, reward, new_state)
#             #randomly sample our experience replay memory
#             minibatch = random.sample(replay, batchSize)
#             X_train = []
#             y_train = []
#             for memory in minibatch:
#                 #Get max_Q(S',a)
#                 old_state, action, reward, new_state = memory
#                 old_qval = model.predict(old_state.reshape(1,64), batch_size=1)
#                 newQ = model.predict(new_state.reshape(1,64), batch_size=1)
#                 maxQ = np.max(newQ)
#                 y = np.zeros((1,4))
#                 y[:] = old_qval[:]
#                 if reward == -1: #non-terminal state
#                     update = (reward + (gamma * maxQ))
#                 else: #terminal state
#                     update = reward
#                 y[0][action] = update
#                 X_train.append(old_state.reshape(64,))
#                 y_train.append(y.reshape(4,))
#
#             X_train = np.array(X_train)
#             y_train = np.array(y_train)
#             print("Game #: %s" % (i,))
#             model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
#             state = new_state
#         if reward != -1: #if reached terminal state, update game status
#             status = 0
#     if epsilon > 0.1: #decrement epsilon over time
#         epsilon -= (1/epochs)

def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != 0:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break
testAlgo(init=0)
