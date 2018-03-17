# thanks to @edersantana and @fchollet for suggestions & help.

import numpy as np
from ple import PLE  # our environment
from ple.games.puckworld import PuckWorld

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from example_support import ExampleAgent, ReplayMemory, loop_play_forever


class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

class Agent(ExampleAgent):
    """
        Our agent takes 1D inputs which are flattened.
        We define a full connected model below.
    """

    def __init__(self, *args, **kwargs):
        ExampleAgent.__init__(self, *args, **kwargs)

        self.state_dim = self.env.getGameStateDims()
        self.state_shape = np.prod((num_frames,) + self.state_dim)
        self.input_shape = (batch_size, self.state_shape)

    def build_model(self):
        model = Sequential()
        model.add(Dense(
            input_dim=self.state_shape, units=250, activation="sigmoid", kernel_initializer="he_uniform"
        ))
        model.add(Dense(
            units=150, activation="sigmoid", kernel_initializer="he_uniform"
        ))
        model.add(Dense(
            self.num_actions, activation="linear", kernel_initializer="he_uniform"
        ))

        model.compile(loss='mse', optimizer=SGD(lr=self.lr))

        self.model = model


def nv_state_preprocessor(state):
    # taken by inspection of source code. Better way is on its way!
    #max_values = np.array([128.0, 20.0, 128.0, 128.0])
    #state = np.array([state.values()]) / max_values

    #TODO: Consider whether we need to do the above rescaling for puckworld...
    state = np.array([state.values()])

    return state.flatten()

if __name__ == "__main__":
    # this takes about 15 epochs to converge to something that performs decently.
    # feel free to play with the parameters below.

    # training parameters
    num_epochs = 25
    num_steps_train = 15000  # steps per epoch of training
    num_steps_test = 3000
    update_frequency = 10 # step frequency of model training/updates

    # agent settings
    batch_size = 32
    num_frames = 4  # number of frames in a 'state'
    frame_skip = 2

    # percentage of time we perform a random action, help exploration.
    epsilon = 1.0
    epsilon_steps = 30000  # decay steps
    epsilon_min = 0.05
    lr = 0.01
    discount = 0.95
    rng = np.random.RandomState(24)

    # memory settings
    max_memory_size = 100000
    min_memory_size = 100  # number needed before model training starts

    epsilon_rate = (epsilon - epsilon_min) / epsilon_steps

    # PLE takes our game and the state_preprocessor. It will process the state
    # for our agent.
    game = PuckWorld(width=500, height=500)
    env = PLE(game, fps=60, state_preprocessor=nv_state_preprocessor)

    agent = Agent(env, batch_size, num_frames, frame_skip, lr,
                  discount, rng, optimizer="sgd")
    agent.build_model()

    memory = ReplayMemory(max_memory_size, min_memory_size)

    env.init()

    for epoch in range(num_epochs):
        steps = 0
        losses, rewards = [], []
        env.display_screen = True

        # training loop
        while steps < num_steps_train:
            state = env.getGameState()
            reward, action = agent.act(state, epsilon=epsilon)
            memory.add([state, action, reward, env.game_over()])

            if steps % update_frequency == 0:
                loss = memory.train_agent_batch(agent)

                if loss is not None:
                    losses.append(loss)
                    epsilon = max([epsilon_min, epsilon - epsilon_rate])
            rewards.append(reward)
            steps += 1

        print "\nTrain Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}".format(epoch, epsilon, np.mean(losses), np.mean(rewards))


        #Set up a testing loop
        steps = 0
        losses, rewards = [], []

        # display the screen
        env.display_screen = True

        # slow it down so we can watch it fail!
        env.force_fps = False

        # testing loop
        while steps < num_steps_test:
            state = env.getGameState()
            reward, action = agent.act(state, epsilon=0.05)
            rewards.append(reward)
            steps += 1

            # done watching after 500 steps.
            if steps > 500:
                env.force_fps = True
                env.display_screen = False


        print "Test Epoch {:02d}: Avg. Reward {:0.3f}".format(epoch, np.mean(rewards))

    print "\nTraining complete. Will loop forever playing!"
    loop_play_forever(env, agent)
