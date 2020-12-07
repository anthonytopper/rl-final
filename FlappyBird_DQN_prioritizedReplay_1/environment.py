import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np

class Environment(object):
    def __init__(self, env_name, args, atari_wrapper=False, test=False, seed=595):
        game = FlappyBird(width=144, height=256, pipe_gap=88)
        self.test=test
        #define reward
        reward_func = rewards = {
            "positive": 1,
            "negative": -1.0,
            "tick": 1,
            "loss": -5.0,
            "win": 1.0
        }

        self.p = PLE(game, fps=30, display_screen=False, force_fps=True, reward_values = reward_func, rng=seed)
        self.observation = np.zeros((144,256,4,3))
        # if atari_wrapper:
        #     clip_rewards = not test
        #     self.env = make_wrap_atari(env_name, clip_rewards)
        # else:
        #     self.env = gym.make(env_name)

        self.action_space = self.p.getActionSet()
        # self.observation_space = self.env.observation_space

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''

        self.p.reset_game()
        observation = self.p.getScreenRGB()
        self.observation[:,:,0:-1,:] = self.observation[:,:,1:,:]
        self.observation[:,:,-1,:] = observation
        return self.observation.reshape(144,256,12)


    def step(self,action):
        reward = self.p.act(action)

        observation = self.p.getScreenRGB()    
        if self.p.game_over():
            done = True
        else:
            done = False
        self.observation[:,:,0:-1,:] = self.observation[:,:,1:,:]
        self.observation[:,:,-1,:] = observation

        return self.observation.reshape(144,256,12), reward, done, None


    def get_action_space(self):
        return self.action_space


    # def get_observation_space(self):
    #     return self.observation_space


    def get_random_action(self):
        return self.action_space.sample()
