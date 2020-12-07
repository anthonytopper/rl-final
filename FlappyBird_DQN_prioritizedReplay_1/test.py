"""

### NOTICE ###
DO NOT revise this file

"""

import argparse
import numpy as np
from environment import Environment
import cv2

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30, name = 'video'):
    rewards = []
    #env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0
        writer = cv2.VideoWriter(name+'.avi', cv2.VideoWriter_fourcc(*'PIM1'), 30, (144, 256), True)
        #playing one game
        i=0
        while(not done):
            action = agent.make_action(state, test=True)
            # print(action)
            state, reward, done, info = env.step(action)
            frame = state[:,:,-3:].astype(np.uint8).swapaxes(0,1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
            episode_reward += reward
            if i>30*150:
                break
            i+=1
        rewards.append(episode_reward)
        writer.release()

        break

    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
