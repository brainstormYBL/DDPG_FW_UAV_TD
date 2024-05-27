import random

import numpy as np
import torch


def train(par, agent, env, viz):
    # Start interation -- episode interation
    reward_store = []
    action_store = []
    rate_store = []
    for index_epi in range(par.episode_max):
        reward_epi = 0  # The reward in each episode
        print("The training in the " + str(index_epi) + " episode is beginning.")
        # Init the environment, return the init state
        state_now = env.reset()
        index_step = 0
        # Interation in each episode until the current episode is finished (done = True)
        while True:
            # Choose an action based on the state
            action = agent.choose_action(state_now)
            # Obtain the next state, reward and done.
            state_next, reward, rate, done = env.step(action, index_step)
            index_step += 1
            reward_epi += reward
            # Store the sample
            agent.store_transition(state_now, action, reward, state_next)
            # Update the state
            state_now = state_next
            # Learn
            agent.learn()
            if done:
                reward_store.append(reward)
                action_store.append(action)
                rate_store.append(rate)
                print("The reward at episode " + str(index_epi) + " is " + str(reward_epi))
                print("The action at episode " + str(index_epi) + " is " + str(action))
                print("Current radius is " + str(env.radius_now.item()))
                print("Current center is " + str(env.center_now))
                if par.visdom_flag:
                    viz.line(X=[index_epi + 1], Y=[reward_epi], win='reward', opts={'title': 'reward'},
                             update='append')
                    viz.line(X=[index_epi + 1], Y=[reward_epi * (10 ** 6)], win='sum rate', opts={'title': 'sum rate'},
                             update='append')
                break
    id_max = np.argmax(np.array(reward_store))
    action_res = action_store[id_max]
    rate_res = rate_store[id_max]
    return np.array(action_res), np.array(rate_res)


def sampling(agent, env, batch_size):
    for index_epi in range(1):
        state_now = env.reset()
        index_step = 0
        # Interation in each episode until the current episode is finished (done = True)
        while True:
            # Choose an action based on the state
            action = agent.choose_action(state_now)
            # Obtain the next state, reward and done.
            state_next, reward, rate, done = env.step(action, index_step)
            index_step += 1
            # Store the sample
            agent.store_transition(state_now, action, reward, state_next)
            # Update the state
            state_now = state_next
            if done:
                break
    samples = random.sample(agent.buffer, batch_size)
    s0, _, _, _ = zip(*samples)
    s0 = torch.tensor(s0, dtype=torch.float)
    mean = torch.mean(s0)
    std = torch.std(s0)

    return mean, std
