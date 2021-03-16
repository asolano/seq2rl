# coding: utf-8
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


# Actor Critic based on PyTorch examples
class Policy(nn.Module):
    def __init__(self, n_state=4, n_actions=2, fc1_dims=128):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_state, fc1_dims)

        # actor layer
        self.action_head = nn.Linear(fc1_dims, n_actions)  # actions
        # critic layer
        self.value_head = nn.Linear(fc1_dims, 1)  # reward dim

        # action and reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))

        # actor: chooses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value of state s_t
        return action_prob, state_values


def select_action(model, observation):
    state = torch.from_numpy(observation).float()
    probabilities, state_value = model(state)

    # Create a categorical distribution over the list of probabilities of actions
    m = Categorical(probabilities)

    # Sample an action using the distribution
    action = m.sample()

    # Save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # Return the action to take
    return action.item()


def finish_episode(model, optimizer, gamma, eps):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []

    # calculate true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discount value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform back propagation
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def train_actor_critic(model, optimizer, gamma, eps, environment, max_episodes, max_steps, log_interval):
    reward_history = deque(maxlen=100)

    total_steps = 0

    for i in range(max_episodes):
        observation = environment.reset()
        episode_reward = 0

        for s in range(max_steps):
            action = select_action(model, observation)
            observation, reward, done, _ = environment.step(action)

            model.rewards.append(reward)
            episode_reward += reward
            total_steps += s
            if done:
                break

        # cumulative moving average
        reward_history.append(episode_reward)
        cumulative_average = sum(reward_history) / len(reward_history)

        # back propagation
        finish_episode(model, optimizer, gamma, eps)

        # log results
        if i % log_interval == 0:
            print(f'Episode {i} last reward {episode_reward:.2f} average reward {cumulative_average:.2f}')

        # check if the environment is solved
        if cumulative_average > environment.spec.reward_threshold:
            print(
                f'Solved. Running reward={cumulative_average:.2f}, threshold={environment.spec.reward_threshold}')
            break
    print(f'total steps={total_steps}, last episode steps={s}')


def test_training():
    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('Acrobot-v1')
    env = gym.make('LunarLander-v2')
    # FIXME continuous actions require a different model
    # env = gym.make('LunarLanderContinuous-v2')

    # FIXME args
    seed = 42
    lr = 7e-4  # 3e-2
    gamma = 0.99
    max_episodes = 2_000
    max_steps = 10_000
    log_interval = 50

    env.seed(seed)
    torch.manual_seed(seed)

    if type(env.action_space) == gym.spaces.box.Box:
        n_actions = env.action_space.shape[0]
    else:
        n_actions = env.action_space.n

    model = Policy(n_state=env.observation_space.shape[0],
                   n_actions=n_actions,
                   fc1_dims=1024)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    eps = np.finfo(np.float32).eps.item()

    train_actor_critic(model, optimizer, gamma, eps, env, max_episodes, max_steps, log_interval)


if __name__ == '__main__':
    test_training()
