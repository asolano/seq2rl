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

        self.episodes = 0

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


def select_action(model, observation, device):
    state = torch.from_numpy(observation).float().to(device)
    probabilities, state_value = model(state)

    # Create a categorical distribution over the list of probabilities of actions
    m = Categorical(probabilities)

    # Sample an action using the distribution
    action = m.sample()

    # Save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # Return the action to take
    return model, action.item()


def finish_episode(model, optimizer, gamma, eps, device):
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

    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], dtype=torch.float32).to(device)))

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

    # Count episodes
    model.episodes += 1

    return model, optimizer


# Train the policy using the environment
def train_actor_critic(policy,
                       optimizer,
                       gamma,
                       eps,
                       env,
                       max_episodes,
                       max_steps,
                       device,
                       log_interval,
                       logger):
    reward_history = deque(maxlen=100)
    total_steps = 0

    for i in range(max_episodes):
        observation = env.reset()
        episode_reward = 0
        min_reward, max_reward = float('inf'), float('-inf')

        for s in range(max_steps):
            policy, action = select_action(policy, observation, device)
            observation, reward, done, _ = env.step(action)

            min_reward = min(reward, min_reward)
            max_reward = max(reward, max_reward)

            policy.rewards.append(reward)
            episode_reward += reward
            total_steps += 1
            if done:
                break

        # cumulative moving average
        reward_history.append(episode_reward)
        cumulative_average = sum(reward_history) / len(reward_history)

        # back propagation
        policy, optimizer = finish_episode(policy, optimizer, gamma, eps, device=device)

        # log results
        if i % log_interval == 0:
            # TODO steps and min/max reward, last is nt that interesting
            print(f'Episode {i}: steps={s+1} rewards={min_reward:.2f}/{max_reward:.2f}/{cumulative_average:.2f} (min/max/cum.avg)')

        # check if the environment is solved
        # FIXME the fake env can mark it as solved by mistake by giving too much reward
        #  if cumulative_average > env.spec.reward_threshold:
        #    print(f'Solved? Running reward={cumulative_average:.2f}, threshold={env.spec.reward_threshold}')
        #    break

    print(f'total steps={total_steps}, last episode steps={s}')
    return policy, optimizer


def test_training_actor_critic(env, device):
    # FIXME args
    lr = 7e-4  # 3e-2
    gamma = 0.99
    max_episodes = 2_000
    max_steps = 1_000
    log_interval = 50

    env.seed(seed)
    torch.manual_seed(seed)

    if type(env.action_space) == gym.spaces.box.Box:
        n_actions = env.action_space.shape[0]
    else:
        n_actions = env.action_space.n

    policy = Policy(n_state=env.observation_space.shape[0],
                    n_actions=n_actions,
                    fc1_dims=512)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    eps = np.finfo(np.float32).eps.item()

    new_policy, optimizer = train_actor_critic(policy,
                                               optimizer,
                                               gamma,
                                               eps,
                                               env,
                                               max_episodes,
                                               max_steps,
                                               device=device,
                                               logger=None,
                                               log_interval=log_interval)

    print(new_policy.episodes)

    return new_policy


if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('Acrobot-v1')
    env = gym.make('LunarLander-v2')
    seed = 42
    env.seed(seed)
    torch.manual_seed(seed)

    from utils import get_device

    device = get_device(cuda=False)

    # FIXME continuous actions require a different model
    # env = gym.make('LunarLanderContinuous-v2')
    test_training_actor_critic(env, device=device)
