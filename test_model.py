import gym
import torch
from torch import optim
import numpy as np

from actor_critic import train_actor_critic, Policy
from fake_env import FakeEnvironment
from main import test_policy_quality

if __name__ == '__main__':
    save_file = 'transformer_model.pt'
    world_model = torch.load(save_file)
    print(world_model)

    env = gym.make('LunarLander-v2')
    device = torch.device('cpu')

    fake_env = FakeEnvironment(env, world_model, seq_length=10, device=device)

    policy = Policy(n_state=8,
                    n_actions=4,
                    fc1_dims=512).to(device)

    lr = 7e-4
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    gamma = 0.99
    max_episodes = 250
    max_steps = 200
    eps = np.finfo(np.float32).eps.item()

    policy, optimizer = train_actor_critic(policy,
                                           optimizer,
                                           gamma,
                                           eps,
                                           # FIXME env works
                                           #env,
                                           fake_env,
                                           max_episodes,
                                           max_steps,
                                           device=device,
                                           log_interval=50,
                                           logger=None)

    avg_reward, policy = test_policy_quality(env, policy, 100, max_steps, device, logger=None)
    print(f'Avg. reward={avg_reward}')
