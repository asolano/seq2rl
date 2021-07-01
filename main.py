# coding: utf-8

import gym
import argparse
import torch
import torch.nn as nn
import time
import math
import os
import random
import numpy as np
import datetime

from utils import (create_logger, set_random_seed, get_device, to_column_batches, get_env_dimensions)
from typing import Tuple, Optional
from model import CustomTransformer
from transformer import train_with_batches, split_dataset, generate_dataset2
from actor_critic import Policy, train_actor_critic, test_policy_quality
from fake_env import FakeEnvironment
from torch import optim


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Applied to Reinforcement Learning')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='RNG seed')
    parser.add_argument('--data-folder',
                        type=str,
                        default='dataset',
                        help='location of data')
    parser.add_argument('--generate',
                        action='store_true',
                        help='generate new dataset from environment'
                        )
    parser.add_argument('--environment',
                        type=str,
                        default='CartPole-v1',
                        help='Gym environment to use')
    parser.add_argument('--rollouts',
                        type=int,
                        default=5000,  # 10000
                        help='number of rollouts when generating data')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        metavar='N',
                        help='transformer training batch size')
    parser.add_argument('--sequence-length',
                        type=int,
                        default=200,
                        help='sequence length')
    parser.add_argument('--clip',
                        type=float,
                        default=0.25,
                        help='gradient clipping')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='dropout rate')
    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        metavar='N',
                        help='report interval')
    parser.add_argument('--log-path',
                        type=str,
                        default='logs',
                        help='path to save the log files')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='upper epoch limit')
    parser.add_argument('--save',
                        type=str,
                        default='best_model.pt',
                        help='path to save the final model')
    parser.add_argument('--num-features',
                        type=int,
                        default=256,
                        help='the number of expected features in the encoder/decoder inputs of the transformer')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='use CUDA')

    args = parser.parse_args()
    return args


def dreamer_algorithm(env_name,
                      args,
                      device,
                      seed_episodes,
                      collect_interval,
                      batch_size,
                      sequence_length,
                      horizon,
                      num_trials,
                      logger=None):
    env = gym.make(env_name)

    # Transformer world model
    save_filename = f'{env_name}_tf_model.pt'

    obs_dim, action_dim, reward_dim, done_dim = get_env_dimensions(env)

    # source and target dimensions
    src_dim = obs_dim + action_dim + done_dim
    tgt_dim = reward_dim + obs_dim + done_dim

    transformer = CustomTransformer(src_dim=src_dim,
                                    tgt_dim=tgt_dim,
                                    d_model=args.num_features,
                                    nhead=2,
                                    num_encoder_layers=2,
                                    num_decoder_layers=2,
                                    dim_feedforward=512,
                                    dropout=args.dropout,
                                    max_seq_length=args.sequence_length
                                    )
    transformer = transformer.to(device)
    tf_optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Actor Critic
    fc1_dims = 512
    if type(env.action_space) == gym.spaces.box.Box:
        n_actions = env.action_space.shape[0]
    else:
        n_actions = env.action_space.n
    policy = Policy(n_state=obs_dim,
                    n_actions=n_actions,
                    fc1_dims=fc1_dims)
    policy = policy.to(device)

    lr = 3e-2
    ac_optimizer = optim.Adam(policy.parameters(), lr=lr)

    train_sources, train_targets = torch.FloatTensor().to(device), torch.FloatTensor().to(device)
    valid_sources, valid_targets = torch.FloatTensor().to(device), torch.FloatTensor().to(device)
    test_sources, test_targets = torch.FloatTensor().to(device), torch.FloatTensor().to(device)

    max_loop_steps = 100  # TODO parameter
    log_interval = 25

    loop_steps = 0
    split = 0.8, 0.1, 0.1

    # FIXME pretrain slightly the policy on real env
    #policy, optimizer = train_actor_critic(policy,
    #                                       optimizer,
    #                                       gamma=0.99,
    #                                       eps=1e-5,
    #                                       env=env,
    #                                       max_episodes=50,
    #                                       max_steps=250,
    #                                       device=device,
    #                                      log_interval=log_interval,
    #                                       logger=logger)

    # fake_env = FakeEnvironment(env=env,
    #                            model=world_model,
    #                            seq_length=args.sequence_length,
    #                            device=device)

    converged = False
    while not converged:
        # reset environment
        # env.reset()

        # collect dataset from real environment
        logger.debug(f'Getting data from environment')

        new_sources, new_targets = generate_dataset2(env,
                                                     num_tokens=seed_episodes * 50,
                                                     device=device,
                                                     policy=policy,
                                                     logger=logger)

        new_train_sources, new_valid_sources, new_test_sources = split_dataset(new_sources, split)
        new_train_targets, new_valid_targets, new_test_targets = split_dataset(new_targets, split)

        # covert to S,N,E shape
        new_train_sources = to_column_batches(new_train_sources, batch_size, device)
        new_train_targets = to_column_batches(new_train_targets, batch_size, device)
        new_valid_sources = to_column_batches(new_valid_sources, batch_size, device)
        new_valid_targets = to_column_batches(new_valid_targets, batch_size, device)
        new_test_sources = to_column_batches(new_test_sources, batch_size, device)
        new_test_targets = to_column_batches(new_test_targets, batch_size, device)

        # FIXME retrains every time
        # add to existing datasets
        # train_sources = torch.cat([train_sources, new_train_sources])
        # train_targets = torch.cat([train_targets, new_train_targets])
        # valid_sources = torch.cat([valid_sources, new_valid_sources])
        # valid_targets = torch.cat([valid_targets, new_valid_targets])
        # test_sources = torch.cat([test_sources, new_test_sources])
        # test_targets = torch.cat([test_targets, new_test_targets])

        train_sources = new_train_sources
        train_targets = new_train_targets
        valid_sources = new_valid_sources
        valid_targets = new_valid_targets
        test_sources = new_test_sources
        test_targets = new_test_targets

        logger.info(f'Train: {train_sources.shape}')
        logger.info(f'Test: {test_sources.shape}')
        logger.info(f'Valid: {valid_sources.shape}')

        # use the datasets to learn/improve the world model
        logger.debug(f'Training world model')
        transformer = train_with_batches(epochs=args.epochs,
                                         sequence_length=args.sequence_length,
                                         clip=args.clip,
                                         device=device,
                                         model=transformer,
                                         optimizer=tf_optimizer,
                                         train_inputs_batches=train_sources,
                                         train_outputs_batches=train_targets,
                                         valid_inputs_batches=valid_sources,
                                         valid_outputs_batches=valid_targets,
                                         test_inputs_batches=test_sources,
                                         test_outputs_batches=test_targets,
                                         save=save_filename,
                                         log_interval=log_interval,
                                         logger=logger)

        # use world model to learn a policy
        logger.debug(f'Training policy')
        gamma = 0.99
        max_episodes = 100
        max_steps = 250
        eps = np.finfo(np.float32).eps.item()

        fake_env = FakeEnvironment(env=env,
                                   model=transformer,
                                   seq_length=args.sequence_length,
                                   device=device)

        # DEBUG use env for the real environment, it seems to learn properly
        policy, ac_optimizer = train_actor_critic(policy,
                                                  ac_optimizer,
                                                  gamma,
                                                  eps,
                                                  fake_env,
                                                  max_episodes,
                                                  max_steps,
                                                  device=device,
                                                  log_interval=log_interval,
                                                  logger=logger)

        # check policies quality in the real environment

        average_score, policy = test_policy_quality(env=env,
                                                    policy=policy,
                                                    num_trials=num_trials,
                                                    max_steps=max_steps,
                                                    logger=logger,
                                                    device=device)
        logger.debug(
            f'new average score over {num_trials} trials={average_score} (solved at {env.spec.reward_threshold})')

        if average_score >= env.spec.reward_threshold:
            converged = True

        # check loop termination
        loop_steps += 1
        if loop_steps >= max_loop_steps:
            logger.debug('Reached max loop steps, breaking early...')
            converged = True


def main():
    args = parse_arguments()

    # Create logs folder
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # TODO log file name pattern, uuid, date, etc..
    prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_path, f'{prefix}_{args.environment}.log')
    logger = create_logger(name='seq2rl', file_name=log_file)

    set_random_seed(args.seed, logger)
    device = get_device(args.cuda, logger)

    dreamer_algorithm(env_name=args.environment,
                      device=device,
                      args=args,
                      seed_episodes=100,
                      collect_interval=100,
                      batch_size=args.batch_size,
                      sequence_length=args.sequence_length,
                      horizon=15,
                      num_trials=20,
                      logger=logger)

    logger.info('All done.')


if __name__ == '__main__':
    main()
