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

from utils import (create_logger, set_random_seed, get_device, to_column_batches)
from typing import Tuple, Optional
from model import CustomTransformer


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
                        default=20,
                        metavar='N',
                        help='transformer training batch size')
    parser.add_argument('--sequence-length',
                        type=int,
                        default=35,
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
                        default=20,  # 20
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


def load_dataset(save_path, environment: str, logger=None) -> Tuple[torch.Tensor, torch.Tensor]:
    input_path = os.path.join(save_path, environment, 'inputs.pt')
    output_path = os.path.join(save_path, environment, 'outputs.pt')

    if logger is not None:
        logger.info(f'Loading source dataset from {input_path}, target dataset from {output_path}')

    try:
        data_inputs = torch.load(input_path)
        data_outputs = torch.load(output_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Dataset not found in path {input_path}, use the --generate option first.")

    return data_inputs, data_outputs


def save_dataset(save_path, environment: str, data_inputs: torch.Tensor, data_outputs: torch.Tensor):
    save_folder = os.path.join(save_path, environment)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    input_path = os.path.join(save_folder, 'inputs.pt')
    output_path = os.path.join(save_folder, 'outputs.pt')

    torch.save(data_inputs, input_path)
    torch.save(data_outputs, output_path)


def action_1d(x):
    if type(x) is torch.Tensor:
        return x.view(-1)
    elif type(x) is np.ndarray:
        return torch.Tensor(x)
    else:
        return torch.Tensor([x])


def get_observation_data(observation):
    is_dict_obs = type(observation) == dict
    if is_dict_obs:
        # FIXME key name
        observation = observation['observation']
    return observation


def generate_dataset(env,
                     num_rollouts=1000,
                     logger=None) -> Tuple[torch.Tensor, torch.Tensor]:
    entries = []

    # FIXME Box, Discrete, Dict...
    # Get dimensions
    is_dict_obs = type(env.observation_space) is gym.spaces.dict.Dict
    if is_dict_obs:
        obs_dim = env.observation_space['observation'].shape[0]
    else:
        obs_dim = env.observation_space.shape[0]

    # FIXME get directly from action_space
    action = env.action_space.sample()
    if type(action) is np.ndarray:
        act_dim = action.shape[0]
    else:
        act_dim = 1

    # FIXME use this
    rew_dim = 1
    done_dim = 1

    for r in range(num_rollouts):
        observation = env.reset()

        done = False
        steps = 0

        last_obs = get_observation_data(observation)

        while not done:
            # TODO Use a callable for the policy
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)

            # FIXME
            observation = get_observation_data(observation)
            # if is_dict_obs:
            #    observation = observation['observation']

            t = torch.cat([
                # np array to tensor
                torch.Tensor(last_obs),
                # scalar or numpy array
                # torch.Tensor([action]),
                action_1d(action),
                # scalar to tensor
                torch.Tensor([reward]),
                torch.Tensor([done])
            ])

            steps += 1

            entries.append(t)
            last_obs = get_observation_data(observation)

            if done:
                env.reset()

        if logger is not None:
            logger.debug(f'Rollout {r + 1}/{num_rollouts} ({steps} steps)')

    env.close()

    big_tensor = torch.cat(entries).reshape(-1, entries[0].shape[0])

    # Big tensor shape is like this (without the time step):

    # t    | obs    act    rew    don
    # 0    | A      B      C      D
    # 1    | E      F      G      H
    # 2    | I      J      K      L
    # ...

    # where
    # t = time step
    # obs = observation
    # act = action
    # rew = reward
    # don = done
    # and [A, B, C... ] are 1D tensors (reshape if needed)

    # Input to output is
    # t=0 [A, B] -> [C, D, E]  obs + action => reward, done, next observation
    # t=1 [E, F] -> [G, H, I]
    # etc.

    # FIXME do the separation after splitting into train/valid/test?

    # TODO Check this for action dim
    # Inputs are the first two "columns" (obs + act)
    # They discard the last row (no next observation)
    # inputs = x[:-1, :split]
    dataset_inputs = big_tensor[:-1, :(obs_dim + act_dim)]  # split is obs+act

    # Outputs are the concatenation of the last "columns" (rew + done) with the first element of the next row (next obs)
    # They discard the first row (no previous state)
    # outputs = torch.cat( [x[:-1, split:], x[1:, :obs_dim]], 1)
    dataset_outputs = torch.cat([big_tensor[:-1, (obs_dim + act_dim):], big_tensor[1:, :obs_dim]], 1)

    return dataset_inputs, dataset_outputs


def split_dataset(data: torch.Tensor, split: Tuple[float, float, float]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO assert splits are valid
    train_split = split[0]
    valid_split = split[1]

    train_cut = int(len(data) * train_split)
    valid_cut = int(len(data) * (train_split + valid_split))

    train = data[:train_cut]
    valid = data[train_cut:valid_cut]
    test = data[valid_cut:]

    return train, valid, test


def get_ith_batch(source, i, count):
    seq_len = min(count, len(source) - 1 - i)
    data = source[i:i + seq_len]
    return data


def train(model, criterion, source_batches, target_batches, args, lr, epoch, device, logger=None):
    # Enable dropout
    model.train()

    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, source_batches.size(0) - 1, args.sequence_length)):
        # Get source and target sequences
        source = get_ith_batch(source_batches, i, count=args.sequence_length)
        target = get_ith_batch(target_batches, i, count=args.sequence_length)

        # Reset gradients
        model.zero_grad()

        # Get output from model
        output = model(src=source, tgt=target, device=device)

        # Calculate and back-propagate loss
        loss = criterion(output, target)
        loss.backward()

        # TODO Confirm if this is needed for transformers
        # Prevent exploding gradients problem in RNNS/LSTMs
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad)

        # Accumulate loss
        total_loss += loss.item()

        # Logging
        if batch > 0 and batch % args.log_interval == 0:
            if logger is not None:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time

                logger.info(f"Epoch {epoch:3d}"
                            f" | {batch:5d}/{len(source_batches) // args.sequence_length:5d} batches"
                            f" | lr {lr:02.5f}"
                            f" | ms/batch {elapsed * 1000 / args.log_interval:5.2f}"
                            f" | loss= {cur_loss:5.4f}"
                            f" | ppl= {math.exp(cur_loss):8.4f}")

                total_loss = 0.0
                start_time = time.time()

        # FIXME needed?
        # if args.dry_run:
        #    break

    return model


def evaluate(model, criterion, source_batches, target_batches, args, device, logger=None):
    model.eval()
    total_loss = 0.0

    log_i = random.randrange(0, source_batches.size(0) - 1, args.sequence_length)
    with torch.no_grad():
        for i in range(0, source_batches.size(0) - 1, args.sequence_length):
            # Get source and target sequences
            source = get_ith_batch(source_batches, i, count=args.sequence_length)
            target = get_ith_batch(target_batches, i, count=args.sequence_length)

            output = model(src=source, tgt=target, device=device)

            if logger is not None and i == log_i:
                logger.debug(f"Sample evaluation"
                             f"\n* batch number: {i}"
                             f"\n* reward"
                             f"\n- output: {output[0, 0, 0]:2.3f}"
                             f"\n- target: {target[0, 0, 0]:2.3f}"
                             f"\n* done"
                             f"\n- output: {output[0, 0, 1]:2.3f}"
                             f"\n- target: {target[0, 0, 1]:2.3f}"
                             f"\n* observation"
                             f"\n- output: {output[0, 0, 2:]}"
                             f"\n- target: {target[0, 0, 2:]}")

            total_loss += len(source) * criterion(output, target).item()

    return total_loss / (len(source_batches) - 1)


# NOTES
# O=observation, A=action, R=reward, D=done
# Transformer input sequence token is O+A, output token is R+D+O (next O)
# Similar to translation for close languages
# It may be possible to pack everything into a single type of token (O+A+R+D) too
# An input/output sequence (NLP sentence) is a series of steps in the environment
# Positional encoding can be used as normal
# An episode is similar to a sentence

def train_world_model(args, device, logger=None) -> torch.Tensor:
    # TODO create a folder for each environment in the dataset save path

    if args.generate:
        env = gym.make(args.environment)
        # TODO Replace rollouts for minimum (approximate) number of samples/tokens?
        data_inputs, data_outputs = generate_dataset(env,
                                                     num_rollouts=args.rollouts,
                                                     logger=logger)
        save_dataset(args.data_folder, args.environment, data_inputs, data_outputs)
    else:
        data_inputs, data_outputs = load_dataset(args.data_folder, args.environment, logger)

    if logger is not None:
        logger.debug(f'Dataset inputs shape = {data_inputs.shape}')
        logger.debug(f'Dataset outputs shape = {data_outputs.shape}')
        logger.info(f'Dataset contains {data_inputs.shape[0]} tokens.')

    # Prepare train/validation/test split
    split = 0.8, 0.1, 0.1
    train_inputs, valid_inputs, test_inputs = split_dataset(data_inputs, split)
    train_outputs, valid_outputs, test_outputs = split_dataset(data_outputs, split)

    train_inputs_batches = to_column_batches(train_inputs, args.batch_size, device=device)
    train_outputs_batches = to_column_batches(train_outputs, args.batch_size, device=device)
    valid_inputs_batches = to_column_batches(valid_inputs, args.batch_size, device=device)
    valid_outputs_batches = to_column_batches(valid_outputs, args.batch_size, device=device)
    test_inputs_batches = to_column_batches(test_inputs, args.batch_size, device=device)
    test_outputs_batches = to_column_batches(test_outputs, args.batch_size, device=device)

    # FIXME more transformer parameters as arguments: heads, layers, etc..
    model = CustomTransformer(src_dim=train_inputs.shape[1],
                              tgt_dim=train_outputs.shape[1],
                              d_model=args.num_features,
                              nhead=2,
                              num_encoder_layers=2,
                              num_decoder_layers=2,
                              dim_feedforward=1024,
                              dropout=args.dropout,
                              max_seq_length=args.sequence_length
                              ).to(device)
    if logger is not None:
        logger.debug(f'Built transformer model: "{model}"')

    # TODO separate from here

    model = retrain_world_model(args,
                                device,
                                model,
                                train_inputs_batches,
                                train_outputs_batches,
                                valid_inputs_batches,
                                valid_outputs_batches,
                                test_inputs_batches,
                                test_outputs_batches,
                                logger)

    return model


def retrain_world_model(args,
                        device,
                        model,
                        train_inputs_batches,
                        train_outputs_batches,
                        valid_inputs_batches,
                        valid_outputs_batches,
                        test_inputs_batches, test_outputs_batches,
                        logger=None) -> torch.Tensor:
    # For predicting the next state the transformer is given input and target sequences
    # Sequences have length L steps (same as BPTT for words)

    criterion = nn.MSELoss()
    lr = 5.0
    best_val_loss = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # Training
        model = train(model,
                      criterion,
                      train_inputs_batches,
                      train_outputs_batches,
                      args,
                      lr,
                      epoch,
                      device=device,
                      logger=logger)

        # Validation
        val_loss = evaluate(model,
                            criterion,
                            source_batches=valid_inputs_batches,
                            target_batches=valid_outputs_batches,
                            args=args,
                            device=device,
                            logger=logger)

        logger.info('=' * 89)
        logger.info(f'End of epoch {epoch:3d}'
                    f' | time: {(time.time() - epoch_start_time):5.3f}s'
                    f' | valid loss {val_loss:5.4f}'
                    f' | valid ppl {math.exp(val_loss):8.4f}')
        logger.info('=' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

    # Test data
    test_loss = evaluate(model,
                         criterion,
                         source_batches=test_inputs_batches,
                         target_batches=test_outputs_batches,
                         args=args,
                         device=device)

    logger.info(f'End of training | test loss {test_loss:5.4f} | test ppl {math.exp(test_loss):8.4f}')

    return model


def test_policy_quality(environment, policy, num_trials, max_steps, logger=None):
    total_reward = 0

    for i in range(num_trials):
        done = False
        trial_reward = 0
        steps = 0
        observation = environment.reset()

        # NOTE infinite environments need a step limit
        while not done and steps < max_steps:
            action = policy(environment, observation)
            observation, reward, done, _ = environment.step(action)
            trial_reward += reward
            steps += 1
        total_reward += trial_reward

    return total_reward / num_trials


def random_policy(environment, observation):
    action = environment.action_space.sample()
    return action


def dreamer_algorithm(env_name,
                      args,
                      device,
                      seed_episodes=5,
                      collect_interval=100,
                      batch_size=50,
                      sequence_length=50,
                      horizon=15,
                      num_trials=100,
                      logger=None):
    converged = False
    print(f'converged={converged}')

    policy = random_policy  # Use random action selection

    train_sources, train_targets = torch.FloatTensor(), torch.FloatTensor()
    valid_sources, valid_targets = torch.FloatTensor(), torch.FloatTensor()
    test_sources, test_targets = torch.FloatTensor(), torch.FloatTensor()

    max_loop_steps = 5  # TODO parameter

    loop_steps = 0
    split = 0.8, 0.1, 0.1

    env = gym.make(env_name)

    while not converged:
        # reset environment
        print(f'resetting environment')
        env.reset()

        # collect dataset from real environment
        print(f'getting {batch_size} data sequences of {sequence_length} length')
        new_sources, new_targets = generate_dataset(env, num_rollouts=seed_episodes, logger=logger)

        new_train_sources, new_valid_sources, new_test_sources = split_dataset(new_sources, split)
        new_train_targets, new_valid_targets, new_test_targets = split_dataset(new_targets, split)

        # add to existing datasets
        train_sources = torch.cat([train_sources, new_train_sources])
        train_targets = torch.cat([train_targets, new_train_targets])
        valid_sources = torch.cat([valid_sources, new_valid_sources])
        valid_targets = torch.cat([valid_targets, new_valid_targets])
        test_sources = torch.cat([test_sources, new_test_sources])
        test_targets = torch.cat([test_targets, new_test_targets])

        # use the datasets to learn/improve the world model
        print(f'learning world model with policy={policy.__name__}')

        # TODO First run is different
        if world_model is None:
            world_model = CustomTransformer(src_dim=train_sources.shape[1],
                                            tgt_dim=train_targets.shape[1],
                                            d_model=args.num_features,
                                            nhead=2,
                                            num_encoder_layers=2,
                                            num_decoder_layers=2,
                                            dim_feedforward=1024,
                                            dropout=args.dropout,
                                            max_seq_length=args.sequence_length
                                            ).to(device)

        else:
            world_model = retrain_world_model(args,
                                              device,
                                              world_model,
                                              train_sources,
                                              train_targets,
                                              valid_sources,
                                              valid_targets,
                                              test_sources,
                                              test_targets)

        # use world model to learn a policy
        print(f'using world model to learn new policy')
        # policy = ...

        # check policy quality in the real environment
        print(f'checking policy quality')
        average_score = test_policy_quality(env,
                                            policy,
                                            num_trials=num_trials,
                                            max_steps=10_000)
        print(f'average score over {num_trials} trials={average_score} (solved={env.spec.reward_threshold})')
        if average_score >= env.spec.reward_threshold:
            converged = True

        # check loop termination
        loop_steps += 1
        if loop_steps >= max_loop_steps:
            print('Reached max loop steps, breaking early...')
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
    device = get_device(args, logger)

    # FIXME testing only
    model = train_world_model(args, device, logger)

    # TODO
    # dreamer_algorithm(args.environment,
    #                   device=device,
    #                   args=args,
    #                   seed_episodes=5,
    #                   collect_interval=100,
    #                   batch_size=50,
    #                   sequence_length=50,
    #                   horizon=15,
    #                   num_trials=100,
    #                   logger=logger)

    logger.info('All done.')


if __name__ == '__main__':
    main()
