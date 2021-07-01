# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import random
import numpy as np
import os
import gym
from typing import Tuple

from actor_critic import select_action
from model import CustomTransformer
from utils import to_column_batches, get_env_dimensions


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


def generate_dataset2(env, device, num_tokens, policy, logger):
    sources = torch.Tensor()
    targets = torch.Tensor()

    observation = env.reset()

    obs_dim, action_dim, reward_dim, done_dim = get_env_dimensions(env)

    # SOS, EOS tokens for sources and targets
    # Add an extra float to the vector, normally a zero
    source_sos = torch.zeros(size=(obs_dim + action_dim + done_dim,))
    source_sos[-1] = 1.0
    target_sos = torch.zeros(size=(reward_dim + obs_dim + done_dim,))
    target_sos[-1] = 1.0
    source_eos = torch.zeros(size=source_sos.size())
    source_eos[-1] = 2.0
    target_eos = torch.zeros(size=target_sos.size())
    target_eos[-1] = 2.0

    # start from SOS
    sources = torch.cat([sources, source_sos.reshape(1, -1)])
    targets = torch.cat([targets, target_sos.reshape(1, -1)])

    while len(sources) < num_tokens:
        if policy is None:
            action = env.action_space.sample()
        else:
            policy, action = select_action(policy, observation, device)

        s = torch.cat([
            torch.Tensor(observation),
            torch.Tensor([action]),
            torch.Tensor([0.0])  # the extra float
        ])
        sources = torch.cat([sources, s.reshape(1, -1)])
        observation, reward, done, _ = env.step(action)
        t = torch.cat([
            torch.Tensor([reward]),
            torch.Tensor(observation),
            torch.Tensor([0.0])  # extra float
        ])
        targets = torch.cat([targets, t.reshape(1, -1)])
        if len(sources) % 1000 == 0:
            if logger is not None:
                logger.info(f'Generated {len(sources)} samples')
        if done:
            observation = env.reset()

            # Add EOS for both sources and targets
            sources = torch.cat([sources, source_eos.reshape(1, -1)])
            targets = torch.cat([targets, target_eos.reshape(1, -1)])
            # SOS to mark beginning of next episode
            sources = torch.cat([sources, source_sos.reshape(1, -1)])
            targets = torch.cat([targets, target_sos.reshape(1, -1)])

    return sources, targets


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


def _train(model,
           optimizer,
           criterion,
           source_batches,
           target_batches,
           sequence_length,
           clip,
           lr,
           epoch,
           device,
           log_interval,
           logger=None):
    # Enable dropout
    model.train()

    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, source_batches.size(0) - 1, sequence_length)):
        # Get source and target sequences
        source = get_ith_batch(source_batches, i, count=sequence_length).to(device)
        target = get_ith_batch(target_batches, i, count=sequence_length).to(device)

        if source.shape[0] != sequence_length:
            print('foo')

        # Pad here
        pad_value = float('inf')
        # FIXME if source.shape[0] != sequence_length:
        pad = (0, 0, 0, 0, 0, sequence_length - source.shape[0])
        source = F.pad(input=source, pad=pad, value=pad_value)
        target = F.pad(input=target, pad=pad, value=pad_value)

        assert source.shape[0] == sequence_length
        assert target.shape[0] == sequence_length

        # Shift target right 1 token
        tgt_inp = target[:-1]

        # Masks
        (
            source,
            tgt_inp,
            src_mask,
            tgt_mask,
            src_key_padding_mask,
            tgt_key_padding_mask,
        ) = _create_masks(source, tgt_inp, model, sequence_length, device)

        target[target == float('inf')] = 0.0

        # Get output from model
        output = model.forward(src=source,
                               tgt=tgt_inp,
                               src_mask=src_mask,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_key_padding_mask
                               )

        optimizer.zero_grad()

        tgt_out = target[1:]

        loss = criterion(output, tgt_out)
        loss.backward()

        optimizer.step()

        # TODO Confirm if this is needed for transformers
        # Prevent exploding gradients problem in RNNS/LSTMs
        # nn.utils.clip_grad_norm(model.parameters(), clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad)

        # Accumulate loss
        total_loss += loss.item()

        # Logging
        if batch > 0 and batch % log_interval == 0:
            if logger is not None:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time

                logger.info(f"Epoch {epoch:3d}"
                            f" | {batch:5d}/{len(source_batches) // sequence_length:5d} batches"
                            f" | lr {lr:02.5f}"
                            f" | ms/batch {elapsed * 1000 / log_interval:5.2f}"
                            f" | loss= {cur_loss:5.4f}"
                            f" | ppl= {math.exp(cur_loss):8.4f}")

                total_loss = 0.0
                start_time = time.time()

    return model, optimizer


def _create_masks(source, target, model, sequence_length, device):
    src_key_padding_mask = None
    tgt_key_padding_mask = None

    tgt_mask = model.transformer.generate_square_subsequent_mask(target.shape[0]).to(device)
    src_mask = torch.zeros((source.shape[0], source.shape[0]), device=device).type(torch.bool)

    src_key_padding_mask = (source == float('inf'))[:, :, 0].transpose(0, 1)
    tgt_key_padding_mask = (target == float('inf'))[:, :, 0].transpose(0, 1)

    # remove the inf
    source[source == float('inf')] = 0.0
    target[target == float('inf')] = 0.0

    return source, target, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask


def _evaluate(model, criterion, source_batches, target_batches, sequence_length, device, logger=None):
    model.eval()
    total_loss = 0.0

    # log_i = random.randrange(0, source_batches.size(0) - 1, sequence_length)

    with torch.no_grad():
        for i in range(0, source_batches.size(0) - 1, sequence_length):
            # Get source and target sequences
            source = get_ith_batch(source_batches, i, count=sequence_length).to(device)
            target = get_ith_batch(target_batches, i, count=sequence_length).to(device)

            # Pad here
            pad_value = float('inf')
            pad = (0, 0, 0, 0, 0, sequence_length - source.shape[0])
            source = F.pad(input=source, pad=pad, value=pad_value)
            target = F.pad(input=target, pad=pad, value=pad_value)

            assert source.shape[0] == sequence_length
            assert target.shape[0] == sequence_length

            tgt_input = target[:-1]

            (
                source,
                tgt_inp,
                src_mask,
                tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask
            ) = _create_masks(source, tgt_input, model, sequence_length, device)

            target[target == float('inf')] = 0.0

            output = model(src=source,
                           tgt=tgt_inp,
                           src_mask=src_mask,
                           tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask
                           )

            tgt_out = target[1:]

            # if logger is not None and i == log_i:
            #     logger.debug(f"Sample evaluation"
            #                  f"\n batch number: {i}"
            #                  f"\n output: {output[-1, 0, :]}"
            #                  f"\n target: {tgt_out[-1, 0, :]}")

            total_loss += len(source) * criterion(output, tgt_out).item()

    return model, total_loss / (len(source_batches) - 1)


def train_transformer(generate,
                      env_name,
                      rollouts,
                      data_folder,
                      batch_size,
                      num_features,
                      dropout,
                      sequence_length,
                      epochs,
                      clip,
                      save,
                      device,
                      log_interval,
                      logger) -> torch.Tensor:
    # TODO create a folder for each environment in the dataset save path
    if generate:
        env = gym.make(env_name)
        data_inputs, data_outputs = generate_dataset2(env, device, rollouts * 50, None, logger)
        save_dataset(data_folder, env_name, data_inputs, data_outputs)
    else:
        data_inputs, data_outputs = load_dataset(data_folder, env_name, logger)

    if logger is not None:
        logger.debug(f'Dataset inputs shape = {data_inputs.shape}')
        logger.debug(f'Dataset outputs shape = {data_outputs.shape}')
        logger.info(f'Dataset contains {data_inputs.shape[0]} tokens.')

    # Prepare train/validation/test split
    split = 0.8, 0.1, 0.1
    train_inputs, valid_inputs, test_inputs = split_dataset(data_inputs, split)
    train_outputs, valid_outputs, test_outputs = split_dataset(data_outputs, split)

    train_inputs_batches = to_column_batches(train_inputs, batch_size, device)
    train_outputs_batches = to_column_batches(train_outputs, batch_size, device)
    valid_inputs_batches = to_column_batches(valid_inputs, batch_size, device)
    valid_outputs_batches = to_column_batches(valid_outputs, batch_size, device)
    test_inputs_batches = to_column_batches(test_inputs, batch_size, device)
    test_outputs_batches = to_column_batches(test_outputs, batch_size, device)

    # FIXME more transformer parameters as arguments: heads, layers, etc..
    model = CustomTransformer(src_dim=train_inputs.shape[1],
                              tgt_dim=train_outputs.shape[1],
                              d_model=num_features,
                              nhead=2,
                              num_encoder_layers=2,
                              num_decoder_layers=2,
                              dim_feedforward=1024,
                              dropout=dropout,
                              max_seq_length=sequence_length
                              ).to(device)
    if logger is not None:
        logger.debug(f'Built transformer model: "{model}"')

    model = train_with_batches(epochs=epochs,
                               sequence_length=sequence_length,
                               clip=clip,
                               device=device,
                               model=model,
                               train_inputs_batches=train_inputs_batches,
                               train_outputs_batches=train_outputs_batches,
                               valid_inputs_batches=valid_inputs_batches,
                               valid_outputs_batches=valid_outputs_batches,
                               test_inputs_batches=test_inputs_batches,
                               test_outputs_batches=test_outputs_batches,
                               save=save,
                               log_interval=log_interval,
                               logger=logger)

    return model


def train_with_batches(epochs,
                       sequence_length,
                       clip,
                       device,
                       model,
                       optimizer,
                       train_inputs_batches,
                       train_outputs_batches,
                       valid_inputs_batches,
                       valid_outputs_batches,
                       test_inputs_batches,
                       test_outputs_batches,
                       save,
                       log_interval,
                       logger=None) -> torch.Tensor:
    criterion = nn.MSELoss()
    lr = 5.0
    best_val_loss = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Training
        model, tf_optimizer = _train(model=model,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     source_batches=train_inputs_batches,
                                     target_batches=train_outputs_batches,
                                     sequence_length=sequence_length,
                                     clip=clip,
                                     lr=lr,
                                     epoch=epoch,
                                     device=device,
                                     log_interval=log_interval,
                                     logger=logger)

        # Validation
        model, val_loss = _evaluate(model=model,
                                    criterion=criterion,
                                    source_batches=valid_inputs_batches,
                                    target_batches=valid_outputs_batches,
                                    sequence_length=sequence_length,
                                    device=device,
                                    logger=logger)

        logger.info('=' * 89)
        logger.info(f'End of epoch {epoch:3d}'
                    f' | time: {(time.time() - epoch_start_time):5.3f}s'
                    f' | valid loss {val_loss:5.4f}'
                    f' | valid ppl {math.exp(val_loss):8.4f}')
        logger.info('=' * 89)

        # TODO stop early if val_loss does not improve after N epochs

        if not best_val_loss or val_loss < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

    # Test data
    model, test_loss = _evaluate(model=model,
                                 criterion=criterion,
                                 source_batches=test_inputs_batches,
                                 target_batches=test_outputs_batches,
                                 sequence_length=sequence_length,
                                 logger=logger,
                                 device=device)

    logger.info(f'End of training | test loss {test_loss:5.4f} | test ppl {math.exp(test_loss):8.4f}')

    return model


if __name__ == '__main__':
    from utils import set_random_seed, get_device, create_logger

    seed = 42
    cuda = True
    logger = create_logger(name='tf')

    set_random_seed(seed, logger)
    device = get_device(cuda, logger)

    env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    model = train_transformer(generate=True,
                              env_name=env_name,
                              rollouts=5_000,
                              clip=0.25,
                              epochs=40,
                              log_interval=50,
                              save=f'model-{env_name}.pt',
                              data_folder='./data',
                              batch_size=64,
                              num_features=256,
                              dropout=0.1,
                              sequence_length=100,
                              device=device,
                              logger=logger)
