import torch
from transformer import generate_dataset


class FakeEnvironment(object):
    def __init__(self, env, model, seq_length, device):
        self.env = env
        self.model = model
        self.seq_length = seq_length

        self.device = device
        self.inputs = torch.Tensor().to(device)
        self.outputs = torch.Tensor().to(device)
        self.last_observation = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec

    def initialize(self):
        # Prefill the history buffer with data from the real environment
        # TODO need only seq_length
        new_sources, new_targets = generate_dataset(self.env,
                                                    num_rollouts=10,
                                                    device=self.device,
                                                    policy=None,  # random
                                                    logger=None)
        src = new_sources[-self.seq_length:, ]
        tgt = new_targets[-self.seq_length:, ]
        # Make S,E into S,N,E
        src = src.unsqueeze(1)
        # Make T,R into T,N,E
        tgt = tgt.unsqueeze(1)

        self.last_observation = tgt[-1, -1, 2:].numpy()

        self.inputs = src.to(self.device)
        self.outputs = tgt.to(self.device)

    def reset(self):
        # Discard old history
        # TODO Remove, initialize replaces them
        # self.inputs = torch.Tensor()
        # self.outputs = torch.Tensor()

        # FIXME keep a distribution of initial observations during training?
        self.env.reset()

        self.initialize()

        return self.last_observation

    def step(self, action):
        # Prepare new input from last observation and the given action
        new_input = torch.cat([
            torch.Tensor(self.last_observation),
            torch.Tensor([action])
        ]).to(self.device)
        # Make S,N,E
        new_input = new_input.reshape(1, 1, -1)

        # Add new input to history buffer
        self.inputs = torch.cat([self.inputs, new_input]).to(self.device)

        # FIXME Clarify
        # Keep inputs and outputs aligned
        # Drop first (oldest) input/output pair
        self.inputs = self.inputs[1:, ]
        self.outputs = self.outputs[1:, ]
        # Add extra output to match input shape, this should be filled by the transformer
        # FIXME zeros? rand?
        self.outputs = torch.cat([
            self.outputs,
            torch.rand(self.outputs[-1:, ].shape).to(self.device)
        ]).to(self.device)

        # Prepare target mask
        tgt_mask = get_attention_mask(self.inputs.size(0))
        tgt_mask = tgt_mask.to(self.device)

        # Forward step to the transformer, predict masked token
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(src=self.inputs,
                                        tgt=self.outputs,
                                        device=self.device,
                                        src_mask=None, # input padding mask
                                        tgt_mask=tgt_mask
                                        )

        # Take last entry, rew+don+obs
        reward = output[-1, -1, 0].item()
        done = output[-1, -1, 1].item()
        done = 1 if done > 0.9 else 0  # FIXME prob. distribution?
        observation = output[-1, -1, 2:].cpu().numpy()

        self.last_observation = observation

        info = None
        return observation, reward, done, info


# FIXME this is also in nn.Transformer
def get_attention_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
