import torch
from transformer import generate_dataset, _pad_and_masks, generate_dataset2


class FakeEnvironment(object):
    def __init__(self, env, model, seq_length, device):
        self.env = env
        self.model = model.to(device)
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

    def _initialize(self):

        sources, targets = generate_dataset2(self.env,
                                             self.device,
                                             self.seq_length, None, None)

        # Prefill the history buffer with data from the real environment
        #sources = torch.Tensor()
        #targets = torch.Tensor()

        # Generate mini-dataset
        # observation = self.env.reset()
        # while len(sources) < self.seq_length:
        #     action = self.env.action_space.sample()
        #     s = torch.cat([torch.Tensor(observation), torch.Tensor([action])])
        #     sources = torch.cat([sources, s.reshape(1, 1, -1)])
        #     observation, reward, done, _ = self.env.step(action)
        #     t = torch.cat([
        #         torch.Tensor([reward]),
        #         torch.Tensor([done]),
        #         torch.Tensor(observation)
        #     ])
        #     targets = torch.cat([targets, t.reshape(1, 1, -1)])
        #     if done:
        #         observation = self.env.reset()

        self.last_observation = targets[-1, 2:].numpy()

        self.inputs = sources.to(self.device)
        self.outputs = targets.to(self.device)

    def reset(self):
        self._initialize()
        return self.last_observation

    def step(self, action):
        # Prepare new input from last observation and the given action
        new_input = torch.cat([
            torch.Tensor(self.last_observation),
            torch.Tensor([action])
        ]).to(self.device)
        # Make S,N,E with batch size 1
        new_input = new_input.reshape(1, -1)

        # Add new input to history buffer
        self.inputs = torch.cat([self.inputs, new_input]).to(self.device)

        # Keep a rolling window of seq_length, keep inputs and outputs aligned
        if self.inputs.size(0) > self.seq_length:
            self.inputs = self.inputs[1:, ]
            self.outputs = self.outputs[1:, ]

        # FIXME shift one right?
        # self.outputs = self.outputs[1:]
        # Prepare target mask
        tgt_mask = get_attention_mask(self.outputs.size(0)).to(self.device)

        # Forward step to the transformer, predict masked token
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(src=self.inputs.unsqueeze(1),
                                        tgt=self.outputs.unsqueeze(1),
                                        tgt_mask=tgt_mask)

        # Prediction is the last item of the sequence
        prediction = output[-1, 0, :]

        # Add to output history
        self.outputs = torch.cat([self.outputs, output[-1,0:]]).to(self.device)

        # Extract elements from prediction
        reward = prediction[0].item()
        done = prediction[1].item()
        done = 1 if done > 0.9 else 0
        observation = prediction[2:].cpu().numpy()

        self.last_observation = observation

        # DEBUG are the envs in sync?
        # true_observation, true_reward, true_done, _ = self.env.step(action)
        # mse = torch.sum((torch.tensor(true_observation) - observation) ** 2)
        # print(f'MSE={mse}')

        info = None
        return observation, reward, done, info


# FIXME this is also in nn.Transformer
def get_attention_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
