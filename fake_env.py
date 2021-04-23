import torch
from transformer import generate_dataset2


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
                                             self.seq_length,
                                             None,
                                             None)

        # R + O + D marker
        self.last_observation = targets[-1, 1:-1].numpy()

        self.inputs = sources.to(self.device)
        self.outputs = targets.to(self.device)

    def reset(self):
        self._initialize()
        return self.last_observation

    def step(self, action):
        # Prepare new input from last observation and the given action
        new_input = torch.cat([
            torch.Tensor(self.last_observation),
            torch.Tensor([action]),
            torch.Tensor([0.0])
        ]).to(self.device)
        # Make S,N,E with batch size 1
        new_input = new_input.reshape(1, -1)

        # Add new input to history buffer
        self.inputs = torch.cat([self.inputs, new_input]).to(self.device)

        # Keep a rolling window of seq_length, keep inputs and outputs aligned
        while self.inputs.size(0) > self.seq_length:
            self.inputs = self.inputs[1:, ]
            self.outputs = self.outputs[1:, ]

        # Prepare target mask
        tgt_mask = get_attention_mask(self.outputs.size(0)).to(self.device)

        # Forward step to the transformer, predict masked token
        self.model.eval()
        with torch.no_grad():
            # NOTE Expensive call
            output = self.model.forward(src=self.inputs.unsqueeze(1),
                                        tgt=self.outputs.unsqueeze(1),
                                        tgt_mask=tgt_mask)

        # Prediction is the last item of the sequence
        prediction = output[-1, 0, :]

        # Add to output history
        self.outputs = torch.cat([self.outputs, output[-1, 0:]]).to(self.device)

        # Extract elements from prediction
        reward = prediction[0].item()
        done = prediction[-1].item()
        # Only EOS should have a 2.0 in that position
        done = done >= 1.9
        observation = prediction[1:-1].cpu().numpy()

        # FIXME same as in generate_dataset
        if done:
            # Add EOS/SOS for both sources and targets
            input_sos = torch.zeros(self.inputs.size(2))
            input_sos[-1] = 1.0
            output_sos = torch.zeros(self.outputs.size(2).shape)
            output_sos[-1] = 1.0
            input_eos = torch.zeros(self.inputs.size(2).shape)
            input_eos[-1] = 2.0
            output_eos = torch.zeros(self.outputs.size(2).shape)
            output_eos[-1] = 2.0
            self.inputs = torch.cat([self.inputs, input_eos.reshape(1, -1)])
            self.inputs = torch.cat([self.inputs, input_sos.reshape(1, -1)])
            self.outputs = torch.cat([self.outputs, output_eos.reshape(1, -1)])
            self.outputs = torch.cat([self.outputs, output_sos.reshape(1, -1)])

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
