"""PyTorch DeepMellow learner for CARLTON-style distributed DCA."""

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def masking(obs):
    """Eq. (15): invalid-action masking using QV-derived channel quality."""
    mask = []
    return_same_channel = True
    for m in obs:
        if m == -1 or m == 0:
            mask.append(-1e9)
        else:
            mask.append(0.0)
            return_same_channel = False
    return np.asarray(mask, dtype=np.float32), return_same_channel


class QResNet(nn.Module):
    """Simple residual MLP that mirrors the legacy dense stack."""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, num_layers - 1))])
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.act(self.in_layer(x))
        for layer in self.hidden:
            x_next = self.act(layer(x))
            x = x + x_next
        return self.out_layer(x)


class DeepMellow(object):
    def __init__(
        self,
        net,
        number_of_actions,
        gamma,
        lr=0.00025,
        optimizer=torch.optim.Adam,
        los_func=None,
        mellowmax_constant=0.02,
        l2_regularization=0.0,
        device=None,
    ):
        self.gamma = float(gamma)
        self.K = int(number_of_actions)
        self.w = float(mellowmax_constant)
        self.l2_regularization = float(l2_regularization or 0.0)
        self.number_of_bits = int(np.floor(np.log2(number_of_actions)) + 1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = net.to(self.device)
        self.loss_func = los_func if los_func is not None else nn.HuberLoss(delta=1.0)
        self.optimizer = optimizer(self.net.parameters(), lr=lr)

    def _flatten_state(self, x):
        # Expected shape: (B, channels_with_bits, 1, history)
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)
        return x.reshape(x.shape[0], -1)

    def forward(self, Z, training=True):
        self.net.train(mode=training)
        x = self._flatten_state(Z)
        return self.net(x)

    def predict(self, x, training=False):
        with torch.no_grad():
            return self.forward(x, training=training)

    def sample_action(self, x, eps, training=True):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)
        x_np = np.expand_dims(x_np, axis=0)

        q_values = self.predict(x_np, training=False).squeeze(0)
        obs = x_np[0, self.number_of_bits:, 0, -1]
        mask, return_same_channel = masking(obs)

        if return_same_channel:
            channel_binary = x_np[0, : self.number_of_bits, 0, -1]
            binary_str = "".join(str(int(i)) for i in channel_binary)
            return int(binary_str, base=2)

        logits = q_values + torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        probs = F.softmax(logits, dim=-1)

        alpha = 0.0
        probs = probs * (1.0 - alpha) + alpha / self.K

        if training and np.random.random() < float(eps):
            action = torch.distributions.Categorical(probs=probs).sample().item()
        else:
            action = torch.argmax(probs).item()

        return int(action)

    def mellowMax(self, x, axis=1):
        """Section III-D mellowmax target operator."""
        c, _ = torch.max(x, dim=axis, keepdim=True)
        exp_x = torch.exp((x - c) * self.w)
        mean_i = torch.mean(exp_x, dim=axis, keepdim=True)
        log_i = torch.log(mean_i)
        out = log_i / self.w + c
        return out.squeeze(axis)

    def learn(self, experience_replay_buffer, batch_size=64):
        states, actions, rewards, next_states, dones, _ = experience_replay_buffer.get_minibatch()
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device)

        q_values = self.forward(states_t, training=True)
        selected_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.forward(next_states_t, training=False)
            next_q = self.mellowMax(next_qs, axis=1)
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_func(selected_q, targets)
        if self.l2_regularization > 0.0:
            l2 = sum(torch.sum(p * p) for p in self.net.parameters()) / max(1, len(list(self.net.parameters())))
            loss = loss + self.l2_regularization * l2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def load_given_weights(self, given_weights):
        self.net.load_state_dict(given_weights, strict=True)

    def get_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict, strict=True)

    def get_trainable_params(self):
        return list(self.net.parameters())


def build_deepmellow(
    num_channels_with_bits,
    number_of_actions,
    gamma,
    lr=0.00025,
    mellowmax_constant=0.02,
    number_of_layers=3,
    number_of_nodes=128,
    l2_regularization=0.0,
    device=None,
):
    net = QResNet(
        input_dim=int(num_channels_with_bits),
        output_dim=int(number_of_actions),
        hidden_dim=int(number_of_nodes),
        num_layers=int(number_of_layers),
    )
    return DeepMellow(
        net=net,
        number_of_actions=number_of_actions,
        gamma=gamma,
        lr=lr,
        mellowmax_constant=mellowmax_constant,
        l2_regularization=l2_regularization,
        device=device,
    )