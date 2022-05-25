import torch
import torch.nn as nn
import torch.nn.functional as F


from nle import nethack


def get_obs_net_fn(env, observation_space, num_actions):
    if env.startswith('NetHack'):
        return lambda: NetHackNet(observation_space=observation_space, num_actions=num_actions)
    else:
        return lambda: AtariNet(observation_space=observation_space, num_actions=num_actions)


def get_policy(env, observation_space, num_actions, use_lstm=False):
    obs_net_fn = get_obs_net_fn(env, observation_space, num_actions)
    if env.startswith('NetHack'):
        return Policy(obs_net_fn, num_actions, use_lstm, 1)
    else:
        return Policy(obs_net_fn, num_actions, use_lstm, 2)


def get_prediction_net(env, observation_space, num_actions):
    obs_net_fn = get_obs_net_fn(env, observation_space, num_actions)
    if env.startswith('NetHack'):
        return PredictionNet(obs_net_fn, num_actions, ('glyphs', 'blstats'))
    else:
        return PredictionNet(obs_net_fn, num_actions, ('frame'))


class PredictionNet(nn.Module):
    def __init__(self, obs_net_fn, num_actions, obs_keys):
        super(PredictionNet, self).__init__()

        self.obs_net = obs_net_fn()
        output_size = self.obs_net.output_size
        self.num_actions = num_actions
        self.obs_keys = obs_keys

        core_output_size = output_size + num_actions
        self.core_fc = nn.Linear(core_output_size, 512)

        self.reward_head = nn.Linear(512, 1)

    def forward(self, inputs):
        last_inputs = {}
        for key in self.obs_keys:
            if f'last_{key}' in inputs:
                last_inputs[key] = inputs[f'last_{key}']
            else:
                tmp = torch.zeros_like(inputs[key])
                tmp[1:].copy_(inputs[key][:-1])
                last_inputs[key] = tmp

        cur_x = self.obs_net(inputs)
        last_x = self.obs_net(last_inputs)
        x = torch.cat((cur_x[..., :256], last_x[..., 256:]), -1)
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)

        one_hot_action = F.one_hot(
            inputs["action"].view(T * B), self.num_actions
        ).float()

        core_input = torch.cat([x, one_hot_action], dim=-1)
        core_output = F.relu(self.core_fc(core_input)) + x

        reward_prediction = self.reward_head(core_output)
        reward_prediction = reward_prediction.view(T, B)

        return (
            dict(reward=reward_prediction),
            x.view(T, B, -1),
            core_output.view(T, B, -1)
        )


class Policy(nn.Module):
    def __init__(self, obs_net_fn, num_actions: int, use_lstm: bool, lstm_num_layers: int = 2):
        super().__init__()

        self.obs_net = obs_net_fn()
        output_size = self.obs_net.output_size
        self.num_actions = num_actions

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(output_size, output_size, lstm_num_layers)

        self.policy = nn.Linear(output_size, self.num_actions)
        self.baseline = nn.Linear(output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = self.obs_net(inputs)
        T, B, *_ = x.shape

        if self.use_lstm:
            x = x.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(x.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.cat(core_output_list)
        else:
            core_output = x
            core_state = tuple()

        core_output = torch.flatten(core_output, 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
            core_output
        )


class AtariNet(nn.Module):
    def __init__(self, observation_space, num_actions):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_space.shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        self.output_size = self.fc.out_features + num_actions + 1

    def forward(self, inputs):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        return core_input.view(T, B, -1)


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


class NetHackNet(nn.Module):
    def __init__(
        self,
        observation_space,
        num_actions,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super(NetHackNet, self).__init__()

        self.glyph_shape = observation_space["glyphs"].shape
        self.blstats_size = observation_space["blstats"].shape[0]

        self.num_actions = num_actions

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = 512

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim**2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.output_size = self.h_dim

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs):
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # print("crop", crop)
        # print("at_xy", glyphs[:, coordinates[:, 1].long(), coordinates[:, 0].long()])

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        return st.view(T, B, -1)
