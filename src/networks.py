import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.utils import renormalize
from rlpyt.models.utils import scale_grad
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import copy


def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 norm_type, num_layers=1, groups=-1,
                 drop_prob=0., bias=True):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2, 3]
        self.drop_prob = drop_prob

        hidden_dim = round(in_channels * expand_ratio)

        if groups <= 0:
            groups = hidden_dim

        conv = nn.Conv2d

        if stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, stride, stride)
            nn.init.normal_(self.downsample.weight, mean=0, std=
                            np.sqrt(2 / (self.downsample.weight.shape[0] *
                            np.prod(self.downsample.weight.shape[2:]))))
        else:
            self.downsample = False

        if expand_ratio == 1:
            conv1 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv2 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            self.conv = nn.Sequential(
                # dw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv2,
                init_normalization(out_channels, norm_type),
            )
            nn.init.constant_(self.conv[-1].weight, 0)
        else:
            conv1 = conv(in_channels, hidden_dim, 1, 1, 0, bias=bias)
            conv2 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv3 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            fixup_init(conv3, num_layers)
            self.conv = nn.Sequential(
                # pw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # dw
                conv2,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv3,
                init_normalization(out_channels, norm_type)
            )
            if norm_type != "none":
                nn.init.constant_(self.conv[-1].weight, 0)

    def forward(self, x):
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        if self.training and np.random.uniform() < self.drop_prob:
            return identity
        else:
            return identity + self.conv(x)


class Residual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, groups=1)


class ResnetCNN(nn.Module):
    def __init__(self, input_channels,
                 depths=(16, 32, 64),
                 strides=(3, 2, 2),
                 blocks_per_group=3,
                 norm_type="bn",
                 resblock=InvertedResidual,
                 expand_ratio=2,):
        super(ResnetCNN, self).__init__()
        self.depths = [input_channels] + depths
        self.resblock = resblock
        self.expand_ratio = expand_ratio
        self.blocks_per_group = blocks_per_group
        self.layers = []
        self.norm_type = norm_type
        self.num_layers = self.blocks_per_group*len(depths)
        for i in range(len(depths)):
            self.layers.append(self._make_layer(self.depths[i],
                                                self.depths[i+1],
                                                strides[i],
                                                ))
        self.layers = nn.Sequential(*self.layers)
        self.train()

    def _make_layer(self, in_channels, depth, stride,):

        blocks = [self.resblock(in_channels, depth,
                                expand_ratio=self.expand_ratio,
                                stride=stride,
                                norm_type=self.norm_type,
                                num_layers=self.num_layers,)]

        for i in range(1, self.blocks_per_group):
            blocks.append(self.resblock(depth, depth,
                                        expand_ratio=self.expand_ratio,
                                        stride=1,
                                        norm_type=self.norm_type,
                                        num_layers=self.num_layers,))

        return nn.Sequential(*blocks)

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs):
        return self.layers(inputs)


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=0,
                 hidden_size=256,
                 norm_type="bn",
                 renormalize=True,
                 resblock=InvertedResidual,
                 expand_ratio=2,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize

        self.residual = residual
        conv = nn.Conv2d
        self.initial_layer = nn.Sequential(conv(channels+num_actions, hidden_size, 3, 1, 1),
                                           nn.ReLU(), init_normalization(hidden_size, norm_type))
        self.final_layer = nn.Conv2d(hidden_size, channels, 3, 1, 1)
        resblocks = []

        for i in range(blocks):
            resblocks.append(resblock(hidden_size,
                                      hidden_size,
                                      stride=1,
                                      norm_type=norm_type,
                                      expand_ratio=expand_ratio,
                                      num_layers=blocks))
        self.resnet = nn.Sequential(*resblocks)
        if self.residual:
            nn.init.constant_(self.final_layer.weight, 0)
        self.train()

    def forward(self, x, action, blocks=True):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.initial_layer(stacked_image)
        if blocks:
            next_state = self.resnet(next_state)
        next_state = self.final_layer(next_state)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        next_state = self.renormalize(next_state)
        return next_state


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "gn", "max", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "gn":
        groups = max(min(32, channels//4), 1)
        return nn.GroupNorm(groups, channels, affine=affine)
    elif type == "max":
        if not one_d:
            return renormalize
        else:
            return lambda x: renormalize(x, -1)
    elif type == "none" or type is None:
        return nn.Identity()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('old_bias_epsilon', torch.empty(out_features))
        self.register_buffer('old_weight_epsilon', torch.empty(out_features, in_features))
        self.reset_parameters()
        self.reset_noise()
        self.use_old_noise = False

    def reset_noise_parameters(self):
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if self.bias:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        else:
            self.bias_sigma.fill_(0)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        if self.bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
        else:
            self.bias_mu.fill_(0)

        self.reset_noise_parameters()

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        self.old_bias_epsilon.copy_(self.bias_epsilon)
        self.old_weight_epsilon.copy_(self.weight_epsilon)
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        use_noise = (self.training or self.sampling) if self.noise_override is None else self.noise_override
        if use_noise:
            weight_eps = self.old_weight_epsilon if self.use_old_noise else self.weight_epsilon
            bias_eps = self.old_bias_epsilon if self.use_old_noise else self.bias_epsilon

            return F.linear(input, self.weight_mu + self.weight_sigma * weight_eps,
                            self.bias_mu + self.bias_sigma * bias_eps)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            norm_type="none",
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride, oc in zip(conv_layers, maxp_strides, channels):
            sequence.extend([conv_layer, init_normalization(oc, norm_type), nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)


class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                            NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=std_init)
                            ]
        else:
            self.linears = [nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]
        self.advantage_layers = [nn.Flatten(-3, -1),
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-3, -1),
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_net = nn.Sequential(*self.advantage_layers)
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value_net = nn.Sequential(*self.value_layers)
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input, old_noise=False):
        [setattr(module, "use_old_noise", old_noise) for module in self.modules()]
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value_net(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_net(input)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class GoalConditioning(nn.Module):
    def __init__(self,
                 pixels=49,
                 feature_dim=64,
                 dqn_hidden_size=256,
                 conv=True,
                 film=True,
                 goal_only_conditioning=False,
                 n_heads=2):
        """
        The basic idea: cat the online and goal states as feature maps,
        and then run a two-layer CNN on it.  We then run this through a flatten
        and an MLP to get FiLM weights, which we use in the DQN head.
        """
        super().__init__()
        output_size = n_heads * dqn_hidden_size
        output_size = output_size * 2 if film else output_size
        input_dim = feature_dim if goal_only_conditioning else feature_dim * 2
        self.film = film
        self.goal_only_conditioning = goal_only_conditioning
        self.n_heads = n_heads

        self.conv = conv
        if conv:
            self.network = nn.Sequential(
                nn.Conv2d(input_dim, feature_dim * 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim * 2, feature_dim * 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(-3, -1),
                nn.Linear(pixels * feature_dim * 2, output_size)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(dqn_hidden_size*2, dqn_hidden_size*2),
                nn.ReLU(),
                nn.Linear(dqn_hidden_size*2, dqn_hidden_size*2),
                nn.ReLU(),
                nn.Linear(dqn_hidden_size*2, output_size)
            )

    def forward(self, states, goals):
        if self.conv:
            goals = goals.view(*states.shape)

        if not self.goal_only_conditioning:
            if self.conv:
                goals = F.normalize(goals, dim=(-1, -2, -3), p=2, eps=1e-5)
                states = F.normalize(states, dim=(-1, -2, -3), p=2, eps=1e-5)
            else:
                goals = F.normalize(goals, dim=(-1), p=2, eps=1e-5)
                states = F.normalize(states, dim=(-1), p=2, eps=1e-5)
            input = torch.cat([states, goals], -3)
        else:
            input = goals

        output = self.network(input)

        if self.film:
            # Split the output into heads and biases/scales
            output = output.view(*output.shape[:-1], self.n_heads, 2, -1)
        else:
            output = output.view(*output.shape[:-1], self.n_heads, -1)
        return output


class GoalConditionedDuelingHead(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    """
    For goal conditioning, we have a few options.
    First, we can concatenate the goal after the first linear.  This is the simplest
    solution, but would almost certainly require at least one more linear before
    the output layer, for practical reasons of allowing diversity.

    Alternatively, we could use FiLM or similar.  We could compute FiLM weights
    with one or two layers from the concatenation of goal and state, and then
    apply these to the state before using the final linear.
    Could also do a residual connection: state/goal -> two layers -> state delta.

    Of course, we could define goals as being in the convolutional feature map
    latent space, but that probably sucks (and has very variable size between architectures).    
    """

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=512,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1,
                 ln_for_dqn=True,
                 conv_goals=True,
                 conditioning_type=["goal_only", "film"],
                 share_l1=False,
                 goal_all_to_all=False):
        super().__init__()

        self.goal_conditioner = GoalConditioning(
            pixels=pixels,
            feature_dim=input_channels,
            dqn_hidden_size=hidden_size,
            goal_only_conditioning="goal_only" in conditioning_type,
            film="film" in conditioning_type,
            n_heads=2,
            conv=conv_goals,
        )
        self.conditioning_style = "film" if "film" in conditioning_type else \
            "sum" if "sum" in conditioning_type else "product"

        self.goal_all_to_all = goal_all_to_all

        if noisy:
            self.goal_linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                                 NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                                 NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                                 NoisyLinear(hidden_size, n_atoms, std_init=std_init),
                                 ]
            self.rl_linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                               NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                               NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                               NoisyLinear(hidden_size, n_atoms, std_init=std_init),
                               ]
        else:
            self.goal_linears = [nn.Linear(pixels * input_channels, hidden_size),
                                 nn.Linear(hidden_size, output_size * n_atoms),
                                 nn.Linear(pixels * input_channels, hidden_size),
                                 nn.Linear(hidden_size, n_atoms),
                                 ]
            self.rl_linears = [nn.Linear(pixels * input_channels, hidden_size),
                               nn.Linear(hidden_size, output_size * n_atoms),
                               nn.Linear(pixels * input_channels, hidden_size),
                               nn.Linear(hidden_size, n_atoms),
                               ]

        if share_l1:
            self.rl_linears[0] = self.goal_linears[0]
            self.rl_linears[2] = self.goal_linears[2]

        self.goal_advantage_layers = [self.goal_linears[0],
                                      nn.ReLU(),
                                      nn.LayerNorm(hidden_size, elementwise_affine=False)
                                      if ln_for_dqn else nn.Identity(),
                                      self.goal_linears[1]]
        self.goal_value_layers = [self.goal_linears[2],
                                  nn.ReLU(),
                                  nn.LayerNorm(hidden_size, elementwise_affine=False)
                                  if ln_for_dqn else nn.Identity(),
                                  self.goal_linears[3]]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.rl_advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.goal_value = nn.Sequential(*self.goal_value_layers)
        self.goal_advantage = nn.Sequential(*self.goal_advantage_layers)
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms
        self.noisy = noisy

        self.rl_advantage = nn.Sequential(
            self.rl_linears[0],
            nn.ReLU(),
            nn.LayerNorm(hidden_size) if ln_for_dqn else nn.Identity(),
            self.rl_linears[1],
        )
        self.rl_value = nn.Sequential(
            self.rl_linears[2],
            nn.ReLU(),
            nn.LayerNorm(hidden_size) if ln_for_dqn else nn.Identity(),
            self.rl_linears[3],
        )

    def forward(self, input, goal):
        if goal is None:
            return self.regular_forward(input)

        x = scale_grad(input, self._grad_scale)
        x = x.flatten(-3, -1)
        advantage_hidden = self.goal_advantage[0:3](x)
        value_hidden = self.goal_value[0:3](x)

        goal_conditioning = self.goal_conditioner(input, goal)

        if self.goal_all_to_all:
            goal_conditioning = goal_conditioning.unsqueeze(0)
            advantage_hidden = advantage_hidden.unsqueeze(1)
            value_hidden = value_hidden.unsqueeze(1)

        if self.conditioning_style == "film":
            advantage_biases = goal_conditioning[..., 0, 0, :]
            advantage_scales = goal_conditioning[..., 0, 1, :]
            value_biases = goal_conditioning[..., 1, 0, :]
            value_scales = goal_conditioning[..., 1, 1, :]
            advantage_hidden = advantage_hidden * advantage_scales + advantage_biases
            value_hidden = advantage_hidden * value_scales + value_biases
        elif self.conditioning_style == "product":
            advantage_hidden = advantage_hidden * goal_conditioning[..., 0]
            value_hidden = value_hidden * goal_conditioning[..., 1]
        elif self.conditioning_style == "sum":
            advantage_hidden = advantage_hidden + goal_conditioning[..., 0]
            value_hidden = value_hidden + goal_conditioning[..., 1]

        if self.goal_all_to_all:
            advantage_hidden = advantage_hidden.flatten(0, 1)
            value_hidden = value_hidden.flatten(0, 1)

        advantage = self.goal_advantage[-2:](advantage_hidden)
        advantage = advantage.view(-1, self._output_size, self._n_atoms) + self.advantage_bias
        value = self.goal_value[-2:](value_hidden).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=-2, keepdim=True))

    def regular_forward(self, input):
        x = scale_grad(input, self._grad_scale)
        x = x.flatten(-3, -1)
        advantage = self.rl_advantage(x)
        advantage = advantage.view(-1, self._output_size, self._n_atoms) + self.rl_advantage_bias
        value = self.rl_value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=-2, keepdim=True))

    def copy_base_params(self, up_to=1):
        if up_to == 0:
            return
        self.rl_value[0:up_to].load_state_dict(self.goal_value[0:up_to].state_dict())
        self.rl_advantage[0:up_to].load_state_dict(self.goal_advantage[0:up_to].state_dict())

    def reset_noise(self):
        for module in self.goal_linears:
            module.reset_noise()
        for module in self.rl_linears:
            module.reset_noise()

    def reset_noise_params(self):
        for module in self.goal_linears:
            module.reset_noise_parameters()
        for module in self.rl_linears:
            module.reset_noise_parameters()

    def set_sampling(self, sampling):
        for module in self.goal_linears:
            module.sampling = sampling
        for module in self.rl_linears:
            module.sampling = sampling


class TransposedBN1D(nn.BatchNorm1d):
    def forward(self, x):
        x_flat = x.view(-1, x.shape[-1])
        if self.training and x_flat.shape[0] == 1:
            return x
        x_flat = super().forward(x_flat)
        return x_flat.view(*x.shape)


class InverseModelHead(nn.Module):
    def __init__(self,
                 input_channels,
                 num_actions=18,):
        super().__init__()
        layers = [nn.Linear(input_channels*2, 256),
                  nn.ReLU(),
                  nn.Linear(256, num_actions)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)

