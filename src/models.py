import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from contextlib import nullcontext

from rlpyt.models.utils import update_state_dict
from rlpyt.utils.tensor import select_at_indexes
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.utils import count_parameters, get_augmentation, from_categorical, find_weight_norm, update_state_dict_compat
from src.networks import *

import copy
EPS = 1e-6  # (NaN-guard)


class SPRCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms,
            dueling,
            jumps,
            spr,
            augmentation,
            target_augmentation,
            eval_augmentation,
            dynamics_blocks,
            norm_type,
            noisy_nets,
            aug_prob,
            projection,
            imagesize,
            dqn_hidden_size,
            momentum_tau,
            renormalize,
            q_l1_type,
            dropout,
            predictor,
            rl,
            bc,
            bc_from_values,
            goal_rl,
            goal_n_step,
            noisy_nets_std,
            residual_tm,
            inverse_model,
            encoder,
            goal_conditioning_type,
            resblock="inverted",
            expand_ratio=2,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            freeze_encoder=False,
            share_l1=False,
            cnn_scale_factor=1,
            blocks_per_group=3,
            ln_for_rl_head=False,
            state_dict=None,
            conv_goal=True,
            goal_all_to_all=False,
            load_head_to=1,
            load_compat_mode=False,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.noisy = noisy_nets
        self.aug_prob = aug_prob
        self.projection_type = projection

        self.dqn_hidden_size = dqn_hidden_size

        if resblock == "inverted":
            resblock = InvertedResidual
        else:
            resblock = Residual

        self.transforms = get_augmentation(augmentation, imagesize)
        self.target_transforms = get_augmentation(target_augmentation, imagesize)
        self.eval_transforms = get_augmentation(eval_augmentation, imagesize)

        self.dueling = dueling
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        if encoder == "resnet":
            self.conv = ResnetCNN(in_channels,
                                  depths=[int(32*cnn_scale_factor),
                                          int(64*cnn_scale_factor),
                                          int(64*cnn_scale_factor)],
                                  strides=[3, 2, 2],
                                  norm_type=norm_type,
                                  blocks_per_group=blocks_per_group,
                                  resblock=resblock,
                                  expand_ratio=expand_ratio,)

        elif encoder.lower() == "normednature":
            self.conv = Conv2dModel(
                in_channels=in_channels,
                channels=[int(32*cnn_scale_factor),
                          int(64*cnn_scale_factor),
                          int(64*cnn_scale_factor)],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[0, 0, 0],
                use_maxpool=False,
                dropout=dropout,
                norm_type=norm_type,
            )
        else:
            self.conv = Conv2dModel(
                in_channels=in_channels,
                channels=[int(32*cnn_scale_factor),
                          int(64*cnn_scale_factor),
                          int(64*cnn_scale_factor)],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[0, 0, 0],
                use_maxpool=False,
                dropout=dropout,
            )

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.latent_shape = fake_output.shape[1:]
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.renormalize = init_normalization(self.hidden_size, renormalize)

        self.jumps = jumps
        self.rl = rl
        self.bc = bc
        self.bc_from_values = bc_from_values
        self.goal_n_step = goal_n_step
        self.use_spr = spr
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size

        self.head = GoalConditionedDuelingHead(self.hidden_size,
                                               output_size,
                                               hidden_size=self.dqn_hidden_size,
                                               pixels=self.pixels,
                                               noisy=self.noisy,
                                               conv_goals=conv_goal,
                                               goal_all_to_all=goal_all_to_all,
                                               share_l1=share_l1,
                                               n_atoms=n_atoms,
                                               ln_for_dqn=ln_for_rl_head,
                                               conditioning_type=goal_conditioning_type,
                                               std_init=noisy_nets_std)

        # Gotta initialize this no matter what or the state dict won't load
        self.dynamics_model = TransitionModel(channels=self.hidden_size,
                                              num_actions=output_size,
                                              hidden_size=self.hidden_size,
                                              blocks=dynamics_blocks,
                                              norm_type=norm_type,
                                              resblock=resblock,
                                              expand_ratio=expand_ratio,
                                              renormalize=self.renormalize,
                                              residual=residual_tm)

        self.momentum_tau = momentum_tau
        if self.projection_type == "mlp":
            self.projection = nn.Sequential(
                                        nn.Flatten(-3, -1),
                                        nn.Linear(self.pixels*self.hidden_size, 512),
                                        TransposedBN1D(512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256)
                                        )
            self.target_projection = self.projection
            projection_size = 256
        elif self.projection_type == "q_l1":
            if goal_rl:
                layers = [self.head.goal_linears[0], self.head.goal_linears[2]]
            else:
                layers = [self.head.rl_linears[0], self.head.rl_linears[2]]
            self.projection = QL1Head(layers, dueling=dueling, type=q_l1_type)
            projection_size = self.projection.out_features
        else:
            projection_size = self.pixels*self.hidden_size

        self.target_projection = self.projection
        self.target_projection = copy.deepcopy(self.target_projection)
        self.target_encoder = copy.deepcopy(self.conv)
        for param in (list(self.target_encoder.parameters()) +
                      list(self.target_projection.parameters())):
            param.requires_grad = False

        if self.bc and not self.bc_from_values:
            self.bc_head = nn.Sequential(nn.ReLU(),
                                         nn.Linear(projection_size, output_size))

        # Gotta initialize this no matter what or the state dict won't load
        if predictor == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(projection_size, projection_size*2),
                TransposedBN1D(projection_size*2),
                nn.ReLU(),
                nn.Linear(projection_size*2, projection_size)
            )
        elif predictor == "linear":
            self.predictor = nn.Sequential(
                nn.Linear(projection_size, projection_size),
            )
        elif predictor == "none":
            self.predictor = nn.Identity()

        self.use_inverse_model = inverse_model
        # Gotta initialize this no matter what or the state dict won't load
        self.inverse_model = InverseModelHead(projection_size,
                                              output_size,)

        print("Initialized model with {} parameters; CNN has {}.".format(count_parameters(self), count_parameters(self.conv)))
        print("Initialized CNN weight norm is {}".format(find_weight_norm(self.conv.parameters()).item()))

        if state_dict is not None:
            if load_compat_mode:
                state_dict = update_state_dict_compat(state_dict, self.state_dict())
            self.load_state_dict(state_dict)
            print("Loaded CNN weight norm is {}".format(find_weight_norm(self.conv.parameters()).item()))
            if rl:
                self.head.copy_base_params(up_to=load_head_to)
                self.head.reset_noise_params()

        self.frozen_encoder = freeze_encoder
        if self.frozen_encoder:
            self.freeze_encoder()

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    def freeze_encoder(self):
        print("Freezing CNN")
        for param in self.conv.parameters():
            param.requires_grad = False

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def do_spr_loss(self, pred_latents, targets, observation):
        pred_latents = self.predictor(pred_latents)

        targets = targets.view(-1, observation.shape[1],
                               self.jumps+1,
                               targets.shape[-1]).transpose(1, 2)
        latents = pred_latents.view(-1, observation.shape[1],
                                    self.jumps+1,
                                    pred_latents.shape[-1]).transpose(1, 2)

        spr_loss = self.spr_loss(latents, targets).view(-1, observation.shape[1]) # split to batch, jumps

        return spr_loss

    @torch.no_grad()
    def calculate_diversity(self, global_latents, observation):
        global_latents = global_latents.view(observation.shape[1], self.jumps+1, global_latents.shape[-1])[:, 0]
        # shape is jumps, bs, dim
        global_latents = F.normalize(global_latents, p=2., dim=-1, eps=1e-3)
        cos_sim = torch.matmul(global_latents, global_latents.transpose(0, 1))
        mask = 1 - (torch.eye(cos_sim.shape[0], device=cos_sim.device, dtype=torch.float))

        cos_sim = cos_sim*mask
        offset = cos_sim.shape[-1]/(cos_sim.shape[-1] - 1)
        cos_sim = cos_sim.mean()*offset

        return cos_sim

    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = maybe_transform(image, transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, transforms, augment=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        if augment:
            flat_images = images.reshape(-1, *images.shape[-3:])
            processed_images = self.apply_transforms(transforms,
                                                     flat_images)
            processed_images = processed_images.view(*images.shape[:-3],
                                                     *processed_images.shape[1:])
            return processed_images
        else:
            return images

    def split_stem_model_params(self):
        stem_params = list(self.conv.parameters()) + list(self.head.parameters())
        model_params = self.dynamics_model.parameters()

        return stem_params, model_params

    def sort_params(self, params_dict):
        return [params_dict[k] for k in sorted(params_dict.keys())]

    def list_params(self):
        all_parameters = {k: v for k, v in self.named_parameters()}
        conv_params = {k: v for k, v in all_parameters.items() if k.startswith("conv")}
        dynamics_model_params = {k: v for k, v in all_parameters.items() if k.startswith("dynamics_model")}

        q_l1_params = {k: v for k, v in all_parameters.items()
                       if (k.startswith("head.goal_value.0")
                       or k.startswith("head.goal_advantage.0")
                       or k.startswith("head.rl_value.0")
                       or k.startswith("head.rl_advantage.0"))}

        other_params = {k: v for k, v in all_parameters.items() if not
                        (k.startswith("target")
                         or k in conv_params.keys()
                         or k in dynamics_model_params.keys()
                         or k in q_l1_params.keys())}

        return self.sort_params(conv_params), \
               self.sort_params(dynamics_model_params), \
               self.sort_params(q_l1_params), \
               self.sort_params(other_params)

    def stem_forward(self, img, prev_action=None, prev_reward=None):
        """Returns the normalized output of convolutional layers."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        with torch.no_grad() if self.frozen_encoder else nullcontext():
                conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
                conv_out = self.renormalize(conv_out)
        return conv_out

    def head_forward(self,
                     conv_out,
                     prev_action,
                     prev_reward,
                     goal=None,
                     logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        p = self.head(conv_out, goal)

        if logits:
            p = F.log_softmax(p, dim=-1)
        else:
            p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    @torch.no_grad()
    def encode_targets(self, target_images, project=True):
        target_images = self.transform(target_images, self.transforms, True)
        target_latents = self.target_encoder(target_images.flatten(0, 1))
        target_latents = self.renormalize(target_latents)
        if project:
            proj_latents = self.target_projection(target_latents)
            proj_latents = proj_latents.view(target_images.shape[0],
                                             target_images.shape[1],
                                             -1)
            return proj_latents, target_latents.view(target_images.shape[0],
                                                     target_images.shape[1],
                                                     -1)
        else:
            return target_latents.view(target_images.shape[0],
                                       target_images.shape[1],
                                       -1)

    def encode_online(self, images, project=True):
        images = self.transform(images, self.transforms, True)
        latents = self.conv(images.flatten(0, 1))
        latents = self.renormalize(latents)
        if project:
            proj_latents = self.projection(latents)
            proj_latents = proj_latents.view(images.shape[0],
                                             images.shape[1],
                                             -1)
            return proj_latents, latents.view(images.shape[0],
                                              images.shape[1],
                                              -1)
        else:
            return latents.view(images.shape[0],
                                images.shape[1],
                                -1)

    def forward(self,
                observation,
                prev_action,
                prev_reward,
                goal=None,
                train=False,
                eval=False):
        """
        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        """
        if train:
            pred_latents = []
            input_obs = observation[0].flatten(1, 2)
            input_obs = self.transform(input_obs, self.transforms, augment=True)
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            if self.rl or self.bc_from_values:
                log_pred_ps = self.head_forward(latent,
                                                prev_action[0],
                                                prev_reward[0],
                                                goal=None,
                                                logits=True)
            else:
                log_pred_ps = None

            if goal is not None:
                goal_log_pred_ps = self.head_forward(latent,
                                                     prev_action[0],
                                                     prev_reward[0],
                                                     goal=goal,
                                                     logits=True)
            else:
                goal_log_pred_ps = None

            pred_latents.append(latent)
            if self.jumps > 0:
                for j in range(1, self.jumps + 1):
                    latent = self.step(latent, prev_action[j])
                    pred_latents.append(latent)

            with torch.no_grad():
                to_encode = max(self.jumps+1, self.goal_n_step)
                target_images = observation[:to_encode].transpose(0, 1).flatten(2, 3)
                target_proj, target_latents = self.encode_targets(target_images, project=True)

            pred_latents = torch.stack(pred_latents, 1)
            proj_latents = self.projection(pred_latents)
            if self.use_spr:
                spr_loss = self.do_spr_loss(proj_latents.flatten(0, 1),
                                            target_proj.flatten(0, 1),
                                            observation)
            else:
                spr_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latent.device)

            if self.bc:
                if self.bc_from_values:
                    bc_preds = from_categorical(log_pred_ps.exp(), limit=10, logits=False)

                if self.bc and not self.bc_from_values:
                    bc_preds = self.bc_head(proj_latents[:, 0])
            else:
                bc_preds = None

            if self.use_inverse_model:
                stack = torch.cat([proj_latents[:, :-1], target_proj.view(*proj_latents.shape)[:, 1:]], -1)
                pred_actions = self.inverse_model(stack.flatten(0, 1))
                pred_actions = pred_actions.view(stack.shape[0], stack.shape[1], *pred_actions.shape[1:])
                pred_actions = pred_actions.transpose(0, 1)
                inv_model_loss = F.cross_entropy(pred_actions.flatten(0, 1),
                                                 prev_action[1:self.jumps + 1].flatten(0, 1), reduction="none")
                inv_model_loss = inv_model_loss.view(*pred_actions.shape[:-1]).mean(0)
            else:
                inv_model_loss = torch.zeros_like(spr_loss).mean(0)

            diversity = self.calculate_diversity(proj_latents, observation)
            update_state_dict(self.target_encoder,
                              self.conv.state_dict(),
                              self.momentum_tau)
            update_state_dict(self.target_projection,
                              self.projection.state_dict(),
                              self.momentum_tau)

            return log_pred_ps,\
                   goal_log_pred_ps,\
                   spr_loss, \
                   target_latents, \
                   target_proj, \
                   diversity, \
                   inv_model_loss, \
                   bc_preds,

        else:
            observation = observation.flatten(-4, -3)

            transforms = self.eval_transforms if eval else self.target_transforms
            img = self.transform(observation, transforms, len(transforms) > 0)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            conv_out = self.renormalize(conv_out)
            p = self.head(conv_out, goal)

            p = F.softmax(p, dim=-1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def select_action(self, obs):
        if self.bc_from_values or not self.bc:
            value = self.forward(obs, None, None, train=False, eval=True)
            value = from_categorical(value, logits=False, limit=10)
        else:
            observation = obs.flatten(-4, -3)
            img = self.transform(observation, self.eval_transforms, len(self.eval_transforms) > 0)
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            conv_out = self.renormalize(conv_out)
            proj = self.projection(conv_out)
            value = self.bc_head(proj)
            value = restore_leading_dims(value, lead_dim, T, B)
        return value

    def step(self, state, action):
        next_state = self.dynamics_model(state, action)
        return next_state


class QL1Head(nn.Module):
    def __init__(self, layers, dueling=False, type=""):
        super().__init__()
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.relu = "relu" in type

        self.encoders = nn.ModuleList(layers)
        self.out_features = sum([encoder.out_features for encoder in self.encoders])

    def forward(self, x):
        x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


def maybe_transform(image, transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * image
        return processed_images

