import torch
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
from torch import nn
import numpy as np
import glob
import gzip
import shutil
from pathlib import Path
import os
EPS = 1e-6


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=tensor.device), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def get_augmentation(augmentation, imagesize):
    if isinstance(augmentation, str):
        augmentation = augmentation.split("_")
    transforms = []
    for aug in augmentation:
        if aug == "affine":
            transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
        elif aug == "rrc":
            transformation = RandomResizedCrop((imagesize, imagesize), (0.8, 1))
        elif aug == "blur":
            transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
        elif aug == "shift" or aug == "crop":
            transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
        elif aug == "intensity":
            transformation = Intensity(scale=0.05)
        elif aug == "none":
            continue
        else:
            raise NotImplementedError()
        transforms.append(transformation)

    return transforms


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


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


def renormalize(tensor, first_dim=-3):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


def extract_epoch(filename):
    """
    Get the epoch from a model save string formatted as name_Epoch:{seed}.pt
    :param str: Model save name
    :return: epoch (int)
    """

    if "epoch" not in filename.lower():
        return 0

    epoch = int(filename.lower().split("epoch_")[-1].replace(".pt", ""))
    return epoch


def get_last_save(base_pattern, retry=True):
    files = glob.glob(base_pattern+"*.pt")
    epochs = [extract_epoch(path) for path in files]

    inds = np.argsort(-np.array(epochs))
    for ind in inds:
        try:
            print("Attempting to load {}".format(files[ind]))
            state_dict = torch.load(Path(files[ind]))
            epoch = epochs[ind]
            return state_dict, epoch
        except Exception as e:
            if retry:
                print("Loading failed: {}".format(e))
            else:
                raise e


def delete_all_but_last(base_pattern, num_to_keep=3):
    files = glob.glob(base_pattern+"*.pt")
    epochs = [extract_epoch(path) for path in files]

    order = np.argsort(np.array(epochs))

    for i in order[:-num_to_keep]:
        os.remove(files[i])
        print("Deleted old save {}".format(files[i]))


def save_model_fn(folder, model_save, seed, use_epoch=True, save_only_last=False):
    def save_model(model, optim, epoch):
        if use_epoch:
            path = Path(f'{folder}/{model_save}_{seed}_epoch_{epoch}.pt')
        else:
            path = Path(f'{folder}/{model_save}_{seed}.pt')

        torch.save({"model": model, "optim": optim}, path)
        print("Saved model at {}".format(path))

        if save_only_last:

            delete_all_but_last(f'{folder}/{model_save}_{seed}')

    return save_model


def find_weight_norm(parameters, norm_type=1.0) -> torch.Tensor:
    r"""Finds the norm of an iterable of parameters.

    The norm is computed over all parameterse together, as if they were
    concatenated into a single vector.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor to find norms of
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == np.inf:
        total_norm = max(p.data.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.data.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def minimal_quantile_loss(pred_values, target_values, taus, kappa=1.0):
    if len(pred_values.shape) == 3:
        output_shape = pred_values.shape[:2]
        target_values = target_values.expand_as(pred_values)
        pred_values = pred_values.flatten(0, 1)
        target_values = target_values.flatten(0, 1)
    else:
        output_shape = pred_values.shape[:1]

    if pred_values.shape[0] != taus.shape[0]:
        # somebody has added states along the batch dimension,
        # probably to do multiple timesteps' losses simultaneously.
        # Since the standard in this codebase is to put time on dimension 1 and
        # then flatten 0 and 1, we can do the same here to get the right shape.
        expansion_factor = pred_values.shape[0]//taus.shape[0]
        taus = taus.unsqueeze(1).expand(-1, expansion_factor, -1,).flatten(0, 1)

    td_errors = pred_values.unsqueeze(-1) - target_values.unsqueeze(1)
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    loss = batch_quantile_huber_loss.squeeze(1)

    # Just use the regular loss as the error for PER, at least for now.
    return loss.view(*output_shape), loss.detach().view(*output_shape)


def scalar_backup(n, returns, nonterminal, qs, discount, select_action=False, selection_values=None):
    """
    :param qs: q estimates
    :param n: n-step
    :param nonterminal:
    :param returns: Returns, already scaled by discount/nonterminal
    :param discount: discount in [0, 1]
    :return:
    """
    if select_action:
        if selection_values is None:
            selection_values = qs
        next_a = selection_values.mean(-1).argmax(-1)
        qs = select_at_indexes(next_a, qs)
    while len(returns.shape) < len(qs.shape):
        returns = returns.unsqueeze(-1)
    while len(nonterminal.shape) < len(qs.shape):
        nonterminal = nonterminal.unsqueeze(-1)
    discount = discount ** n
    qs = nonterminal*qs*discount + returns
    return qs


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def c51_backup(n_step,
               returns,
               nonterminal,
               target_ps,
               select_action=False,
               V_max=10.,
               V_min=10.,
               n_atoms=51,
               discount=0.99,
               selection_values=None):

    z = torch.linspace(V_min, V_max, n_atoms, device=target_ps.device)

    if select_action:
        if selection_values is None:
            selection_values = target_ps
        target_qs = torch.tensordot(selection_values, z, dims=1)  # [B,A]
        next_a = torch.argmax(target_qs, dim=-1)  # [B]
        target_ps = select_at_indexes(next_a.to(target_ps.device), target_ps)  # [B,P']

    delta_z = (V_max - V_min) / (n_atoms - 1)
    # Make 2-D tensor of contracted z_domain for each data point,
    # with zeros where next value should not be added.
    next_z = z * (discount ** n_step)  # [P']
    next_z = nonterminal.unsqueeze(-1)*next_z.unsqueeze(-2)  # [B,P']
    ret = returns.unsqueeze(-1)  # [B,1]

    num_extra_dims = len(ret.shape) - len(next_z.shape)
    next_z = next_z.view(*([1]*num_extra_dims), *next_z.shape)

    next_z = torch.clamp(ret + next_z, V_min, V_max)  # [B,P']

    z_bc = z.view(*([1]*num_extra_dims), 1, -1, 1)  # [1,P,1]
    next_z_bc = next_z.unsqueeze(-2)  # [B,1,P']
    abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
    projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.

    # projection_coeffs is a 3-D tensor: [B,P,P']
    # dim-0: independent data entries
    # dim-1: base_z atoms (remains after projection)
    # dim-2: next_z atoms (summed in projection)

    target_ps = target_ps.unsqueeze(-2)  # [B,1,P']
    if not select_action and len(projection_coeffs.shape) != len(target_ps.shape):
        projection_coeffs = projection_coeffs.unsqueeze(-3)
    target_p = (target_ps * projection_coeffs).sum(-1)  # [B,P]
    target_p = torch.clamp(target_p, EPS, 1)
    return target_p


class DataWriter:
    def __init__(self,
                 save_data=True,
                 data_dir="/project/rrg-bengioy-ad/schwarzm/atari",
                 save_name="",
                 checkpoint_size=1000000,
                 game="Pong",
                 imagesize=(84, 84),
                 mmap=True):

        self.save_name = save_name
        self.save_data = save_data
        if not self.save_data:
            return

        self.pointer = 0
        self.checkpoint = 0
        self.checkpoint_size = checkpoint_size
        self.imagesize = imagesize
        self.dir = Path(data_dir) / game.replace("_", " ").title().replace(" ", "")
        os.makedirs(self.dir, exist_ok=True)
        self.mmap = mmap
        self.reset()

    def reset(self):
        self.pointer = 0
        obs_data = np.zeros((self.checkpoint_size, *self.imagesize), dtype=np.uint8)
        action_data = np.zeros((self.checkpoint_size,), dtype=np.int32)
        reward_data = np.zeros((self.checkpoint_size,), dtype=np.float32)
        terminal_data = np.zeros((self.checkpoint_size,), dtype=np.uint8)

        self.arrays = []
        self.filenames = []

        for data, filetype in [(obs_data, 'observation'),
                               (action_data, 'action'),
                               (reward_data, 'reward'),
                               (terminal_data, 'terminal')]:
            filename = Path(self.dir / f'{filetype}_{self.checkpoint}{self.save_name}.npy')
            if self.mmap:
                np.save(filename, data)
                data_ = np.memmap(filename, mode="w+", dtype=data.dtype, shape=data.shape,)
                del data
            else:
                data_ = data
            self.arrays.append(data_)
            self.filenames.append(filename)

    def save(self):
        for data, filename in zip(self.arrays, self.filenames):
            if not self.mmap:
                np.save(filename, data)
            del data  # Flushes memmap
            with open(filename, 'rb') as f_in:
                new_filename = os.path.join(self.dir, Path(os.path.basename(filename)[:-4]+".gz"))
                with gzip.open(new_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(filename)

    def write(self, samples):
        if not self.save_data:
            return

        self.arrays[0][self.pointer] = samples.env.observation[0, 0, -1, 0]
        self.arrays[1][self.pointer] = samples.agent.action
        self.arrays[2][self.pointer] = samples.env.reward
        self.arrays[3][self.pointer] = samples.env.done

        self.pointer += 1
        if self.pointer == self.checkpoint_size:
            self.checkpoint += 1
            self.save()
            self.reset()


def update_state_dict_compat(osd, nsd):
    updated_osd = {k.replace("head.advantage", "head.goal_advantage").
                   replace("head.value", "head.goal_value").
                   replace("head.secondary_advantage_head", "head.rl_advantage").
                   replace("head.secondary_value_head", "head.rl_value")
                   : v for k, v in osd.items()}
    filtered_osd = {k: v for k, v in updated_osd.items() if k in nsd}
    missing_items = [k for k, v in updated_osd.items() if k not in nsd]
    if len(missing_items) > 0:
        print("Could not load into new model: {}".format(missing_items))
    nsd.update(filtered_osd)
    return nsd


def calculate_true_values(states,
                          goal,
                          distance,
                          gamma,
                          final_value,
                          nonterminal,
                          distance_scale,
                          reward_scale=10.,
                          all_to_all=False):
    """
    :param states: (batch, jumps, dim)
    :param goal:  (batch, dim)
    :param distance: distance function (state X state X scale -> R).
    :param gamma: rl discount gamma in [0, 1]
    :param nonterminal: 1 - done, (batch, jumps).
    :return: returns: discounted sum of rewards up to t, (batch, jumps);
            has shape (batch, batch, jumps) if all_to_all enabled
    """
    nonterminal = nonterminal.transpose(0, 1)

    if all_to_all:
        states = states.unsqueeze(1)
        goal = goal.unsqueeze(0)
        nonterminal = nonterminal.unsqueeze(1)

    goal = goal.unsqueeze(-2)
    distances = distance(states, goal, distance_scale)
    deltas = distances[..., 0:-1] - distances[..., 1:]
    deltas = deltas*reward_scale

    final_values = torch.zeros_like(deltas)
    # final_values[..., -1] = final_value
    # import ipdb; ipdb.set_trace()
    for i in reversed(range(final_values.shape[1]-1)):
        final_values[..., i] = deltas[..., i] + gamma*nonterminal[..., i]*final_values[..., i+1]

    if all_to_all:
        final_values = final_values.flatten(0, 1)

    return final_values.transpose(0, 1)


@torch.no_grad()
def sanity_check_gcrl(states,
                      nonterminal,
                      actions,
                      distance,
                      gamma,
                      distance_scale,
                      reward_scale,
                      network,
                      window=50,
                      conv_goal=True
                      ):
        reps = network.encode_targets(states.flatten(2, 3))
        goal_latents = (reps[1] if conv_goal else reps[0])
        goal = goal_latents[window]

        input_latents = reps[1].view(*reps[1].shape[:-1], -1, 7, 7)
        input_latents = input_latents[:-1]
        spatial_goal = goal.unsqueeze(0)
        spatial_goal = spatial_goal.view(*spatial_goal.shape[:-1], -1, 7, 7).expand_as(input_latents)
        pred_values = network.head_forward(input_latents.flatten(0, 1), None, None, spatial_goal.flatten(0, 1))
        pred_values = pred_values.view(input_latents.shape[0], input_latents.shape[1], *pred_values.shape[1:])

        actions = actions.contiguous()
        pred_values = pred_values.contiguous()
        pred_values = select_at_indexes(actions[:-1], pred_values)
        pred_values = from_categorical(pred_values, limit=10, logits=False)

        returns = calculate_true_values(goal_latents.transpose(0, 1),
                                        goal,
                                        distance,
                                        gamma,
                                        pred_values[-1],
                                        nonterminal[:window],
                                        distance_scale,
                                        reward_scale)

        return pred_values, returns


def discount_return_n_step(reward, done, n_step, discount, return_dest=None,
                           done_n_dest=None, do_truncated=False):
    """Time-major inputs, optional other dimension: [T], [T,B], etc.  Computes
    n-step discounted returns within the timeframe of the of given rewards. If
    `do_truncated==False`, then only compute at time-steps with full n-step
    future rewards are provided (i.e. not at last n-steps--output shape will
    change!).  Returns n-step returns as well as n-step done signals, which is
    True if `done=True` at any future time before the n-step target bootstrap
    would apply (bootstrap in the algo, not here)."""
    rlen = reward.shape[0]
    if not do_truncated:
        rlen -= (n_step - 1)
    return_ = torch.zeros(
        (rlen,) + reward.shape[1:], dtype=reward.dtype, device=reward.device)
    done_n = torch.zeros(
        (rlen,) + reward.shape[1:], dtype=done.dtype, device=done.device)
    return_[:] = reward[:rlen].float()  # 1-step return is current reward.
    done_n[:] = done[:rlen].float()  # True at time t if done any time by t + n - 1

    done_dtype = done.dtype
    done_n = done_n.type(reward.dtype)
    done = done.type(reward.dtype)

    if n_step > 1:
        if do_truncated:
            for n in range(1, n_step):
                return_[:-n] += (discount ** n) * reward[n:n + rlen] * (1 - done_n[:-n])
                done_n[:-n] = torch.max(done_n[:-n], done[n:n + rlen])
        else:
            for n in range(1, n_step):
                return_ += (discount ** n) * reward[n:n + rlen] * (1 - done_n)
                done_n = torch.max(done_n, done[n:n + rlen])  # Supports tensors.
    done_n = done_n.type(done_dtype)
    return return_, done_n

