import gzip
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
from rlpyt.utils.collections import namedarraytuple
import torch
from torch.utils.data import DataLoader, Dataset
import os

from .rlpyt_atari_env import AtariEnv

from torch._six import int_classes as _int_classes
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)


OfflineSamples = namedarraytuple("OfflineSamples", ["all_observation", "all_action", "all_reward", "done", "done_n", "return_"])

class DQNReplayDataset(Dataset):
  def __init__(self, data_path: Path, tmp_data_path: Path, game: str, checkpoint: int, frames: int, k_step: int, max_size: int, full_action_set: bool, dataset_on_gpu: bool, dataset_on_disk: bool) -> None:
    data = []
    self.dataset_on_disk = dataset_on_disk
    assert not (dataset_on_disk and dataset_on_gpu)
    for filetype in ['reward', 'action', 'terminal', 'observation']:
      filename = Path(data_path / f'{game}/{filetype}_{checkpoint}.gz')
      print(f'Loading {filename}')

      # There's no point in putting rewards, actions and terminals on disk.
      # They're tiny and it'll just cause more I/O.
      on_disk = dataset_on_disk and filetype == "observation"

      g = gzip.GzipFile(filename=filename)
      data__ = np.load(g)
      if filetype == "reward":
        self.has_parallel_envs = len(data__.shape) > 1
        if self.has_parallel_envs:
            self.n_envs = data__.shape[1]
        else:
            self.n_envs = 1
      if not self.has_parallel_envs:
        data__ = np.expand_dims(data__, 1)

      data___ = np.copy(data__[:max_size])
      print(f'Using {data___.size * data___.itemsize} bytes')
      if not on_disk:
        del data__
        data_ = torch.from_numpy(data___)
      else:
        new_filename = os.path.join(tmp_data_path, Path(os.path.basename(filename)[:-3]+".npy"))
        print("Stored on disk at {}".format(new_filename))
        np.save(new_filename, data___,)
        del data___
        del data__
        data_ = np.load(new_filename, mmap_mode="r+")

      if (filetype == 'action') and full_action_set:
        action_mapping = dict(zip(data_.unique().numpy(),
                                  AtariEnv(re.sub(r'(?<!^)(?=[A-Z])', '_', game).lower()).ale.getMinimalActionSet()))
        data_.apply_(lambda x: action_mapping[x])
      if dataset_on_gpu:
        print("Stored on GPU")
        data_ = data_.cuda(non_blocking=True)
        del data___
      data.append(data_)

    self.game = game
    self.rewards = data[0]
    self.actions = data[1]
    self.terminal = data[2]
    self.observations = data[3]
    self.f = frames
    self.k = k_step
    self.size = min(self.actions.shape[0], max_size)
    self.effective_size = (self.size - self.f - self.k + 1)

  def __len__(self) -> int:
    return self.effective_size*self.n_envs

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_ind = index // self.effective_size
    time_ind = index % self.effective_size
    sl = slice(time_ind, time_ind+self.f+self.k)
    if self.dataset_on_disk:
        obs = torch.from_numpy(self.observations[sl, batch_ind])
    else:
        obs = (self.observations[sl, batch_ind])
    return tuple([obs,
                  self.actions[sl, batch_ind],
                  self.rewards[sl, batch_ind],
                  self.terminal[sl, batch_ind],
                  ])


class MultiDQNReplayDataset(Dataset):
  def __init__(self, data_path: Path, tmp_data_path: Path, games: List[str], checkpoints: List[int], frames: int, k_step: int, max_size: int, full_action_set: bool, dataset_on_gpu: bool, dataset_on_disk: bool) -> None:
    self.games = [DQNReplayDataset(data_path, tmp_data_path, game, ckpt, frames, k_step, max_size, full_action_set, dataset_on_gpu, dataset_on_disk) for ckpt in checkpoints for game in games]

  def __len__(self) -> int:
    return len(self.games) * len(self.games[0])

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    game_index = index % len(self.games)
    index = index // len(self.games)
    return self.games[game_index][index]


def sanitize_batch(batch: OfflineSamples) -> OfflineSamples:
    has_dones, inds = torch.max(batch.done, 0)
    for i, (has_done, ind) in enumerate(zip(has_dones, inds)):
        if not has_done:
            continue
        batch.all_observation[ind+1:, i] = batch.all_observation[ind, i]
        batch.all_reward[ind+1:, i] = 0
        batch.return_[ind+1:, i] = 0
        batch.done_n[ind+1:, i] = True
    return batch


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
    return_[:] = reward[:rlen]  # 1-step return is current reward.
    done_n[:] = done[:rlen]  # True at time t if done any time by t + n - 1

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

def get_offline_dataloaders(
  *,
  data_path: Path,
  tmp_data_path: Path,
  games: List[str],
  checkpoints: List[int],
  frames: int,
  k_step: int,
  n_step_return: int,
  discount: float,
  samples: int,
  test_game: str,
  test_samples: int,
  dataset_on_gpu: bool,
  dataset_on_disk: bool,
  batch_size: int,
  full_action_set: bool,
  num_workers: int,
  pin_memory: bool,
  prefetch_factor: int,
  **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
  def collate(batch):
    #batch = list(filter(lambda x: not x[3].any(), batch)) # filter samples with a terminal state
    observation, action, reward, done = torch.utils.data.dataloader.default_collate(batch)
    observation = torch.einsum('bthw->tbhw', observation).unsqueeze(2).repeat(1, 1, frames, 1, 1)
    for i in range(1, frames):
        observation[:, :, i] = observation[:, :, i].roll(-i, 0)
    observation = observation[:-frames].unsqueeze(3) # tbfchw
    action = torch.einsum('bt->tb', action)[frames-1:].long()
    reward = torch.einsum('bt->tb', reward)[frames:]
    done = torch.einsum('bt->tb', done)[frames:].bool()
    return_, done_n = discount_return_n_step(reward, done, n_step_return, discount)
    return sanitize_batch(OfflineSamples(observation, action, reward, done[:-n_step_return], done_n, return_))

  dataset = MultiDQNReplayDataset(data_path, tmp_data_path, games, checkpoints, frames, k_step, samples, full_action_set, dataset_on_gpu, dataset_on_disk)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate, drop_last=True, prefetch_factor=prefetch_factor)

  #test_dataset = DQNReplayDataset(data_path, test_game, frames, k_step, test_samples, dataset_on_gpu)
  #test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate, drop_last=True)
  
  #random_dataset = DQNReplayDataset(data_path, test_game, frames, k_step, test_samples, dataset_on_gpu)
  #random_dataset.observations.random_(0, 255)
  #random_dataset.actions.random_(random_dataset.actions.min(), random_dataset.actions.max())
  #random_dataloader = DataLoader(random_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate, drop_last=True)
  
  return dataloader, None, None #test_dataloader, random_dataloader


class CacheEfficientSampler(torch.utils.data.Sampler):

    def __init__(self, num_blocks, block_len, num_repeats=20):
        self.num_blocks = num_blocks
        self.block_len = block_len  # For now, assume all have same length
        self.num_repeats = num_repeats

    def num_samples(self) -> int:
        # dataset size might change at runtime
        return self.block_len*self.num_blocks

    def __iter__(self):
        n = self.num_samples()
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        self.block_ids = [np.arange(self.num_blocks)] * (self.block_len // self.num_repeats)

        blocks = torch.randperm(n//self.num_repeats, generator=generator) % self.num_blocks

        subsamplers = [torch.utils.data.SubsetRandomSampler(torch.arange(i*self.block_len, (i+1)*self.block_len), generator=generator) for i in range(len(self.num_blocks))]

        for block in blocks:
            for i in range(self.num_repeats):
                yield from subsamplers[block]

    def __len__(self):
        return self.num_samples
