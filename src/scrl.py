import torch.nn.functional as F
import torch
import torch.nn as nn


def exp_distance(x, y, t):
    dist = norm_dist(x, y, t)
    return 1 - torch.exp(-dist)


def norm_dist(x, y, t):
    x = F.normalize(x, p=2, dim=-1, eps=1e-3)
    y = F.normalize(y, p=2, dim=-1, eps=1e-3)
    dist = (x - y).pow(2).sum(-1)
    return t*dist


def calculate_returns(states,
                      goal,
                      distance,
                      gamma,
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

    cum_discounts = nonterminal * gamma
    cum_discounts = cum_discounts.cumprod(-1)

    discounted_rewards = reward_scale*deltas*cum_discounts
    returns = discounted_rewards.cumsum(-1)

    if all_to_all:
        returns = returns.flatten(0, 1)

    return returns.transpose(0, 1)


# Possible sampling schemes:
# 1.  Contrastive: sometimes sample future states in other trajectories
#  Advantage: goal is guaranteed to be a "legal" latent
#             Can just implement as taking other goals in batch.
#  Disadvantage: Might avoid some beneficial exploration in latent space.
# 2.  Purely HER:
#   Downside: likely not to be diverse enough
# 3. HER+noise:  Sample as HER but add noise.
#   Adds diversity, maybe not enough.
# 4.  At random: sample random normalized vectors.
#   Problem: Mostly unreachable, maybe not valid latents.
# Preferred solution: mixture of all methods.
def sample_goals(future_observations, encoder):
    """
    :param future_observations: (batch, jumps, c, h, w)
    :param encoder: map from observations to latents.
    :return: goals, (batch, dim).
    """
    future_observations = future_observations.flatten(2, 3)
    target_time_steps = torch.randint(1, future_observations.shape[0],
                                      future_observations.shape[1:2],
                                      device=future_observations.device)
    target_time_steps = target_time_steps[None, :, None, None, None].expand(-1, -1, *future_observations.shape[2:])

    target_obs = torch.gather(future_observations, 0, target_time_steps)

    goals = encoder(target_obs)
    return goals


def sample_goals_random(batch_size, dim, device):
    goals = F.relu(torch.randn((batch_size, dim), device=device, dtype=torch.float))
    return F.normalize(goals, dim=-1, eps=1e-3)


def add_noise(goals, noise_weight=1):
    noise = F.normalize(torch.randn_like(goals), dim=-1, eps=1e-3)
    weights = torch.rand((goals.shape[0], 1), device=goals.device, dtype=goals.dtype)*noise_weight

    goals = weights * noise + (1 - weights)*goals
    return F.normalize(goals, dim=-1, eps=1e-3)


def permute_goals(goals, permute_probability=0.2):
    """
    :param goals: (batch, dim) matrix of goal states
    :param permute_probability: p in [0, 1] of permuting goals.
    :return: (batch, dim) permuted goals.
    """
    if permute_probability <= 0:
        return goals
    original_indices = torch.arange(0, goals.shape[0], device=goals.device, dtype=torch.long)
    indices = torch.randint_like(original_indices, 0, goals.shape[0])

    permute_mask = torch.rand(indices.shape[0], device=goals.device) < permute_probability
    permute_mask = permute_mask.long()

    new_indices = permute_mask*indices + (1 - permute_mask)*original_indices

    goals = torch.gather(goals, 0, new_indices.unsqueeze(-1).expand(-1, goals.size(-1)))

    return goals


