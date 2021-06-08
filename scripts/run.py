from pathlib import Path
from typing import Any, Dict, Optional
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from src.agent import SPRAgent
from src.algos import SPRCategoricalDQN
from src.models import SPRCatDqnModel
from src.rlpyt_atari_env import AtariEnv
from src.rlpyt_utils import OnlineEval, OfflineEval, OneToOneSerialEvalCollector, SerialSampler
from src.utils import get_last_save, save_model_fn


@hydra.main(config_name="config")
def main(args: DictConfig):
    args = OmegaConf.merge(configs['ernbw'], args)
    print(OmegaConf.to_yaml(args))

    # Offline pretraining
    state_dict = None
    start_itr = 0
    if args.model_load is not None:
        if args.load_last:
            state_dict, start_itr = get_last_save(f'{args.model_folder}/{args.model_load}')
            print("Loading checkpoint from {}".format(start_itr))
        else:
            state_dict = torch.load(Path(f'{args.model_folder}/{args.model_load}.pt'))
    elif args.offline_model_save is not None:
        try:
            state_dict, start_itr = get_last_save(f'{args.model_folder}/{args.offline_model_save}_{args.seed}')
            print("Loaded in old model to resume")
        except Exception as e:
            print(e)
            print("Could not load existing pretraining model; not continuing")
            state_dict = None
            start_itr = 0

    if state_dict is not None:
        if "model" in state_dict:
            model_state_dict = state_dict["model"]
            optim_state_dict = state_dict["optim"]
        else:
            model_state_dict = state_dict
            optim_state_dict = None
    else:
        model_state_dict = optim_state_dict = None

    if args.offline.runner.epochs > 0:
        print("Offline pretraining")
        offline_args = OmegaConf.merge(args, args.offline)
        print(OmegaConf.to_yaml(offline_args, resolve=True))
        config: Dict[str, Any] = OmegaConf.to_container(offline_args, resolve=True)

        dl_kwargs = config["runner"]["dataloader"]
        dl_kwargs['data_path'] = Path(dl_kwargs['data_path'])
        k_step_base = dl_kwargs['jumps']+1
        if config["algo"]["goal_weight"] > 0:
            k_step_base = max(k_step_base, config["algo"]['goal_window'])
        dl_kwargs['k_step'] = k_step_base + dl_kwargs['n_step_return']

        if args.offline_model_save is not None:
            save_fn = save_model_fn(args.model_folder, args.offline_model_save, args.seed, args.save_by_epoch, args.save_last_only)
        else:
            save_fn = None

        config["algo"]["min_steps_learn"] = 0
        agent, _, _, _ = train(config, save_fn=save_fn,
                               offline=True,
                               state_dict=model_state_dict,
                               optim_state_dict=optim_state_dict,
                               start_itr=start_itr)

        model_state_dict = agent.model.state_dict()

    if args.runner.n_steps > 0 and args.do_online:
        print("Online training")
        print(OmegaConf.to_yaml(args, resolve=True))
        config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
        config["runner"]["log_interval_steps"] = args.runner.n_steps // args.num_logs
        if args.offline.runner.epochs > 0:
            config["model_load"] = config["offline_model_save"]
        _, _, _, _ = train(config, offline=False,
                           state_dict=model_state_dict,
                           optim_state_dict=optim_state_dict)


def train(config: Dict[str, Any], *, offline: bool,
          state_dict: Optional[Dict[str, torch.Tensor]],
          optim_state_dict: Optional[Dict[str, torch.Tensor]],
          save_fn=None,
          start_itr=0):
    if config["public"]:
        wandb.init(config=config, project="SGI", group="offline" if offline else "online", reinit=True, anonymous="allow")
    else:
        wandb.init(config=config, **config["wandb"], group="offline" if offline else "online", reinit=True)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if state_dict is not None:
        print("Initializing with pretrained model")
        config["agent"]["model_kwargs"]["state_dict"] = state_dict
    if offline and optim_state_dict is not None:
        print("Initializing optimizer with previous settings")
        config["algo"]["initial_optim_state_dict"] = optim_state_dict

    algo = SPRCategoricalDQN(**config["algo"])  # Run with defaults.
    agent = SPRAgent(ModelCls=SPRCatDqnModel, **config["agent"])
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        eval_CollectorCls=OneToOneSerialEvalCollector,
        **config["sampler"],
    )
    runner_type = OfflineEval if offline else OnlineEval
    runner = runner_type(algo=algo, agent=agent, sampler=sampler, save_fn=save_fn, start_itr=start_itr, **config["runner"])

    with logger_context(**config["context"]):
        runner.train()

    return algo, agent, sampler, runner


if __name__ == "__main__":
    main()
