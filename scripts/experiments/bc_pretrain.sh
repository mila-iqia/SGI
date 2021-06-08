#!/bin/bash
declare -A map=( ["pong"]="Pong" ["breakout"]="Breakout" ["up_n_down"]="UpNDown" ["kangaroo"]="Kangaroo" ["bank_heist"]="BankHeist" ["assault"]="Assault" ["boxing"]="Boxing" ["battle_zone"]="BattleZone" ["frostbite"]="Frostbite" ["crazy_climber"]="CrazyClimber" ["chopper_command"]="ChopperCommand" ["demon_attack"]="DemonAttack" ["alien"]="Alien" ["kung_fu_master"]="KungFuMaster" ["qbert"]="Qbert" ["ms_pacman"]="MsPacman" ["hero"]="Hero" ["seaquest"]="Seaquest" ["jamesbond"]="Jamesbond" ["amidar"]="Amidar" ["asterix"]="Asterix" ["private_eye"]="PrivateEye" ["gopher"]="Gopher" ["krull"]="Krull" ["freeway"]="Freeway" ["road_runner"]="RoadRunner" )
export game=$1
shift
export seed=$1

python -m scripts.run public=True model_folder=./ offline.runner.save_every=2500 \
    env.game=pong seed=1 offline_model_save=bc_${game}_resnet_${seed} \
    offline.runner.epochs=20 offline.runner.dataloader.games=[${map[${game}]}] \
    offline.runner.no_eval=1 \
    +offline.algo.bc_weight=1 \
    +offline.algo.goal_weight=0 \
    +offline.algo.inverse_model_weight=0 \
    +offline.algo.spr_weight=0 \
    +offline.algo.target_update_tau=0.01 \
    +offline.agent.model_kwargs.momentum_tau=0.01 \
    +offline.algo.jumps=0 \
    do_online=False \
    algo.batch_size=256 \
    +offline.agent.model_kwargs.noisy_nets_std=0 \
    offline.runner.dataloader.dataset_on_disk=True \
    offline.runner.dataloader.samples=1000000 \
    offline.runner.dataloader.checkpoints='[1,25,50]' \
    offline.runner.dataloader.num_workers=2 \
    offline.runner.dataloader.data_path=./data/ \
    offline.runner.dataloader.tmp_data_path=./