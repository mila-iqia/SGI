#!/bin/bash
declare -A map=( ["pong"]="Pong" ["breakout"]="Breakout" ["up_n_down"]="UpNDown" ["kangaroo"]="Kangaroo" ["bank_heist"]="BankHeist" ["assault"]="Assault" ["boxing"]="Boxing" ["battle_zone"]="BattleZone" ["frostbite"]="Frostbite" ["crazy_climber"]="CrazyClimber" ["chopper_command"]="ChopperCommand" ["demon_attack"]="DemonAttack" ["alien"]="Alien" ["kung_fu_master"]="KungFuMaster" ["qbert"]="Qbert" ["ms_pacman"]="MsPacman" ["hero"]="Hero" ["seaquest"]="Seaquest" ["jamesbond"]="Jamesbond" ["amidar"]="Amidar" ["asterix"]="Asterix" ["private_eye"]="PrivateEye" ["gopher"]="Gopher" ["krull"]="Krull" ["freeway"]="Freeway" ["road_runner"]="RoadRunner" )
export game=$1
shift
export seed=$1

python -m scripts.run public=True env.game=$game seed=$seed num_logs=10  \
    model_load=sgiw_${game}_resnet_${seed} \
    model_folder=./ \
    algo.encoder_lr=0.000001 \
    algo.q_l1_lr=0.00003\
    algo.clip_grad_norm=-1 \
    algo.clip_model_grad_norm=-1