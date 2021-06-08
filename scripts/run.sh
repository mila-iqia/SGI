#!/bin/bash
source $HOME/.bashrc
module load python/3.8.2
[[ -d '/project/' ]] && module load StdEnv/2020 gcc/9.3.0 cuda/11.0 opencv/4.5.1
if [[ $(hostname) != *"blg"* && $(hostname) != *"cdr"* && $(hostname) != *"gra"* ]]; then
  module load python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
fi

echo Running on $HOSTNAME
nvidia-smi
date

PROJECT_DIR=$HOME/transfer-ssl-rl
if [[ $(hostname) == *"cedar"* && ${USER} != 'schwarzm' ]]; then
  PROJECT_DIR=${SCRATCH}/transfer-ssl-rl
elif [[ ${USER} == 'schwarzm' && $(hostname) == *"cedar"* ]]; then
  PROJECT_DIR=${SCRATCH}/Github/transfer-ssl-rl/
elif [[ ${USER} == 'schwarzm' ]]; then
  PROJECT_DIR=${HOME}/Github/transfer-ssl-rl/
fi

cd ${PROJECT_DIR}
mkdir -p models
mkdir -p ${SLURM_TMPDIR}/atari
export TMP_DATA_DIR=${SLURM_TMPDIR}/atari

#Set up virtualenv
echo 'Installing dependencies'
python -m venv ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate
if [[ $(hostname) == *"blg"* ||  $(hostname) == *"cedar"* ||  $(hostname) == *"gra"*  ]]; then
  pip install --no-index -U pip
  pip install --no-index --find-links=/scratch/schwarzm/wheels_38 -r scripts/requirements_cc.txt
else
  pip install -U pip
  pip install -r requirements.txt
fi
#conda activate pytorch
#export PATH=~/anaconda3/envs/pytorch/bin:~/miniconda3/envs/pytorch/bin:$PATH

# Set default data directories for reading and writing
if [[ -d '/network/' ]]; then # mc
  export DATA_DIR=/network/tmp1/rajkuman/atari/
  export USER_DATA_DIR=/network/tmp1/${USER}/atari/
   python -m atari_py.import_roms Roms/ # need to manually load on MC for some reason
elif [[ $(hostname) == *"cedar"* ]]; then # cc
  export DATA_DIR=/project/rrg-bengioy-ad/rajkuman/atari
  export USER_DATA_DIR=/project/rrg-bengioy-ad/${USER}/atari
  echo 'Setting W&B to offline'
  wandb off
  export WANDB_MODE=dryrun
elif [[ $(hostname) == *"blg"* ]]; then # cc
  export DATA_DIR=/project/rrg-bengioy-ad/rajkuman/atari
  export USER_DATA_DIR=/project/rrg-bengioy-ad/${USER}/atari

  echo 'Setting W&B to offline'
  wandb off
  export WANDB_MODE=dryrun
elif [[ $(hostname) == *"gra"* ]]; then # cc
  export DATA_DIR=/project/rrg-bengioy-ad/rajkuman/atari
  export USER_DATA_DIR=/project/def-bengioy/${USER}/atari

  echo 'Setting W&B to offline'
  wandb off
  export WANDB_MODE=dryrun
fi

mkdir -p ${USER_DATA_DIR}

echo 'Starting experiment'
python -u -m scripts.run wandb.dir=$PROJECT_DIR "$@"
