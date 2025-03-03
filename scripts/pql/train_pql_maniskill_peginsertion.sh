#!/usr/bin/bash

SEED=2024
# Task args
ENV_ID=PegInsertionSide-v1
WORKSPACE=outputs
EXP="v0-${SEED}"
GROUP="${ENV_ID}-pql-2-8"
ENTITY=sumanoid
PROJECT_NAME=ManiSkill-PQL-PegInsertion
EXP_NAME="$ENV_ID/$GROUP/$EXP"

NUM_ENVS=4096
BATCH_SIZE=8192
MAX_EPISODE_STEPS=100

MAX_MINS=60
MAX_TIME=$((MAX_MINS * 60))

#############################################

python scripts/train_pql.py \
    +task.name=$ENV_ID \
    +task.num_envs=$NUM_ENVS \
    +task.max_episode_steps=$MAX_EPISODE_STEPS \
    +task.control_mode=pd_joint_delta_pos \
    headless=True \
    \
    algo.num_gpus=1 \
    algo.p_learner_gpu=0 algo.v_learner_gpu=0 \
    algo.critic_actor_ratio=2 \
    algo.critic_sample_ratio=8 \
    algo.distl=False \
    algo.cri_class=DoubleQ \
    \
    logging.wandb.project="$PROJECT_NAME" \
    logging.wandb.entity="$ENTITY" \
    logging.wandb.group="$GROUP" \
    logging.wandb.name="$EXP_NAME" \
    logging.workspace="$WORKSPACE" \
    logging.clear_out=True \
    \
    max_time=$MAX_TIME \
    \
    algo.gamma=0.99 \
    algo.batch_size=$BATCH_SIZE
