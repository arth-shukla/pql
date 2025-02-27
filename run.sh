#!/usr/bin/bash

ENV_ID="Anymal"
PROJECT_NAME="IsaacGymEnvs-SAC-${ENV_ID}-Testing"

python scripts/train_pql.py task=$ENV_ID algo.num_gpus=1 \
    algo.p_learner_gpu=0 algo.v_learner_gpu=0 \
    algo.critic_actor_ratio=2 \
    algo.critic_sample_ratio=8 \
    algo.distl=False \
    algo.cri_class=DoubleQ \
    logging.wandb.project="$PROJECT_NAME" \
    logging.wandb.group="pql" \
    logging.wandb.name="$ENV_ID/pql_testing"

python scripts/train_pql.py task=$ENV_ID algo.num_gpus=1 \
    algo.p_learner_gpu=0 algo.v_learner_gpu=0 \
    algo.critic_actor_ratio=2 \
    algo.critic_sample_ratio=8 \
    algo.distl=False \
    algo.cri_class=DoubleQ \
    logging.wandb.project="$PROJECT_NAME" \
    logging.wandb.group="pql" \
    logging.wandb.name="$ENV_ID/pql_testing_no-eval" \
    algo.eval_freq=null

# python scripts/train_baselines.py algo=sac_algo task=$ENV_ID \
#     logging.wandb.project="$PROJECT_NAME" \
#     logging.wandb.group="pql" \
#     logging.wandb.name="$ENV_ID/pql_sac_1_step_testing" \
#     algo.nstep=1
