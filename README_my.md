# PQL

## Installation

### IsaacGymEnvs

```bash
mamba create -n pql python=3.8
mamba activate pql

wget https://developer.nvidia.com/isaac-gym-preview-4 -O isaac.tar.gz
tar -xzf isaac.tar.gz -C .
pip install -e isaacgym/python

git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
pip install -e IsaacGymEnvs

pip install -e .
pip install "numpy==1.21.0"
```

### ManiSkill

```bash
mamba create -n pql_maniskill python=3.9
mamba activate pql_maniskill

git clone -b humanoid https://github.com/haosulab/ManiSkill.git
pip install -e ManiSkill

pip install -e .
pip install "numpy==1.23.0"
```
