```
mamba create -n pql python=3.8
mamba activate pql

wget https://developer.nvidia.com/isaac-gym-preview-4 -O isaac.tar.gz
tar -xzf isaac.tar.gz -C .
pip install -e isaacgym/python

git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
pip install -e IsaacGymEnvs
```


