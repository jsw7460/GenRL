# GenRL

JAX-based Offline RL / Imitation Learning Framework

## Supported Algorithm / Policy Architecture
1. Diffusion
2. Decision Transformer
3. Multi-layer Perceptron (Basic MLP)
(Verified in basic environments (HalfCheetah, Walker2d))

## Dataset 
We use Minari, an offline-rl dataset framework operated by Farama-Foundataion. However, we have made modifications such as sub-trajectory sampling (for decision transformer) and loading the dataset into memory cache. All code related to the dataset can be found in genrl/rl/buffers/...

## Training
Policy Architectures can be found in genrl/policies/low/policy/nn/...\
The loss functions (e.g., behavior cloning, diffusion denoise) are defined in genrl/policies/low/policy/agent.py

Policies are passed to the BC class in algos/bc/bc.py to perform behavior cloning. 
All configurations related to training can be found in config/train/. 
Please modify them according to your own path. 
Especially, check the save_prefix, save_suffix, and save_interval in config/train/base.yaml. 
Models being trained are saved every save_interval in {save_prefix} / {save_suffix}.

Refer to the scripts/train_bc.py or scripts/train_mujoco.py for the training execution code.

## Evaluation
Use the following command:

```python3 scripts/eval_bc.py pretrained_suffix=YOUR_SAVE_SUFFIX date=yyyy-mm-dd step=STEP_FOR_LOAD```

For pretrained_suffix, enter the same string as the save_suffix used for training. date refers to the date you ran the training code. step refers to the trained step of the mode you want to load.
