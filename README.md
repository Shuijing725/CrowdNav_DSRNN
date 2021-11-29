# DSRNN_CrowdNav
This repository contains the codes for our paper titled "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning" in ICRA 2021. 
For more details, please refer to the [project website](https://sites.google.com/illinois.edu/crowdnav-dsrnn/home) and 
[arXiv preprint](https://arxiv.org/abs/2011.04820).
For experiment demonstrations, please refer to the [youtube video](https://youtu.be/bYO-1IAjzgY).


## Abstract
Safe and efficient navigation through human crowds is an essential capability for mobile robots. 
Previous work on robot crowd navigation assumes that the dynamics of all agents are known and well-defined. In addition, the performance of previous methods deteriorates in partially observable environments and environments with dense crowds. 
To tackle these problems, we propose decentralized structural-Recurrent Neural Network (DS-RNN), a novel network that reasons about spatial and temporal relationships for robot decision making in crowd navigation. 
We train our network with model-free deep reinforcement learning without any expert supervision. 
We demonstrate that our model outperforms previous methods in challenging crowd navigation scenarios. 
We successfully transfer the policy learned in the simulator to a real-world TurtleBot 2i.

<img src="/figures/open.jpg" width="450" />


## Setup
1. Install Python3.6 (The code may work with other versions of Python, but 3.6 is highly recommended).
2. Install the required python package using pip or conda. For pip, use the following command:  
```
pip install -r requirements.txt
```
For conda, please install each package in `requirements.txt` into your conda environment manually and 
follow the instructions on the anaconda website.  

3. Install [OpenAI Baselines](https://github.com/openai/baselines#installation).   
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

4. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.  


## Getting started
This repository is organized in three parts: 
- `crowd_sim/` folder contains the simulation environment. Details of the simulation framework can be found
[here](crowd_sim/README.md).
- `crowd_nav/` folder contains configurations and non-neural network policies
- `pytorchBaselines/` contains the code for the DSRNN network and ppo algorithm.  
 
Below are the instructions for training and testing policies.

### Change configurations
1. Environment configurations and training hyperparameters: modify `crowd_nav/configs/config.py`
- For FoV environment (left in the figure below): change the value of `robot.FOV` in `config.py`
- For Group environment (right in the figure below): set `sim.group_human` to `True` in `config.py`

<img src="/figures/FOV_env.png" height="270" /> <img src="/figures/group_env.png" height="270" />



### Run the code
1. Train a policy. 
```
python train.py 
```

2. Test policies.  
Please modify the test arguments in the begining of `test.py`.     
We provide two trained example weights for each type of robot kinematics:  
    - Holonomic: `data/example_model/checkpoints/27776.pt` 
    - Unicycle: `data/example_model_unicycle/checkpoints/55554.pt`  
```
python test.py 
```

3. Plot training curve.
```
python plot.py
```
(We only tested our code in Ubuntu 16.04 and 18.04 with Python 3.6.)

## Learning curves
Learning curves of DS-RNN in 360 degrees FoV environment with 5 humans.

<img src="/figures/reward.png" width="450" />
<img src="/figures/loss.png" width="450" />

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517-3524}
}
```

## Credits
Other contributors:  
[Peixin Chang](https://github.com/PeixinC)  
[Neeloy Chakraborty](https://github.com/TheNeeloy)  

Part of the code is based on the following repositories:  

[1] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[2] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

[3] A. Vemula, K. Muelling, and J. Oh, “Social attention: Modeling attention in human crowds,” in IEEE international Conference on Robotics and Automation (ICRA), 2018, pp. 1–7.
(Github: https://github.com/jeanoh/big)

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.