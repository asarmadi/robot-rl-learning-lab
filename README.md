# Robot RL Learning Lab
This repository documents my hands-on study of deep reinforcement learning algorithms, with a focus on robotic control and manipulation. I reproduced core policy-gradient and actor-critic methods, ran controlled experiments, and extended the examples to MuJoCo/robotics environments.

## Requirements
I have tested this code on python3.13 and listed all the requiremetns under requirements.txt to be installed on your virtual environment.

To create a virtual environment, you can run this command:
```bash
python3.13 -m venv env
```
It creates a virtual environment called env. Then you can activate that environment to install the requirements:
```bash
source ./env/Scripts/activate
```
for Linux:
```bash
source ./env/bin/activate
```
Then install the requirements
```bash
pip install -r requirements.txt
```

## Running the experiments
For running the experiments, run the following command in your main directory (e.g., ./robot_rl_learning_lab/):
```bash
python -m experiments.random_walk_1d
```