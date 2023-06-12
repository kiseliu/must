
Codes for the paper "[One cannot stand for everyone! Leveraging Multiple User Simulators to train Task-oriented Dialogue Systems](https://openreview.net/forum?id=Y2E5-_HL0DV)" (ACL 2023).


# Dataset Preparation
### The training data of user simulators:

* **GPT** user simulator:
`data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_mwz.json`.
* **GPT$_{\mathrm{IL}}$** user simulator: `data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_gpt_il.json`.


# Training & Inference
### Train the user simulators
```
# train GPT, GPT_IL user simulators
python -u simulator_gpt_act/model.py -mode train
```

### Train the system agents
```
# training the systems with AgenX, RNNX user simulators
nohup python -u run_rl_training.py > rl_repro.log 2>&1 &

# training the systems with GPT, GPT_IL user simulators
nohup python -u run_rl_training_with_gpt.py > rl_gpt.log 2>&1 &

# training the systems with MUST
nohup python -u run_must_training.py > rl_must.log 2>&1 &
```

### Evaluate the system agents
```
bash evaluation_matrix.sh
```

# Citation

# Acknowledgement
This code is modified upon [the released code](https://github.com/wyshi/user-simulator) of previous EMNLP-IJCNLP 2019 paper "[How to Build User Simulators to Train RL-based Dialog Systems](https://aclanthology.org/D19-1206.pdf)".