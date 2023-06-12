
```
# training AgenX, RNNX user simulators
nohup python -u run_rl_training.py > rl_repro.log 2>&1 &

# training GPT, GPT_IL user simulators
nohup python -u run_rl_training_with_gpt.py > rl_gpt.log 2>&1 &