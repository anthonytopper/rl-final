# RL_project

FlappyBird_DQN_prioritizedReplay_1 contains code(.py), trained_model(.pth), demonstration video(.avi) and training reward log(.txt).

## Training

```
python main.py --train_dqn --history_size 6000 --mode smooth1_prioritized --delay 10000 --name smooth1_prioritized
```
Running the scipt will train DQN with prioritized replay with smooth loss penalty. --history_size specifies replay buffer size, and --delay indicates when to apply the smooth penalty.

## Testing

```
python main.py --test_dqn --name DQN_prioritized
```

Running the --test_dqn with a model name will output test reward with a video saved to the executed folder.

For library installation, plese see .ipynb file.
