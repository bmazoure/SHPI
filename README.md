To run the code:

1. Install requirements for Python environemnt: `pip install -r requirements.txt`
2. From the top directory, run `python run_experiments.py --env_name toy --n_features 2 --n_actions 10 --n_timesteps_offline 150 --n_timesteps_online 150 --n_timesteps_test 150 --n_traj_offline 10 --n_traj_online 1000 --delta 20 --rho 30 --k 5 --policy_improvement_steps=5 --Y_bias="none" --eps_behavior 0.3 --dice_RL 0 --seed 0`
3. The folder "runs" contains the online evaluation results.

Note: all files under "scripts" ending with "_dice" are taken directly from https://github.com/google-research/dice_rl and adapted to match the function signatures we have in our scripts. They are subject to the original license from Google LLC.