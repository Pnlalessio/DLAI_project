import argparse
from pathlib import Path
import numpy as np
import torch
import gym
from self_attention_agent import SelfAttentionAgent



def test(args):
	test_episodes = args.num_episodes
	# Loads the best parameters obtained from the cma-es evolution strategy used during the training phase
	best_solution = np.load(args.test_from_solution_at)
	# Here you can set the options of the environment
	render_mode = 'human'
	env_name = 'procgen:procgen-starpilot-v0'
	distribution_mode="easy"
	num_levels=0
	use_backgrounds=False
	restrict_themes=True
	use_monochrome_assets=False
	# Instantiates the agent
	if args.active_double_self_attention_on_best_patches == "yes":
		act_dbl_self_att = True
	else:
		act_dbl_self_att = False
	agent = SelfAttentionAgent(isTest=True, render_mode = render_mode, feature_retrieval_strategy = args.feature_retrieval_strategy, active_double_self_attention_on_best_patches = act_dbl_self_att)
	# Inizializes the environment
	if render_mode == 'human':
		env = gym.make(env_name, render_mode = render_mode, distribution_mode = distribution_mode, num_levels = num_levels, use_backgrounds = use_backgrounds, restrict_themes = restrict_themes, use_monochrome_assets = use_monochrome_assets)
	else:
		env = gym.make(env_name = env_name, distribution_mode = distribution_mode, num_levels = num_levels, use_backgrounds = use_backgrounds, restrict_themes = restrict_themes, use_monochrome_assets = use_monochrome_assets)

	# Initializes the agent with the best parameters
	agent.inizialize_learnable_parameters(best_solution)
	total_test_rewards = 0
	max_reward = 0
	for episode in range(test_episodes):
		ep_total_reward = agent.rollout(env, 1, episode)
		total_test_rewards += ep_total_reward
		if ep_total_reward > max_reward:
			max_reward = ep_total_reward
		print(f"EPISODE {episode} EP_TOTAL_REWARD {ep_total_reward}")
	print("Mean reward: ", total_test_rewards / test_episodes)
	print("Max reward: ", max_reward)

def make_argparse():
	parser = argparse.ArgumentParser(description = "DLAI Project")
	parser.add_argument("--test_from_solution_at", type=Path, default='solutions/only_positions/current_best_solution.npy')
	parser.add_argument("--num_episodes", type=int, default=100)
	parser.add_argument("--feature_retrieval_strategy", type = str, default = "positions")
	parser.add_argument("--active_double_self_attention_on_best_patches", type = str, default = "no")
	return parser.parse_args()

if __name__== "__main__":
	args = make_argparse()
	test(args)