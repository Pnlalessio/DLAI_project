import numpy as np
import torch
import gym
from self_attention_agent import SelfAttentionAgent

def evaluate(best_solution, args, act_dbl_self_att, env):
	'''This function tests the current best solution found by the CMA-ES algorithm 
	on 100 episodes, calculating the average and maximum reward values across the episodes.'''
	test_episodes = 100
	agent = SelfAttentionAgent(isTest = True, feature_retrieval_strategy = args.feature_retrieval_strategy, active_double_self_attention_on_best_patches = act_dbl_self_att)
	agent.inizialize_learnable_parameters(best_solution)
	total_test_rewards = 0
	max_reward = 0
	for episode in range(test_episodes):
		ep_total_reward = agent.rollout(env, 1, episode)
		if ep_total_reward > max_reward:
			max_reward = ep_total_reward
		total_test_rewards += ep_total_reward
	mean_reward = total_test_rewards / test_episodes
	print("The mean reward is {} and the max reward is {} (out of 100 episodes)".format(mean_reward, max_reward))
	return mean_reward, max_reward
