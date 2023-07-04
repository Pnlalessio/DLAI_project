import argparse
from pathlib import Path
import cma
import torch
import numpy as np
from multiprocessing import Pool
from functools import partial
from torch import nn
from self_attention_agent import SelfAttentionAgent
from train_evaluation import evaluate
import gym

def fitness(solution, agent, episode, num_rollouts):
    '''This function takes as input a solution (a numpy array with the same number of elements 
    as the trainable parameters of the entire network), an instance of the "SelfAttentionAgent", 
    the current game episode, and the number of rollouts needed to evaluate each generated solution. 
    It constructs the game environment, initializes the agent with the newly generated parameters from 
    the CMA-ES using the cma.CMAEvolutionStrategy(...).ask() method, performs num_rollouts rollouts, 
    calculates their average reward, and returns the negated value. The returned value is then used 
    as input for the cma.CMAEvolutionStrategy(...).tell(...) method. 
    The tell(...) method calls this fitness(...) function for each solution generated from the ask() 
    method. This way, each solution is evaluated based on this fitness function, determines the best solutions, 
    and takes their average. This average provides a new best solution, which becomes the starting point for 
    the next iteration of the CMA-ES.'''
    env = make_env()
    agent.inizialize_learnable_parameters(solution)
    reward = agent.rollout(env, num_rollouts, episode)
    return -reward

def train(rememberTraining, args):
    episode = 0
    # Initialize the agent
    if args.active_double_self_attention_on_best_patches == "yes": # Initialize the agent with Double Self Attention
    	act_dbl_self_att = True
    else: # Initialize the agent without Double Self Attention
        act_dbl_self_att = False
    agent = SelfAttentionAgent(isTest=False, feature_retrieval_strategy = args.feature_retrieval_strategy, active_double_self_attention_on_best_patches = act_dbl_self_att)
    if episode == 0 and rememberTraining == True: # Resume training from the last interruption
        parameters = np.load(args.resume_old_solution_from)
        print("Resume training from the last interruption...")
    else:
        print("Train for the first time...")
        parameters = np.random.random(size = agent.get_learnable_parameters_number())
    # Initialize the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(parameters, args.initial_std_cma,
                                         {'popsize': args.popsize_cma,
                                         'seed': args.seed,
                                         })

    best_ever = parameters
    num_workers = args.number_workers
    with Pool(num_workers) as p:
        while not es.stop() and episode < args.max_iterations_cma:
            solutions = es.ask() # es.ask() generates 'popsize' number of potential solutions for each iteration of the genetic algorithm.
            es.tell(solutions, p.map(partial(fitness, agent = SelfAttentionAgent(isTest = False, feature_retrieval_strategy = args.feature_retrieval_strategy, active_double_self_attention_on_best_patches = act_dbl_self_att), episode = episode, num_rollouts = args.number_rollouts), solutions))
            '''es.tell() Parallelly evaluate multiple solutions based on the fitness function. 
            Utilize only the top 25% (those above average) of solutions generated for each generation. 
            Compute the Covariance matrix for the next generation. In the next iteration, explore a new 
            set of candidate solutions using the updated mean and the updated covariance matrix.'''
            es.disp()
            episode += 1
            best_ever = es.result.xbest
            np.save('solutions/current_best_solution.npy', best_ever)
            if episode % args.evaluate_every == 0:
                '''Every args.evaluate_every iterations, a test is conducted on 100 episodes 
            	using the best solution found by the algorithm up to that point. 
            	It calculates the average and maximum reward values across the 100 episodes.'''
                mean_reward, max_reward = evaluate(best_solution = np.load('solutions/current_best_solution.npy'), args = args, act_dbl_self_att = act_dbl_self_att, env = make_env())
                np.save('solutions/best_solution_{}_iterations_mean_reward={}_max_reward={}.npy'.format(episode, mean_reward, max_reward), best_ever)

    # Get the best solution
    best_ever = es.result.xbest

def make_env():
    '''The make_env() function is a function used to generate a new game environment.
    This section sets the hyperparameters used to construct the environment.'''
    env_name = 'procgen:procgen-starpilot-v0'
    distribution_mode="easy"
    start_level=0
    num_levels=0
    use_backgrounds=False
    restrict_themes=True
    use_monochrome_assets=False
    env = gym.make(env_name, distribution_mode = distribution_mode, start_level = start_level, num_levels = num_levels, use_backgrounds = use_backgrounds, restrict_themes = restrict_themes, use_monochrome_assets = use_monochrome_assets)
    return env

def make_argparse():
    '''The make_argparse() function is used to choose the hyperparameters used to initialize the agent's training.'''
    parser = argparse.ArgumentParser(description = "DLAI Project")
    parser.add_argument("--seed", type = int, default = 1) # This hyperparameter sets the seed used within the genetic algorithm (cma-es) during training.
    parser.add_argument("--max_iterations_cma", type = int, default = 2000) # This hyperparameter determines the maximum number of iterations performed in the CMA-ES.
    parser.add_argument("--number_rollouts", type = int, default = 5) # This hyperparameter determines the number of rollouts (consecutive execution of multiple game episodes) the agent performs before calculating the average rewards (each rollout generates a reward) to be passed to the tell() evaluation method of CMA-ES. This generates a fitness value for a particular generated solution.
    parser.add_argument("--popsize_cma", type = int, default = 256) # This hyperparameter sets the number of solutions generated by CMA-ES to be evaluated in each iteration.
    parser.add_argument("--initial_std_cma", type = float, default = 1.0) # This hyperparameter sets the standard deviation value used to initialize the CMA-ES.
    parser.add_argument("--number_workers", type = int, default = 8) # This hyperparameter determines the number of workers used to parallelize the training of CMA-ES through the "multiprocessing" package. Due to the absence of a dedicated GPU, this package was utilized to expedite the training process.
    parser.add_argument("--evaluate_every", type = int, default = 500) # This hyperparameter determines after how many iterations of CMA-ES the best solution found up to that point in training is tested.
    parser.add_argument("--remember_training_cma", type = str, default = "no") # The possible values for this hyperparameter are "yes" or "no" based on whether one wants to continue training from the previous stopping point or start fresh from scratch.
    parser.add_argument("--feature_retrieval_strategy", type = str, default = "colors_and_positions") # This hyperparameter determines the type of features used as input for the controller: choose "positions" to use the coordinates (centers) of the top patches as features, or choose "colors_and_positions" to include the average colors of the top patches as well.
    parser.add_argument("--active_double_self_attention_on_best_patches", type = str, default = "yes") # This hyperparameter can be set to "yes" if you want to utilize the Double Self Attention during training, or "no" if you prefer not to use it.
    parser.add_argument("--resume_old_solution_from", type = Path, default = None) # This hyperparameter is used to indicate the path where we want to take the pre-trained solution.
    return parser.parse_args()

if __name__== "__main__": 
    args = make_argparse()
    if args.remember_training_cma == "yes":
        assert args.resume_old_solution_from != None, "Specify the path of the best solution from which you want to resume training. Use the command --resume_old_solution_from"
        train(rememberTraining = True, args = args)
    else:
        train(rememberTraining = False, args = args)
