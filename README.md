# Neuroevolution of Self-interpretable agents on procgen game
This repository contains the Deep Learning and Applied AI project, aiming to replicate the principles described in the paper "Neuroevolution of Self-Interpretable Agents" on the procgen game StarPilot. The goal is to demonstrate that an agent, despite having a limited number of trainable parameters, can generalize effectively on unseen episodes, achieving satisfying results while navigating procgen game environments. This achievement is made possible by leveraging attention mechanisms and small, simple networks containing only a few thousand trainable parameters. Furthermore, I will present my own proposal (the "Double" Self Attention) that leverages attention principles in a slightly different manner, leading the agent to exhibit distinct behaviors.

# Attention bottleneck mechanism
The core concept combines attention with image processing by dividing them into N patches. A self-attention mechanism assesses patch importance and information content. To enhance focus, a trainable module selects k patches from N, creating an attention bottleneck. This prioritizes crucial patches while disregarding less impactful ones. These crucial patches form the patch importance vector.

# Patch Selection, Feature retrieval and Controller
By sorting the patch importance vector, we extract the indices of the top K patches. These indices are associated with their respective features and fed into the controller module for action determination. The controller can receive two types of features: the coordinates of the K best patch centers (as in the original paper) or a novel approach involving both patch coordinates and average RGB color values. The controller architecture resembles the one described in the paper, utilizing an LSTM for sequential decision-making, with an additional linear layer and ReLU activation function preceding the output.

# My "Double" Self Attention
The concept is to use the initial self-attention to select the top K patches crucial for each timestep. Preserving the centers' coordinates' mapping to the original observation, these patches are used to compute a new 'Double' self-attention. The 'Double' self-attention calculates the dot product solely among these K best patches, already chosen by the initial self-attention, rather than the entire observation. This allows us to focus specifically on patches that play the most crucial role and are thus located closest to the agent's current patch. The "Double" self-attention compels the agent to pay greater attention to the most important patches among those deemed relevant by the initial self-attention.

# Training with CMA-ES
The network was trained with Covariance-Matrix Adaptation Evolution Strategy (CMA-ES), a genetic algorithm, due to the non-differentiable nature of obtaining the top K sorted patches. Genetic algorithms excel in reinforcement learning, handling scenarios with rewards available either at each timestep or cumulatively at the task's end.  
I trained the different models using a 2018 MacBook Pro without a GPU compatible with PyTorch. To speed up the training process, given the lack of a GPU, I utilized the "multiprocessing" package to parallelize the CMA-ES training across 8 workers. This approach allowed each model to complete approximately 2500 iterations of CMA-ES in about a week and a half.  

There are 2 options:
1. If you want to train a new network from scratch, simply copy one of the following commands (without copying the apexes) into the terminal, depending on the type of training you want to perform.
First, position yourself at the 'solutions' folder level without entering it.
The command 'python train.py' allows training an agent with default hyperparameters (described and explained in the 'def make_argparse()' function within the 'train.py' file). If you want to train the agent differently, with different features, or modify certain CMA-ES hyperparameters, you can append one or more of the following options to the 'python train.py' command:  
--seed 'an integer representing the CMA-ES seed'  
--max_iterations_cma 'an integer representing the maximum number of iterations performed in the CMA-ES.'  
--number_rollouts 'an integer representing the number of rollouts'    
--popsize_cma 'an integer representing the population size of the CMA-ES'    
--initial_std_cma 'an float representing the initial standard deviation of the CMA-ES'  
--number_workers 'an integer representing the number of workers used to train in parallel the agent'  
--evaluate_every 'an integer representing after how many iterations the model is evaluated'  
--remember_training_cma 'Enter the string 'no' if you want to start training from scratch, or 'yes' if you want to resume training from a previously calculated solution.'  
--feature_retrieval_strategy 'choose the string "positions" to use the coordinates (centers) of the top patches as features, or choose "colors_and_positions" to include the average colors of the top patches as well.'  
--active_double_self_attention_on_best_patches 'Enter the string "yes" if you want to utilize the Double Self Attention during training, or "no" if you prefer not to use it.'  
 Example of a command (do not copy the quotation marks):    
'python train.py --evaluate_every 500 --feature_retrieval_strategy positions --active_double_self_attention_on_best_patches no'

2. To resume training from a previously calculated solution, you can execute one of the following commands (do not copy the quotation marks) and if you want, you can add the other options to personalize these basic commands:  
   'python train.py --remember_training_cma yes --feature_retrieval_strategy positions --active_double_self_attention_on_best_patches no'  
   'python train.py --remember_training_cma yes --feature_retrieval_strategy colors_and_positions --active_double_self_attention_on_best_patches no'  
   'python train.py --remember_training_cma yes --feature_retrieval_strategy colors_and_positions --active_double_self_attention_on_best_patches yes'  

# Testing on pre-trained models.

# Procgen StarPilot gameplay demo
