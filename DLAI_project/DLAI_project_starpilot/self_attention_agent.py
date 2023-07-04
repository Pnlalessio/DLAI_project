import numpy as np
import torch
import torch.nn as nn
import cv2
import random
import math
from get_patches import get_patches, show_blue_or_white_patches
from feature_retrieval_functions import f, get_best_patches_color_means, map_best_patches_on_obs, map_new_indices_to_old_indices
from agent_models import SelfAttention, LSTMController


class SelfAttentionAgent():
    def __init__(self, L = 64, M = 7, S = 4, isTest = False, render_mode = 'rgb_array', feature_retrieval_strategy = 'colors_and_positions', active_double_self_attention_on_best_patches = False): #buono era 20 k, M = 5, S = 5, d = 4
        self.L = L #L is the observation size.
        self.M = M #M is the patch size.
        self.S = S # S is the stride.
        self.isTest = isTest # if True we are testing, if False we are training.
        self.render_mode = render_mode # This says what is the env render_mode value.
        self.N = (((L - M) // S) + 1) ** 2 # N is the number of patches.
        self.d_in = (M**2)*3 # This hyperparameter represents the product of the number of pixels in a patch and the number of channels.
        self.d = 4 # d is the dimension of the transformed space in which the Key and Query matrices.
        self.K = 10 # K represents the number of patches we want to sample after performing self-attention.
        self.n_kq = 2 # (one vector for keys and one vector for queries in one Self Attention layer).
        self.num_attentions = 1
        if feature_retrieval_strategy == 'colors_and_positions':
            '''When training or testing the agent using color channel averages in addition 
            to patch center coordinates as features, the self.color_channels_to_consider 
            variable is set to 3 to account for the three RGB channels.'''
            self.color_channels_to_consider = 3
        else:
            self.color_channels_to_consider = 0
        self.attention = SelfAttention(dimension_input=self.d_in, d=self.d) # Initialize the default self-attention as presented in the paper "Neuroevolution of Self-Interpretable Agents.
        self.active_double_self_attention_on_best_patches = active_double_self_attention_on_best_patches
        if self.active_double_self_attention_on_best_patches == True:
            self.K = self.K * 2
            self.double_attention_on_patches = SelfAttention(dimension_input=self.d_in, d=self.d) # Initialize the 'Double' self-attention if the hyperparameter 'active_double_self_attention_on_best_patches' is True.
            '''In this case, there will be two self-attentions to consider in both training and testing. The first is 
            the default one shown in the original paper, while the second is my own invention as a potential improvement. 
            It involves applying a second attention mechanism to the top K patches obtained from the application of the first self-attention.'''
            self.num_attentions = 2
            self.n_kq = 2 * 2 # We have two self attention layer so we need double the number of queries and keys.

        '''Below, the controller is initialized to determine the next action to take with a certain probability, 
        based on the agent's past timesteps and the features detected in the current timestep. Similar to the original 
        paper, the controller used is an LSTM network, which is a recurrent network that remembers the previous states 
        the agent has encountered.'''
        self.LSTMController = LSTMController(num_classes=5, input_size=2 * math.floor(self.K / self.num_attentions) + math.floor(self.K / self.num_attentions) * self.color_channels_to_consider, hidden_size=16, num_layers=1, seq_length=1, L=self.L)
        self.learnable_parameters_number = self.n_kq * self.d_in * self.d + np.sum(p.numel() for p in self.LSTMController.parameters()) # Here, we calculate the number of trainable parameters in the entire network used by the agent.
        self.feature_retrieval_strategy = feature_retrieval_strategy
        '''The "remember_actions" dictionary is used when we want the agent to explore more (an exploration I personally 
        added to the original paper) to avoid falling into local minima during training with the CMA-ES genetic algorithm. 
        Each key represents a possible action the agent can take, and the associated values indicate the number of times that 
        action was performed during an episode. The values are reset at the end of each episode. This dictionary serves as a 
        record to keep track of which actions were executed and how many times they were performed up to that specific timestep 
        in the episode.The penalty_actions tensor represents actions available to the agent during exploration. 
        Its indices correspond to actions, and the values represent penalties, reducing the probability of repeating an 
        action in the next step if it occurs too frequently in this episode. The penalty increases with the number of 
        times an action is executed within the episode.'''
        self.remeber_actions = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0} 
        self.penalty_actions = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    def take_an_action(self, observation, episode):
        observation = observation/255 # Each observation (an RGB image) is normalized by dividing each pixel value by 255 (the maximum value in RGB).
        observation = cv2.resize(observation, dsize=(self.L, self.L), interpolation=cv2.INTER_CUBIC)
        '''With the 'get_patches(...)' function the current observation is divided into N patches according to the guidelines of the original paper. This function 
        returns the patches (a tensor of size (N, M, M, C) where N is the number of patches, M is the width/height of each patch, 
        and C is the number of channels), along with the coordinates of the patch centers and the average color values of each patch.'''
        patches, positions, patches_color_means = get_patches(observation, self.L, self.N, self.M, self.S)
        flattened_patches = torch.reshape(patches, (self.N, self.d_in))
        '''In the following 6 lines, by projecting the flattened patches into Keys and Queries (see the SelfAttention module), 
        I derive an attention matrix with dimensions (N, N). Then, To determine the patch importance vector, I apply 
        softmax to each row of the matrix and subsequently sum it along the columns. Next, sorting the patch importance 
        vector, I extract the indices of the K highest-ranking patches.'''
        attention_matrix = self.attention(flattened_patches)
        softmax = nn.Softmax(dim=1)
        attention_matrix = softmax(attention_matrix)
        patch_importance_vector = torch.einsum('ij->j', [attention_matrix])
        patch_importance_indices = torch.argsort(input=patch_importance_vector, descending=True)
        patch_importance_best_k_indices = patch_importance_indices[:self.K]

        '''This following code block is executed only when training or testing the agent using the principles of the 
        'Double' Self Attention concept I introduced. The idea was to utilize the first self-attention from the original 
        paper to select the top K patches (most crucial for the task at each timestep). 
        Then, without losing the mapping of the centers' coordinates to the original observation, these selected patches 
        were used to compute a new self-attention (named 'Double'). This 'Double' self-attention calculates the dot 
        product only between these K best patches, already chosen by the first self-attention and not 
        with respect to the entire observation.'''
        if self.active_double_self_attention_on_best_patches == True:
            if self.isTest and self.render_mode == 'human':
                best_k_positions = f(positions, patch_importance_best_k_indices) # 'f'' is the function that maps the importance indices of the top K patches to their respective coordinates.
                show_blue_or_white_patches(observation, best_k_positions, self.M, self.K, "white") # Attention visualization function used to render which patches the agent focuses on.
            bst_ptch_mapping = map_best_patches_on_obs(patch_importance_best_k_indices, self.K) # Return a dictonary that remembers the indices of best k patches selcted from the first attention.
            self.k_second = math.floor(self.K / 2) # Set the new K for the 'Double' self attention.
            best_k_patches = patches[patch_importance_best_k_indices]
            flattened_best_patches = torch.reshape(best_k_patches, (self.K, self.d_in)) # Flatten the patches.
            best_patches_attention_matrix = self.double_attention_on_patches(flattened_best_patches) # Compute the Double Self Attention on the previous K best patches obtaied.
            softmax_2 = nn.Softmax(dim=1)
            best_patches_attention_matrix = softmax_2(best_patches_attention_matrix) # Apply the softmax on each rows.

            best_patches_importance_vector = torch.einsum('ij->j', [best_patches_attention_matrix]) # Compute the best patches importance vector.
            best_patches_importance_indices = torch.argsort(input=best_patches_importance_vector, descending=True) # Order the vetor in descending order.
            patch_importance_best_k_second_indices = best_patches_importance_indices[:self.k_second] # Extract the new best k_second indices.
            best_patches_mapped_indices = map_new_indices_to_old_indices(bst_ptch_mapping, patch_importance_best_k_second_indices) # Map new indices to old indices.
            patch_importance_best_k_indices = best_patches_mapped_indices


        positions = f(positions, patch_importance_best_k_indices)
        features = positions
        features = torch.div(features, self.L) #I normalize the positions by dividing the largest possible value so that each coordinate is between 0 and 1.
        if self.feature_retrieval_strategy == 'colors_and_positions':
            '''The get_best_patches_color_means function is called when I want to train or test the agent with the coordinates 
            of patch centers and their corresponding color means across all three RGB channels simultaneously.'''
            color_means = get_best_patches_color_means(patches_color_means, patch_importance_best_k_indices)
            features = torch.cat((features, color_means), dim=1) # Concatenate center coordinates with average color values of the top K patches.

        if self.isTest and self.render_mode == 'human':
            show_blue_or_white_patches(observation, positions, self.M, math.floor(self.K / self.num_attentions), "blue") # Attention visualization function used to render which patches the agent focuses on.
        features = torch.reshape(features, (1, -1))
        
        with torch.autograd.no_grad():
            features = torch.unsqueeze(features, 0)
            '''The features extracted from the top-k patches are fed into the LSTM module, which computes and returns probabilities 
            representing the likelihood of each action being executed in the next step, based on the previous events.'''
            actions = self.LSTMController.forward(features)
        actions = torch.reshape(actions, (-1,))
        '''This last code snippet in the "take_an_action(...)" method determines the number of episodes (CMA-ES iterations) 
        for which I want to induce additional exploration by the agent. The idea is that, for each episode during the first 200 
        iterations (this is a hyperparameter depending on the task, available time, and desired level of exploration), the action 
        executed at each timestep receives a penalty of 0.05 on the probability of being chosen in the next step of the episode. 
        At the same time, all other unexecuted actions in the current timestep receive a bonus of 0.05 on the probability of being 
        selected in the next step. This way, if the main network consisting of self-attention and LSTM is pushing the agent to 
        behave consistently, this exploration intensification mechanism gradually weakens the repeatedly chosen action's probability 
        while rapidly boosting the probabilities of other actions.'''
        if self.isTest == False and episode <= 200: #Stop wide exploration after the first 200 episodes
            actions = actions - self.penalty_actions
        action = torch.argmax(actions)
        action = action.item()

        '''To make the agent's task more challenging, we restricted its ability to move diagonally, 
        which would have made the task easier. For instance, if the agent needed to move towards 
        the bottom-left corner and was in the central position, allowing diagonal moves would 
        have enabled it to directly move diagonally. However, in this case, by disallowing 
        diagonal moves, the agent now requires two consecutive steps to accomplish what could be 
        done in one step with diagonal moves. For example, if the agent needs to move towards the 
        bottom-left corner, it must first perform the "backward" action and then the "downward" 
        action in two separate steps to emulate the desired behavior. This forces the agent to 
        minimize errors and avoid dead ends, making it more challenging to escape difficult situations.'''
        if action == 0:
            name_action = "0"
            output = 1
        elif action == 1:
            name_action = "1"
            output = 3
        elif action == 2:
            name_action = "2"
            output = 5
        elif action == 3:
            name_action = "3"
            output = 7
        elif action == 4:
            name_action = "4"
            output = 9

        self.remeber_actions[name_action] += 0.05
        for key in self.remeber_actions.keys():
            if key != name_action:
                if key == "7":
                    self.remeber_actions[key] -= 0.05
                elif key == "4":
                    self.remeber_actions[key] -= 0.05
                elif key != "4" and key != "7":
                    self.remeber_actions[key] -= 0.05




        self.penalty_actions[0] = self.remeber_actions["0"]/10
        self.penalty_actions[1] = self.remeber_actions["1"]/10
        self.penalty_actions[2] = self.remeber_actions["2"]/10
        self.penalty_actions[3] = self.remeber_actions["3"]/10
        self.penalty_actions[4] = self.remeber_actions["4"]/10

        return output

    def rollout(self, env, num_rollouts, episode):
        total_reward = 0
        for i in range(num_rollouts):
            # This cycle executes num_rollouts rollouts
            observation = env.reset()
            self.LSTMController.reset() # At the start of each game episode, the learnable parameters of the LSTM controller are reset.
            done = False
            step = 0
            while not done and step < 1000:
                '''This 'while' loop enables the agent to navigate the game environment. It takes action using 
                the described method, take_an_action(...), which activates the entire network. It then calls the step 
                method on the environment, linking the chosen action to the environment. The loop calculates the reward 
                value resulting from that action and the observation to be used in the next step. And the cycle repeats.'''
                action = self.take_an_action(observation, episode)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
                
        self.remeber_actions = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
        self.penalty_actions = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        return total_reward/num_rollouts

    def get_learnable_parameters_number(self):
        return self.learnable_parameters_number
        
    def inizialize_learnable_parameters(self, ler_parameters):
        '''In this method, at each iteration of CMA-ES, we set the values of the learnable parameters to the values of 
        the best solution returned and evaluated by the same genetic algorithm.'''
        ler_parameters = torch.from_numpy(ler_parameters)
        ler_parameters = ler_parameters.float()
        idx = 1
        self_attention_parameters_number = 2 * self.d_in * self.d
        new_ler_self_attention_parameters = ler_parameters[:self_attention_parameters_number]
        if self.active_double_self_attention_on_best_patches == True:
            new_ler_double_self_attention_parameters = ler_parameters[self_attention_parameters_number : 2 * self_attention_parameters_number]
            idx = 2
            self.double_attention_on_patches.set_params(new_ler_double_self_attention_parameters)
        new_ler_lstm_parameters = ler_parameters[idx * self_attention_parameters_number:]
        self.attention.set_params(new_ler_self_attention_parameters)
        self.LSTMController.set_params(new_ler_lstm_parameters)
