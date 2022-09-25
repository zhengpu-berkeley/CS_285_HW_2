import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data 
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        qvals = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, qvals, terminals)
        train_log = self.actor.update(observations, actions, advantages, qvals)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # Note: rewards_list is a list of lists of rewards with the inner list
        # being the list of rewards for a single trajectory.
        
        # HINT: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these).

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        # Note: q_values should first be a 2D list where the first dimension corresponds to 
        # trajectories and the second corresponds to timesteps, 
        # then flattened to a 1D numpy array. ----> use np.concatenate instead of np.array

        if not self.reward_to_go:
            q_val_list = []
            for rewards in rewards_list:
                discounted_return = self._discounted_return(rewards)
                q_val_list.append(discounted_return)
            return np.concatenate(q_val_list)

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_val_list = []
            for rewards in rewards_list:
                discounted_cumsum = self._discounted_cumsum(rewards)
                q_val_list.append(discounted_cumsum)
            return np.concatenate(q_val_list)


    def estimate_advantage(self, obs: np.ndarray, rews_list: np.ndarray, q_values: np.ndarray, terminals: np.ndarray):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors

            assert values_unnormalized.ndim == q_values.ndim
            ## DONE: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values
            values = values_unnormalized * np.std(q_values) + np.mean(q_values)

            batch_size = obs.shape[0]

            ## append T+1 being 0 for recursive calculation
            values = np.append(values, [0])
            ## flattings rews_list
            rews = np.concatenate(rews_list)
            ## Empty array to populate with advantages
            advantages = np.zeros(batch_size + 1)

            # if we do gae
            print('here:::')
            print(self.gae_lambda)
            print('here ---- here')
            if self.gae_lambda is not None:
                
                for i in reversed(range(batch_size)):
                    ## append a dummy T+1 value for simpler recursive calculation
                    values = np.append(values, [0])

                    ## combine rews_list into a single array
                    rews = np.concatenate(rews_list)

                    ## create empty numpy array to populate with GAE advantage
                    ## estimates, with dummy T+1 value for simpler recursive calculation
                    batch_size = obs.shape[0]
                    advantages = np.zeros(batch_size + 1)

                    for i in reversed(range(batch_size)):
                        ## TODO: recursively compute advantage estimates starting from
                            ## timestep T.
                        ## HINT: use terminals to handle edge cases. terminals[i]
                            ## is 1 if the state is the last in its trajectory, and
                            ## 0 otherwise.
                        delta_i = rews[i] + self.gamma * values[i+1] - values[i]
                        advantages[i] = self.gamma * self.gae_lambda * advantages[i+1] + delta_i
                    # remove dummy advantage
                    advantages = advantages[:-1]
                        

            # if we do not do gae
            else:
                for i in reversed(range(batch_size)):
                    if terminals[i] == 1:
                        advantages[i] = rews[i] - values[i]
                    else:
                        next_rew_sum_recovered_from_adv = advantages[i+1] + values[i+1]
                        rew_sum = rews[i] + self.gamma * next_rew_sum_recovered_from_adv
                        advantages[i] =  rew_sum - values[i]
                advantages = advantages[:-1]
            

        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            small_number = 0.000001
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + small_number)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################


    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        reward_len = len(rewards)
        rewards_np = np.array(rewards)
        discount_weights = np.logspace(0, reward_len, num=reward_len, base=self.gamma, endpoint=False)
        weighted_rewards = np.sum(rewards_np * discount_weights)
        list_of_discounted_returns = np.ones(reward_len) * weighted_rewards

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        reward_len = len(rewards)
        reward_from_last = [rewards[reward_len-1]]
        for index in range(reward_len-2,-1,-1):
            cumsum_reward = reward_from_last[-1]*(self.gamma) + rewards[index]
            reward_from_last.append(cumsum_reward)
        list_of_discounted_cumsums = reward_from_last
        list_of_discounted_cumsums.reverse()
        return np.array(list_of_discounted_cumsums)


