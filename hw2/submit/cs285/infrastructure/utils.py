import numpy as np
import time
import copy

############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # DONE: get this from hw1

    # initialize env for the beginning of a new rollout
    ob = env.reset() # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob) # HINT: query the policy's get_action function
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # DONE end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        # if (steps % 500) == 0:
            # print("steps = {}".format(steps))
        
        if done or (steps >= max_path_length):
            rollout_done = True # HINT: this is either 0 or 1
            #print('steps = {} ; max_path_length = {}'.format(steps, max_path_length))
            #print('rollout is done')
        else:
            rollout_done = False

        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    # DONE: get this from hw1
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        DONE implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    # this is a single batch...
    # while loop collects enough data for one batch 
    # if data per rollout not enough.
    timesteps_this_batch = 0
    paths = [] # list to be filled with dictionaries
    while timesteps_this_batch < min_timesteps_per_batch:

        # curr_path_dict = {"observation" : np.array(obs, dtype=np.float32),
                        # "image_obs" : np.array(image_obs, dtype=np.uint8),
                        # "reward" : np.array(rewards, dtype=np.float32),
                        # "action" : np.array(acs, dtype=np.float32),
                        # "next_observation": np.array(next_obs, dtype=np.float32),
                        # "terminal": np.array(terminals, dtype=np.float32)}
        curr_path_dict = sample_trajectory(env, policy, max_path_length, render)
        timesteps_this_batch += get_pathlength(curr_path_dict)
        paths.append(curr_path_dict)


    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    # DONE: get this from hw1
    """
        Collect ntraj rollouts.

        DONE implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    n_paths = [] # list to be filled with dictionaries

    for i in range(ntraj):
        # print("sampling {}-th trajectory".format(i))
        curr_path_dict = sample_trajectory(env, policy, max_path_length, render)
        n_paths.append(curr_path_dict)

    return n_paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

############# EXTRA ZZP HELPER
def get_pathlength(path):
    return len(path["reward"])