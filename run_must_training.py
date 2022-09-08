from collections import deque
import os
import sys
root_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(root_path))


import time
import random
from tqdm import tqdm
from beeprint import pp

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from simulator.user import User
from simulator.loose_user import LooseUser
from simulator.system import System
from simulator.loose_system import LooseSystem
# from simulator.env import Enviroment
from simulator.env_multi_users import Enviroment

import simulator.dialog_config as dialog_config

from rl.pg_mus import PolicyGradientREINFORCE
from rl.policy_model import Net

from sequicity_user.seq_user import Seq_User
from sequicity_user.seq_user_act import Seq_User_Act

# from simulator_gpt.gpt_user import GPT_User
from simulator_gpt_act.gpt_user import GPT_User

from config import Config

config = Config()
state_dim   = dialog_config.STATE_DIM
num_actions = dialog_config.SYS_ACTION_CARDINALITY

MODE = dialog_config.RL_WARM_START
WARM_START_EPISODES = config.warm_start_episodes
MAX_EPISODES = config.n_episodes
UNIFORM_EPISODES = config.uniform_episodes
RESET_EPISODES = config.reset_episodes
reset_number = 0
MAX_STEPS    = 200
TEST_EVERY = 2000
NUM_TEST = 200

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

device = config.device
print('device = ', device)
# pp(config)
# print('---' * 30)
# pp(dialog_config)

if config.loose_agents:
    system = LooseSystem(config=config)
else:
    system = System(config=config)

us_agen_t = LooseUser(nlg_sample=False, nlg_template=True)
us_agen_r = LooseUser(nlg_sample=True, nlg_template=False)

# us_agen_t = GPT_User(nlg_sample=None, nlg_template=None, 
#                         model_path='simulator_gpt_act/models/at/b4_g16_lr0.005/epoch32_trloss0.0803_gpt2')
# us_agen_r = GPT_User(nlg_sample=None, nlg_template=None,
#                         model_path='simulator_gpt_act/models/ar/b4_g16_lr0.005/epoch52_trloss0.1133_gpt2')
# us_agen_g = GPT_User(nlg_sample=None, nlg_template=None,
#                         model_path='simulator_gpt_act/models/ag/b4_g16_lr0.005/epoch33_trloss0.0777_gpt2')
us_gpt = GPT_User(nlg_sample=None, nlg_template=None)

# users = [us_agen_t, us_agen_r, us_agen_g, us_gpt]

us_rnn_t = Seq_User_Act(nlg_sample=False, nlg_template=True)
# us_rnn_r = Seq_User_Act(nlg_sample=True, nlg_template=False)
users = [us_agen_t, us_agen_r, us_rnn_t, us_gpt]

env = Enviroment(users=users, system=system, verbose=True, config=config)
sys_act = None
status = []



def get_bit_vector(system):
    # index_to_action_dict = {0: SystemAct.ASK_TYPE,
    #                         1: [SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT, SystemAct.NO_OTHER],
    #                         2: SystemAct.PROVIDE_INFO,
    #                         3: [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL],
    #                         4: SystemAct.GOODBYE,
    #                         5: SystemAct.ASK_RESERVATION_INFO}

    if config.with_bit_all:
        # not reservation, 5 is 0; len(results) == 0, 235 are zero; len(informed)==0, 12345 are zero
        # no repetition, if len(informed)==3, 0 is zero; if reservable, 5 is zero
        reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
        reservable = np.all(reservable)
        small_value = config.small_value
        if len(system.state['informed']['name']) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value
            bit_vecs[0] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1


            if not reservable:
                bit_vecs[3] = small_value
            else:
                bit_vecs[3] = 1
                bit_vecs[5] = small_value
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        assert len(informed_so_far)
        if np.sum(informed_so_far) > 1:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1

            if not reservable:
                bit_vecs[3] = small_value
            else:
                #bit_vecs[0] = 0
                bit_vecs[3] = 1
                bit_vecs[5] = small_value

            if np.all(informed_so_far):
                bit_vecs[0] = 0

            return bit_vecs
        else:
            bit_vecs = [1, small_value, small_value, small_value, small_value, small_value]
            return bit_vecs

    elif config.with_bit_more:
        # not reservation, 5 is 0; len(results) == 0, 235 are zero; len(informed)==0, 12345 are zero
        reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
        reservable = np.all(reservable)
        small_value = config.small_value
        if len(system.state['informed']['name']) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value
            # bit_vecs[0] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1


            if not reservable:
                bit_vecs[3] = small_value
            else:
                bit_vecs[3] = 1
                # bit_vecs[5] = small_value
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        assert len(informed_so_far)
        if np.sum(informed_so_far) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1

            if not reservable:
                bit_vecs[3] = small_value
            else:
                #bit_vecs[0] = 0
                bit_vecs[3] = 1
                # bit_vecs[5] = small_value

            # if np.all(informed_so_far):
            #     bit_vecs[0] = 0

            return bit_vecs
        else:
            bit_vecs = [1, small_value, small_value, small_value, small_value, small_value]
            return bit_vecs

    elif config.with_bit_rep_only:
        reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
        reservable = np.all(reservable)
        small_value = config.small_value
        if len(system.state['informed']['name']) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            # bit_vecs[4] = small_value
            bit_vecs[0] = small_value
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        if np.all(informed_so_far):
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[0] = small_value
        else:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY

        return bit_vecs


def load_policy_model(model_dir="model/test_nlg_no_warm_up_with_nlu.pkl"):
    print('model_dir = ', model_dir)
    model = torch.load(model_dir, map_location='cuda:0')
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY, config=config).to(device)
    net.load_state_dict(model)
    net.eval()
    return net


def compute_ucb(mab_dict, multi_armed_bandits_dict, steps, reset_n):
    print('multi_armed_bandits_dict = ', multi_armed_bandits_dict)
    print('mab_dict = ', mab_dict)

    total = 0
    mab_dist = {}
    min_succ_rate = min(mab_dict.values())*0.75
    if not multi_armed_bandits_dict:
        for k, v in mab_dict.items():
            mab_dist[k] = 1/(mab_dict[k]-min_succ_rate)
            total += mab_dist[k]
    else:
        # new_t = sum(list(multi_armed_bandits_dict.values()))
        # assert t == new_t
        for k, v in multi_armed_bandits_dict.items():
            real_steps = steps - reset_n*RESET_EPISODES
            b = np.sqrt(np.log(real_steps) / v)
            x_j = mab_dict[k] - min_succ_rate
            print('abs steps = ', steps, 'rel steps = ', real_steps, 'check = ', k, b, x_j)

            mab_dist[k] = 1/(x_j + b)
            total += 1/(x_j + b)

    mab_dist_list = []
    dist_sum = 0
    for i, u in enumerate(env.users):
        name = u.name
        dist_sum += mab_dist[name] / total
        mab_dist_list.append(dist_sum)
    print('mab_dist = ', mab_dist, mab_dist_list)

    if steps >= UNIFORM_EPISODES:
        print('Reset user simulator distribution with UCB')
        env.reset_user_dist(mab_dist_list)


def run_one_dialog(env, pg_reinforce, assigned_name=None):
    print("#"*30)
    print("Test Episode "+"-"*20)
    print("#"*30)
    cur_mode = dialog_config.RL_TRAINING
    state = env.reset(mode=cur_mode, user_name=assigned_name)
    user_name = env.user.name
    total_rewards = 0
    total_t = 0

    while True:
        if config.with_bit:
            bit_vecs = get_bit_vector(system)
        else:
            bit_vecs = None
        # print('*** bit_vec: ', bit_vecs)
        action = pg_reinforce.sampleAction(state, rl_test=True, bit_vecs=bit_vecs)
        action = action.item()
        next_state, reward, done = env.step(provided_sys_act=action, mode=cur_mode)

        total_rewards += reward
        # reward = -10 if done else 0.1 # normalize reward
        pg_reinforce.storeRollout(state, action, reward, bit_vecs=bit_vecs)

        state = next_state
        total_t += 1
        if done:
            break

    pg_reinforce.cleanUp()
    print("Finished after {} timesteps".format(total_t))
    print('dialog_status: {}'.format(env.success))
    print("Reward for this episode: {}".format(total_rewards))
    print("#" * 30)

    return total_rewards, total_t, user_name, env.success

def test(env, pg_reinforce, n=50):
    reward_list = []
    dialogLen_list = []
    success_list = []
    # print(i_episode)

    mab_dict = {}
    for u in users:
        u_succ_list = []
        u_name = u.name
        for i_test in range(n):
            assert len(pg_reinforce.reward_buffer) == 0
            cur_reward, cur_dialogLen, user_name, cur_success = run_one_dialog(env, pg_reinforce, assigned_name=u_name)
            assert cur_success is not None
            reward_list.append(cur_reward)
            dialogLen_list.append(cur_dialogLen)
            # print(cur_reward)
            # print(cur_dialogLen)
            # print(cur_success)
            success_list.append(int(cur_success))
            u_succ_list.append(int(cur_success))
        succ_rate = sum(u_succ_list)/n
        mab_dict[u_name] = succ_rate
    return reward_list, dialogLen_list, success_list, mab_dict


if config.resume:
    policy_net = load_policy_model(config.resume_rl_model_dir)
else:
    policy_net = Net(state_dim=state_dim, num_actions=num_actions, config=config).to(device)#

optimizer = optim.Adam(lr=config.lr, params=policy_net.parameters(), weight_decay=5e-5)

pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=state_dim,
                     num_actions=num_actions ,
                     config=config,
                     device=device,
                     init_exp=config.init_exp,         # initial exploration prob
                     reset_exp=config.reset_exp,
                     final_exp=config.final_exp,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=config.discounted_factor, # discount future rewards
                     reg_param=0.01,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_every=100,
                     batch_size=config.batch_size,
                     verbose=True,
                     with_bit=config.with_bit,
                     replay=config.replay)


MAX_TEST_SUC = -1
cnt = 0
multi_armed_bandits_dict = {}
# _, _, _, mab_dict = test(env=env, pg_reinforce=pg_reinforce, n=NUM_TEST)
# compute_ucb(mab_dict, multi_armed_bandits_dict, 0)

while True:
    print("-------------------START OVER-------------------")
    episode_history = deque(maxlen=100)
    mean_reward_test = []
    mean_len_test = []
    mean_success_test = []
    test_id = []
    cur_time = "-".join([str(t) for t in list(time.localtime())])
    for i_episode in tqdm(range(MAX_EPISODES)):
        t = i_episode - WARM_START_EPISODES + 1

        if i_episode >= WARM_START_EPISODES:
            MODE = dialog_config.RL_TRAINING

        if MODE == dialog_config.RL_TRAINING and (i_episode - WARM_START_EPISODES + 1) % TEST_EVERY == 0:
            reward_list, len_list, success_list, mab_dict = test(env=env, pg_reinforce=pg_reinforce, n=NUM_TEST)
            mean_reward_test.append(np.mean(reward_list))
            mean_len_test.append(np.mean(len_list))
            mean_success_test.append(np.mean(success_list))
            test_id.append(i_episode - WARM_START_EPISODES)

            compute_ucb(mab_dict, multi_armed_bandits_dict, t, reset_number)

        if t % RESET_EPISODES == 0:
            pg_reinforce.resetModel()
            multi_armed_bandits_dict = {}
            reset_number += 1

        print("-*-" * 20)
        # initialize
        state = env.reset(mode=MODE)
        usr_name = env.user.name
        if usr_name not in multi_armed_bandits_dict:
            multi_armed_bandits_dict[usr_name] = 0
        multi_armed_bandits_dict[usr_name] += 1

        total_rewards = 0     # each dialog reward
        total_t = 0

        while True:
            if config.with_bit:
                bit_vecs = get_bit_vector(system)
            else:
                bit_vecs = None
            if MODE == dialog_config.RL_TRAINING:
                action = pg_reinforce.sampleAction(state, bit_vecs=bit_vecs, rl_test=False)
                action = action.item()
            elif MODE == dialog_config.RL_WARM_START:
                action = None
            next_state, reward, done = env.step(provided_sys_act=action, mode=MODE)
            total_rewards += reward

            if MODE == dialog_config.RL_WARM_START:
                action = env.system.action_to_index(env.last_sys_act.act)
            pg_reinforce.storeRollout(state, action, reward, bit_vecs=bit_vecs, user_type=usr_name)

            state = next_state
            total_t += 1
            if done:
                break

        pg_reinforce.updateModel(mode=MODE, user_type=usr_name)
        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}".format(i_episode))
        print("Finished after {} timesteps".format(total_t+1))
        print('dialog_status: {}'.format(env.success))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
        if mean_rewards >= 48.0 and len(episode_history) >= 100:
            print("Environment {} solved after {} episodes".format("Restaurant", i_episode+1))
            break

        if i_episode > 2500 and mean_rewards < -9:
            break

        if MODE == dialog_config.RL_TRAINING and ((i_episode - WARM_START_EPISODES + 1) % TEST_EVERY == 0):
            print('mean_reward_test = ', mean_reward_test)
            test_history = zip(test_id, mean_reward_test, mean_len_test, mean_success_test)

            pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(config.save_dir + str(cnt) + "_" + cur_time + ".csv", index=False)
            if mean_success_test[-1] >= MAX_TEST_SUC:
                MAX_TEST_SUC = mean_success_test[-1]
                torch.save(policy_net.state_dict(), config.save_dir + str(cnt) + "_" + cur_time + '_' + str(i_episode) +   ".pkl")

    if mean_success_test[-1] >= MAX_TEST_SUC:
        MAX_TEST_SUC = mean_success_test[-1]
        print("max_test_success in the end", MAX_TEST_SUC)
        torch.save(policy_net.state_dict(), config.save_dir + str(cnt) + "_" + cur_time + '_' + str(i_episode) +   ".pkl")

    print(mean_reward_test)
    test_history = zip(test_id, mean_reward_test, mean_len_test, mean_success_test)

    pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(
        config.save_dir + str(cnt) + "_" + cur_time + ".csv", index=False)

    if i_episode == (MAX_EPISODES-1):
        break

    cnt += 1