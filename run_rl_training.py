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
from simulator.env import Enviroment
import simulator.dialog_config as dialog_config

from rl.my_pg import PolicyGradientREINFORCE
from rl.policy_model import Net

from sequicity_user.seq_user import Seq_User
from sequicity_user.seq_user_act import Seq_User_Act

from config import Config

config = Config()
device = config.device
print('device = ', device)

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

if config.loose_agents:
    user = LooseUser(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = LooseSystem(config=config)
else:
    user = User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = System(config=config)

if config.use_sl_simulator:
    if config.use_sl_generative:
        user = Seq_User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    else:
        user = Seq_User_Act(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)

pp(config)
print('---' * 30)
pp(dialog_config)
print('---' * 30)
print('\nUser type =', user.name)


def run_one_dialog(env, pg_reinforce):
    print("#"*30)
    print("Test Episode "+"-"*20)
    print("#"*30)
    cur_mode = dialog_config.RL_TRAINING
    state = env.reset(mode=cur_mode)
    total_rewards = 0
    total_t = 0

    while True:
        # env.render()
        # print(state[np.newaxis, :])
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

    return total_rewards, total_t, env.success

def test(env, pg_reinforce, n=50):
    reward_list = []
    dialogLen_list = []
    success_list = []
    # print(i_episode)
    for i_test in range(n):
        assert len(pg_reinforce.reward_buffer) == 0
        cur_reward, cur_dialogLen, cur_success = run_one_dialog(env, pg_reinforce)
        assert cur_success is not None
        reward_list.append(cur_reward)
        dialogLen_list.append(cur_dialogLen)
        # print(cur_reward)
        # print(cur_dialogLen)
        # print(cur_success)
        success_list.append(int(cur_success))
    return reward_list, dialogLen_list, success_list


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

            # if len(system.state['results']) == 0:
            #     bit_vecs[2] = small_value
            #     bit_vecs[3] = small_value
            #     bit_vecs[5] = small_value
            # else:
            #     bit_vecs[2] = 1
            #     bit_vecs[3] = 1
            #     bit_vecs[5] = 1
            #
            # if not reservable:
            #     bit_vecs[3] = small_value
            # else:
            #     bit_vecs[3] = 1
            #     bit_vecs[5] = small_value
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


state_dim   = dialog_config.STATE_DIM
num_actions = dialog_config.SYS_ACTION_CARDINALITY
env = Enviroment(user=user, system=system, verbose=True, config=config)

if config.resume:
    policy_net = load_policy_model(config.resume_rl_model_dir)
else:
    policy_net = Net(state_dim=state_dim, num_actions=num_actions, config=config).to(device)#


optimizer = optim.Adam(lr=config.lr, params=policy_net.parameters(),
                                  weight_decay=5e-5)

pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=state_dim,
                     num_actions=num_actions ,
                     config=config,
                     device=device,
                     init_exp=config.init_exp,         # initial exploration prob
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

MODE = dialog_config.RL_WARM_START
WARM_START_EPISODES = config.warm_start_episodes
MAX_EPISODES = config.n_episodes
MAX_STEPS    = 200
TEST_EVERY = 1000
NUM_TEST = 200


MAX_TEST_SUC = -1
cnt = 0
while True:
    print("-------------------START OVER-------------------")
    episode_history = deque(maxlen=100)
    mean_reward_test = []
    mean_len_test = []
    mean_success_test = []
    test_id = []
    cur_time = "-".join([str(t) for t in list(time.localtime())])
    for i_episode in tqdm(range(MAX_EPISODES)):
        if i_episode >= WARM_START_EPISODES:
            MODE = dialog_config.RL_TRAINING

        if MODE == dialog_config.RL_TRAINING and (i_episode - WARM_START_EPISODES + 1) % TEST_EVERY == 0:
            reward_list, len_list, success_list = test(env=env, pg_reinforce=pg_reinforce, n=NUM_TEST)
            mean_reward_test.append(np.mean(reward_list))
            mean_len_test.append(np.mean(len_list))
            mean_success_test.append(np.mean(success_list))
            test_id.append(i_episode - WARM_START_EPISODES)

        print("-*-" * 20)
        # initialize
        state = env.reset(mode=MODE)
        total_rewards = 0
        total_t = 0

        while True:
            if config.with_bit:
                bit_vecs = get_bit_vector(system)
            else:
                bit_vecs = None
            if MODE == dialog_config.RL_TRAINING:
                action = pg_reinforce.sampleAction(state, bit_vecs=bit_vecs,
                                                   rl_test=False)
                action = action.item()
            elif MODE == dialog_config.RL_WARM_START:
                action = None
            next_state, reward, done = env.step(provided_sys_act=action, mode=MODE)
            total_rewards += reward

            if MODE == dialog_config.RL_WARM_START:
                action = env.system.action_to_index(env.last_sys_act.act)
            pg_reinforce.storeRollout(state, action, reward, bit_vecs=bit_vecs)

            state = next_state
            total_t += 1
            if done:
                break

        pg_reinforce.updateModel(mode=MODE)
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