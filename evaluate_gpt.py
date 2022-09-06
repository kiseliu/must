from collections import deque
import sys
sys.path.append('./')
import json

import time
import random
import argparse
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
from simulator.env_new import Enviroment
import simulator.dialog_config as dialog_config

from rl.my_pg import PolicyGradientREINFORCE
from rl.policy_model import Net

from simulator_gpt_act.gpt_user import GPT_User

from evaluation_matrix_config import EvalConfig
import pdb

config = EvalConfig()
device = config.device

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

# ########################## #########################
parser = argparse.ArgumentParser()
# parser.add_argument('-resume_rl_model_dir', default='model/save/gpt_simulator/best/0_2021-8-6-2-50-38-4-218-0_5399.pkl')
# parser.add_argument('-resume_rl_model_dir', default='model/save/template/oneHot_newReward_bitMore/0_2019-5-18-21-59-15-5-138-1.pkl')
# parser.add_argument('-resume_rl_model_dir', default='model/save/nlg_sample/oneHot_newReward_bitMore/best/0_2019-5-19-15-13-5-6-139-1.pkl')
# parser.add_argument('-resume_rl_model_dir', default='model/save/seq2seq/oneHot_newReward_bitMore/0_2019-5-19-22-28-16-6-139-1.pkl')
# parser.add_argument('-resume_rl_model_dir', default='model/save/sl_simulator/template/oneHot_oldReward_bitMore/0_2019-5-19-23-46-10-6-139-1.pkl')
# parser.add_argument('-resume_rl_model_dir', default='model/save/sl_simulator/retrieval/oneHot_oldReward_bitMore/best/0_2019-5-19-19-2-18-6-139-1.pkl')
# parser.add_argument('-resume_rl_model_dir', default='model/save/sl_simulator/oneHot_oldReward_bitMore/best/0_2019-5-19-3-27-15-6-139-1.pkl')
parser.add_argument('-resume_rl_model_dir', default='model/save/gpt_simulator/reward/oneHot_oldReward_bitMore/0_2021-9-8-9-41-20-2-251-0_27299.pkl')
parser.add_argument('-config', nargs='*')
args = parser.parse_args()

if args.config:
    for pair in args.config:
        k, v = tuple(pair.split('='))
        dtype = type(getattr(config, k))
        if dtype == type(None):
            raise ValueError()
        if dtype is bool:
            v = False if v == 'False' else True
        else:
            v = dtype(v)
        setattr(config, k, v)

if config.use_sl_simulator == True:
    config.use_new_reward = False
else:
    config.use_new_reward = True
output_name = 'gpt_aug'
print('output name = ', output_name)



if config.loose_agents:
    # user = LooseUser(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = LooseSystem(config=config)
else:
    # user = User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = System(config=config)

user = GPT_User(nlg_sample=None, nlg_template=None)
# pp(config)
# print('---' * 30)
# pp(dialog_config)


def run_one_dialog(env, pg_reinforce, ith):
    print()
    print("#"*30)
    print("Test Episode " + str(ith) +"-"*20)
    print("#"*30)
    cur_mode = dialog_config.RL_TRAINING
    state = env.reset(mode=cur_mode)   # user state
    # print('user state = ', state)
    total_rewards = 0
    total_t = 0

    while True:
        if config.with_bit:
            bit_vecs = get_bit_vector(system)
        else:
            bit_vecs = None
        # print('bit_vec: ', bit_vecs)
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
    
    # pp(env.dials)

    pg_reinforce.cleanUp()
    # print("Finished after {} timesteps".format(total_t))
    print('dialog_status: {}'.format(env.success))
    # print("Reward for this episode: {}".format(total_rewards))
    print("#" * 30)
    print()
    return total_rewards, total_t, env.success

def test(env, pg_reinforce, n=50):
    reward_list = []
    dialogLen_list = []
    success_list = []
    all_dials = []
    # print(i_episode)
    for i_test in tqdm(range(n)):
        assert len(pg_reinforce.reward_buffer) == 0
        cur_reward, cur_dialogLen, cur_success = run_one_dialog(env, pg_reinforce, i_test)
        assert cur_success is not None
        reward_list.append(cur_reward)
        dialogLen_list.append(cur_dialogLen)
        # print(cur_reward)
        # print(cur_dialogLen)
        success_list.append(int(cur_success))
        # dials['success'] = int(cur_success)
        # all_dials.append(dials)

    # with open('dials.json', 'w', encoding='utf-8') as fw:
    #     json.dump(all_dials, fw, indent=2)
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
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        if np.all(informed_so_far):
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[0] = small_value
        else:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY

        return bit_vecs


def load_policy_model(model_dir="model/test_nlg_no_warm_up_with_nlu.pkl"):
    model = torch.load(model_dir, map_location='cpu')
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY, config=config).to(device)
    net.load_state_dict(model)
    net.eval()
    return net

print('*****sys type = ', args.resume_rl_model_dir)
policy_net = load_policy_model(args.resume_rl_model_dir)
optimizer = optim.Adam(lr=config.lr, params=policy_net.parameters(),
                                  weight_decay=5e-5)

state_dim   = dialog_config.STATE_DIM
num_actions = dialog_config.SYS_ACTION_CARDINALITY
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

env = Enviroment(user=user, system=system, verbose=True, config=config)
MODE = dialog_config.RL_TRAINING #dialog_config.RL_WARM_START
WARM_START_EPISODES = 0 #config.warm_start_episodes
MAX_EPISODES = 1
MAX_STEPS    = 200
TEST_EVERY = 1
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
        if MODE == dialog_config.RL_TRAINING and \
           (((i_episode - WARM_START_EPISODES + 1) % TEST_EVERY == 0)):# or (i_episode == WARM_START_EPISODES)):
            reward_list, len_list, success_list = test(env=env, pg_reinforce=pg_reinforce, n=NUM_TEST)
            full_result = zip(reward_list, len_list, success_list)
            pd.DataFrame(full_result, columns=["reward", "len", "success"]).to_csv(
                config.save_dir + output_name + "_full.csv", index=False)
            mean_reward_test.append(np.mean(reward_list))
            mean_len_test.append(np.mean(len_list))
            mean_success_test.append(np.mean(success_list))
            test_id.append(i_episode - WARM_START_EPISODES)

        print("-" * 20)
        # initialize
        state = env.reset(mode=MODE)
        total_rewards = 0
        total_t = 0

    # print(mean_reward_test)
    test_history = zip(test_id, mean_reward_test, mean_len_test, mean_success_test)
    pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(
        config.save_dir + output_name + ".csv", index=False)

    if i_episode == (MAX_EPISODES-1):
        break

    cnt += 1