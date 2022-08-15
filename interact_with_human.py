import random
from tqdm import tqdm
from beeprint import pp

import numpy as np

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

from evaluation_matrix_config import EvalConfig

config = EvalConfig()
device = config.device

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

# -------------------set the output name-------------------
rl_model_dir = './model/save/sl_simulator/oneHot_oldReward_bitMore/best/0_2019-5-19-3-27-15-6-139-1.pkl'

if rl_model_dir.split('/')[3] == 'sl_simulator':
    sysname = 'sl_' + rl_model_dir.split('/')[4]
else:
    sysname = 'rule_' + rl_model_dir.split('/')[3]
output_name = 'human_' + sysname
config.INTERACTIVE = True

# -------------------set user and system-------------------
if config.loose_agents:
    user = LooseUser(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = LooseSystem(config=config)
else:
    user = User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = System(config=config)

pp(config)
print('---' * 30)
pp(dialog_config)


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


def run_one_dialog(env, pg_reinforce):
    print("#"*30)
    print("Test Episode "+"-"*20)
    print("#"*30)
    cur_mode = dialog_config.INTERACTIVE
    state = env.reset(mode=cur_mode)   # user state
    total_rewards = 0
    total_t = 0

    while True:
        if config.with_bit:
            bit_vecs = get_bit_vector(system)
            print('sys state = ')
            pp(system.state)
        else:
            bit_vecs = None
        print('bit_vec: ', bit_vecs)
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


def load_policy_model(model_dir="model/test_nlg_no_warm_up_with_nlu.pkl"):
    model = torch.load(model_dir, map_location='cpu')
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY, config=config).to(device)
    net.load_state_dict(model)
    net.eval()
    return net

policy_net = load_policy_model(rl_model_dir)
optimizer = optim.Adam(lr=config.lr, params=policy_net.parameters(),
                                  weight_decay=5e-5)

pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=dialog_config.STATE_DIM,
                     num_actions=dialog_config.SYS_ACTION_CARDINALITY,
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
NUM_TEST = 50

for _ in tqdm(range(NUM_TEST)):
    assert len(pg_reinforce.reward_buffer) == 0
    cur_reward, cur_dialogLen, cur_success = run_one_dialog(env, pg_reinforce)