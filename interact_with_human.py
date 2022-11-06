import random
import os
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

# if rl_model_dir.split('/')[3] == 'sl_simulator':
#     sysname = 'sl_' + rl_model_dir.split('/')[4]
# else:
#     sysname = 'rule_' + rl_model_dir.split('/')[3]
# output_name = 'human_' + sysname

# -------------------set user and system-------------------
if config.loose_agents:
    user = LooseUser(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = LooseSystem(config=config)
else:
    user = User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = System(config=config)

# pp(config)
# print('---' * 30)
# pp(dialog_config)


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


def run_one_dialog(env, pg_reinforce, goal_id):
    print("#"*30)
    print("Test Episode "+"-"*20)
    print("#"*30)
    cur_mode = dialog_config.INTERACTIVE
    _, state, _ = env.reset(goal_id=goal_id, mode=cur_mode)   # user state
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
        next_state, reward, done, _ = env.step(provided_sys_act=action, mode=cur_mode)
        print('next_state = ', next_state)
        print('reward = ', reward)
        print('done = ', done)

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
    print('\nsys type = ', model_dir)
    model = torch.load(model_dir, map_location='cpu')
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY, config=config).to(device)
    net.load_state_dict(model)
    net.eval()
    return net


def interact(env, rl_model_dir, goal_id):
    policy_net = load_policy_model(rl_model_dir)
    optimizer = optim.Adam(lr=config.lr, params=policy_net.parameters(), weight_decay=5e-5)

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

    assert len(pg_reinforce.reward_buffer) == 0
    run_one_dialog(env, pg_reinforce, goal_id)

def compute_succ_rate():
    agent = [0, 0]
    agenr = [0, 0]
    rnnt = [0, 0]
    gpt_sl = [0, 0]
    gpt_must = [0, 0]
    gpt_uniform = [0, 0]
    gpt_merging = [0, 0]
    def extract_res(path):
        data = open(path, 'r').read().split('\n')
        for line in data:
            if line.startswith('# result'):
                return [path[-6:], line]
        return None

    dir_path = 'evaluation_results/human_machine_dials/'
    idx = []
    for f in os.listdir(dir_path):
        if f.startswith('dial'):
            f_path = dir_path + f
            # print(f_path)
            res = extract_res(f_path)
            if res:
                idx.append(res)

    idx_order = sorted(idx, key=lambda x: x[0])
    for l in idx_order:
        line = ' '.join(l)
        if 'all wrong' in line:
            agent[1]+=1
            agenr[1]+=1
            rnnt[1]+=1
            gpt_sl[1]+=1
            gpt_must[1]+=1
            gpt_uniform[1]+=1
            gpt_merging[1]+=1
        elif 'wrong' in line:
            if 'agent' in line:
                agent[1]+=1
            
            if 'agenr' in line:
                agenr[1]+=1
            
            if 'rnnt' in line:
                rnnt[1]+=1

            if 'gpt_sl' in line:
                gpt_sl[1]+=1

            if 'gpt_must' in line:
                gpt_must[1]+=1

            if 'gpt_uniform' in line:
                gpt_uniform[1]+=1
                
            if 'gpt_merging' in line:
                gpt_merging[1]+=1

    print('agent = ', agent, (50-agent[1])/50)
    print('agenr = ', agenr, (50-agenr[1])/50)
    print('rnnt = ', rnnt, (50-rnnt[1])/50)
    print('gpt_sl = ', gpt_sl, (50-gpt_sl[1])/50)
    print('gpt_must = ', gpt_must, (50-gpt_must[1])/50)
    print('gpt_uniform = ', gpt_uniform, (50-gpt_uniform[1])/50)
    print('gpt_merging = ', gpt_merging, (50-gpt_merging[1])/50)

if __name__ == '__main__':
    goal_ids = ['WOZ20421.json', 'SNG02248.json', 'WOZ20528.json', 'WOZ20650.json', 'WOZ20452.json', 
                'WOZ20225.json', 'WOZ20337.json', 'WOZ20208.json', 'WOZ20320.json', 'SNG0665.json', 
                'WOZ20109.json', 'WOZ20538.json', 'WOZ20469.json', 'WOZ20667.json', 'SNG0536.json', 
                'SNG0488.json', 'WOZ20379.json', 'WOZ20544.json', 'WOZ20162.json', 'WOZ20451.json', 
                'SNG0600.json', 'SNG0489.json', 'SNG1353.json', 'SNG1224.json', 'SNG0558.json', 
                'SNG0747.json', 'SNG0554.json', 'SNG0653.json', 'SNG0516.json', 'SNG01847.json', 
                'SSNG0167.json', 'SNG01162.json', 'SSNG0145.json', 'SSNG0080.json', 'SSNG0028.json', 
                'SNG0663.json', 'SSNG0146.json', 'SNG0620.json', 'SSNG0063.json', 'SSNG0096.json', 
                'SSNG0185.json', 'SSNG0078.json', 'SNG0633.json', 'SSNG0103.json', 'SSNG0066.json', 
                'SNG01611.json', 'SNG0638.json', 'SNG01407.json', 'SSNG0020.json', 'SSNG0151.json']

    sys_agent = 'model/save/template/oneHot_newReward_bitMore/0_2019-5-18-21-59-15-5-138-1.pkl'
    sys_agenr = 'model/save/nlg_sample/oneHot_newReward_bitMore/best/0_2019-5-19-15-13-5-6-139-1.pkl'
    sys_ageng = 'model/save/seq2seq/oneHot_newReward_bitMore/0_2019-5-19-22-28-16-6-139-1.pkl'
    sys_rnnt = 'model/save/sl_simulator/template/oneHot_oldReward_bitMore/0_2019-5-19-23-46-10-6-139-1.pkl'
    sys_rnnr = 'model/save/sl_simulator/retrieval/oneHot_oldReward_bitMore/best/0_2019-5-19-19-2-18-6-139-1.pkl'
    sys_rnng = 'model/save/sl_simulator/oneHot_oldReward_bitMore/best/0_2019-5-19-3-27-15-6-139-1.pkl'
    sys_gpt_sl = 'model/save/gpt_simulator/0_2021-9-28-20-28-2-1-271-0_42999.pkl'
    sys_gpt_must = 'model/save/gpt_simulator/0_2021-10-18-10-18-14-0-291-0_7999.pkl'
    sys_gpt_unif = 'model/save/gpt_simulator/best/0_2022-9-15-22-25-39-3-258-0_136999.pkl'
    sys_gpt_merfing = 'model/save/gpt_simulator/best/0_2022-9-12-9-50-3-0-255-0_43999.pkl'
    # sys_models = [sys_agent, sys_agenr, sys_ageng, sys_rnnt, sys_rnnr, sys_rnng, sys_gpt_sl, sys_gpt_must]
    sys_models = [sys_agent, sys_agenr, sys_rnnt, sys_gpt_sl, sys_gpt_must, sys_gpt_unif, sys_gpt_merfing]

    config.INTERACTIVE = True
    env = Enviroment(user=user, system=system, verbose=True, config=config)
    sys_idx = -1
    goal_idx = 49
    
    # interact(env, sys_models[sys_idx], goal_ids[goal_idx])

    # compute the results of human evaluations
    compute_succ_rate()