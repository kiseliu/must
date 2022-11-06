# coding:utf-8
from audioop import reverse
import json
from beeprint import pp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

corpus = 'data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_mwz.json'

def calc_agenda_dist(path):
    data = json.loads(open(path, 'r', encoding='utf-8').read())

    def calc_dist(dials):
        succ = dials['success']

        policy_dist = {}
        if succ:
            turns = dials['turns']

            for turn in turns:
                usr_act = turn['usr_act']
                # pp(usr_act)

                if 'act' in usr_act and usr_act['act']:
                    act_intent = usr_act['act']

                    policy_dist[act_intent] = policy_dist.get(act_intent, 0) + 1
                    # if act_intent not in policy_dist:
                    #     policy_dist[act_intent] = policy_dist.get(act_intent, {})

                    # for key in list(usr_act['parameters'].keys()):
                    #     policy_dist[act_intent][key] = policy_dist[act_intent].get(key, 0) + 1
        # print(policy_dist)
        return policy_dist

    user_policy_dist = {}
    for dials in data:
        policy_dist = calc_dist(dials)

        for k, v in policy_dist.items():
            user_policy_dist[k] = user_policy_dist.get(k, 0) + v

    pp(user_policy_dist)

    total_num = sum(list(user_policy_dist.values()))
    for k, v in user_policy_dist.items():
        print(k, v /total_num)


def calc_multiwoz_dist():
    data = json.loads(open(corpus, 'r', encoding='utf-8').read())
    # print(len(data))
    # pp(data[0])

    user_policy_dist = {}
    for dials in data:
        turns = dials['dials']

        for turn in turns:
            if 'usr_act' in turn and turn['usr_act']:
                usr_act = turn['usr_act']
                # pp(usr_act)

                for k, v in usr_act.items():
                    user_policy_dist[k] = user_policy_dist.get(k, 0) + 1

    print(user_policy_dist)
    total_num = sum(list(user_policy_dist.values()))
    for k, v in user_policy_dist.items():
        print(k, v /total_num)

def plot_policy_dist():
    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    spec = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)

    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15) 

    # plot 1:
    ax1 = fig.add_subplot(spec[0, 0])

    size = 7
    x = np.arange(size)

    a = [0.317, 0.030, 0.118, 0.101, 0.186, 0.016, 0.232]
    a.reverse()

    b = [0.369, 0.072, 0.185, 0.033, 0.095, 0.014, 0.232]
    b.reverse()

    kedu = ['inform type', 'inform type change', 'ask info', 'anything else',
            'make reservation', 'make reservation change time', 'goodbye']
    kedu.reverse()

    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 1

    print(x)
    print(width)
    print(x + width)
    # ax1.bar(x + 0.2, a, width=width, label='Agenda-based US')
    # ax1.bar(x + width + 0.2, b, width=width, label='Learning-based US', tick_label = kedu)
    ax1.barh(x + 0.2, a, height=width, label='ABUS')
    ax1.barh(x + width + 0.2, b, height=width, label='NUS', tick_label = kedu)


    # ax1.set_xticks(kedu) # 设置刻度
    # ax1.set_xticklabels(kedu, rotation = 15)
    ax1.legend(prop={'size': 15})
    
    # xpoints = [i * 2000 for i in list(range(1, num+1))]
    # ax1.plot(xpoints, succ_unif, color=uni_corlor, label='$\mathrm{MUST}_{\mathrm{uniform}}$')
    # ax1.plot(xpoints, succ_must, color=must_color, label='$\mathrm{MUST}_{\mathrm{adaptive}}$')
    # ax1.set_xlim(0, max_lim)

    # ax1.set_xticks(kedu) # 设置刻度
    # ax1.set_xticklabels(kedu, rotation = angle)

    # ax1.set_xlabel('The number of dialogues \n (1) The learning curves of the dialogue system S.', fontsize=font_size)
    # ax1.set_ylim(0, 100)
    # ax1.set_ylabel('The average success rate', fontsize=font_size)
    # ax1.legend(loc='lower right', prop={'size': 15})
    # plt.xticks(rotation=270)
    fig.tight_layout()
    plt.savefig('simulator_gpt_act/visualization/policy_dist.png', dpi=600)

# agent = calc_agenda_dist('evaluation_results/simulated_agenda_dataset/dials_at_at.json')
# agenr = calc_agenda_dist('evaluation_results/simulated_agenda_dataset/dials_ar_ar.json')
# ageng = calc_agenda_dist('evaluation_results/simulated_agenda_dataset/dials_ag_ag.json')

calc_multiwoz_dist()
plot_policy_dist()