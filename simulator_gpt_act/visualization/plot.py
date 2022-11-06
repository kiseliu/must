# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# kedu = [0, 40000, 80000, 120000, 160000, 200000]
# max_lim = 200000

kedu = [0, 20000, 40000, 60000, 80000, 100000]
max_lim = 100000
angle = 0
num = 50
font_size = 13

# uni_corlor = 'navajowhite'
uni_corlor = 'plum'
must_color = 'darkorange'

def read_csv(path):
    df = pd.read_csv(path)
    # print(df)
    return df

def accumulate():
    must = read_csv('simulator_gpt_act/visualization/succ_rates/must_sample_times.csv')
    # print(must)

    new_must = []
    value = must.values
    for line in value[:num]:
        line = [i/sum(list(line)) for i in list(line)]
        new_must.append(line)
        # print(line)
    agen_t = [line[0] for line in new_must]
    agen_r = [line[1] for line in new_must]
    rnn_t = [line[2] for line in new_must]
    gpt = [line[3] for line in new_must]
    return agen_t, agen_r, rnn_t, gpt

    # # US-AgenT,US-AgenR,US-RNNT,US-GPT-MWZ
    # def accum_sum(ls):
    #     ls = list(ls)
    #     assert len(ls) == 50

    #     first = ls[:20]
    #     middle = [ls[19] + i for i in ls[20:40]]
    #     last = [ls[39] + i for i in ls[40:]]
    #     return first + middle + last

    # agen_t = must['US-AgenT'][:num]
    # agen_r = must['US-AgenR'][:num]
    # rnn_t = must['US-RNNT'][:num]
    # gpt = must['US-GPT-MWZ'][:num]
    # return accum_sum(agen_t), accum_sum(agen_r), accum_sum(rnn_t), accum_sum(gpt)

def plot_whole():
    must = read_csv('model/save/gpt_simulator/old-4k-sd42/must/0_2022-9-11-14-42-59-6-254-0.csv')
    succ_must = [i*100 for i in must['success']][:num]

    uniform = read_csv('model/save/gpt_simulator/old-4k-sd42/uniform/0_2022-9-11-14-39-28-6-254-0.csv')
    succ_unif = [i*100 for i in uniform['success']][:num]

    fig = plt.figure(constrained_layout=True, figsize=(5, 10))
    spec = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[1, 1])

    # plot 1:
    ax1 = fig.add_subplot(spec[0, 0])
    
    xpoints = [i * 2000 for i in list(range(1, num+1))]
    ax1.plot(xpoints, succ_unif, color=uni_corlor, label='$\mathrm{MUST}_{\mathrm{uniform}}$')
    ax1.plot(xpoints, succ_must, color=must_color, label='$\mathrm{MUST}_{\mathrm{adaptive}}$')
    ax1.set_xlim(0, max_lim)

    ax1.set_xticks(kedu) # 设置刻度
    ax1.set_xticklabels(kedu, rotation = angle)

    ax1.set_xlabel('The number of dialogues \n (1) The learning curves of the dialogue system S.', fontsize=font_size)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('The average success rate', fontsize=font_size)
    ax1.legend(loc='lower right', prop={'size': 15})

    # plot 2:
    # {'US-AgenT': 10108, 'US-AgenR': 9889, 'US-RNNT': 10030, 'US-GPT-MWZ': 9972} ~40k
    # {'US-AgenT': 8103, 'US-AgenR': 18030, 'US-RNNT': 6513, 'US-GPT-MWZ': 7354}  40k~80k
    # {'US-AgenT': 3913, 'US-AgenR': 6939, 'US-RNNT': 4239, 'US-GPT-MWZ': 4909}   80k~100k

    sample_num = {'US-AgenT': 8103, 'US-RNNT': 6513, 
                  'US-AgenR': 18030, 'US-GPT-MWZ': 7354}
    total = sum(list(sample_num.values()))
    values = [sample_num.get('US-AgenT')/total, sample_num.get('US-AgenR')/total, 
              sample_num.get('US-RNNT')/total, sample_num.get('US-GPT-MWZ')/total]
    labels = ['AgenT', 'AgenR', 'RNNT', '$\mathrm{GPT}_{\mathrm{SL}}$'] 

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            # 同时显示数值和占比的饼图
            return '{p:.1f}%'.format(p=pct)
        return my_autopct

    ax2 = fig.add_subplot(spec[1, 0])
    ax2.pie(values, labels=labels, autopct=make_autopct(values))
    ax2.set_xlabel('(2) The proportion of each user \n simulator that has been sampled.', fontsize=font_size)

    # agen_t, agen_r, rnn_t, gpt = accumulate()

    # ax2 = fig.add_subplot(spec[1, 0])
    # xpoints = [i * 2000 for i in list(range(1, num+1))]
    # ax2.plot(xpoints, agen_t, color=uni_corlor, label='AgenT')
    # ax2.plot(xpoints, agen_r, color='r', label='AgenR')
    # ax2.plot(xpoints, rnn_t, color='g', label='RNNT')
    # ax2.plot(xpoints, gpt, color='b', label='$GPT_{SL}$')

    # ax2.set_xlim(0, max_lim)
    # ax2.set_xticks(kedu) # 设置刻度
    # ax2.set_xticklabels(kedu, rotation = angle)
    # ax1.set_xlabel('The number of dialogues \n 1) The learning curves of dialogue system S.', fontsize=font_size)

    # ax2.set_ylim(0, 1)
    # ax1.set_ylabel('The average success rate', fontsize=font_size)
    # ax2.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig('simulator_gpt_act/visualization/pie.png', dpi=800)


def extract_sys_succ(us):
    must = read_csv('simulator_gpt_act/visualization/succ_rates/must.csv')
    uniform = read_csv('simulator_gpt_act/visualization/succ_rates/uniform.csv')
    return [i*100 for i in must[us]], [i*100 for i in uniform[us]]

def plot_subfigures():
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # plot 1:
    ax1 = fig.add_subplot(spec[0, 0])

    xpoints = [i * 2000 for i in list(range(1, num+1))]
    y_must, y_uniform = extract_sys_succ('US-AgenT')
    ax1.plot(xpoints, y_uniform[:num], color=uni_corlor, label='$\mathrm{MUST}_{\mathrm{uniform}}$')
    ax1.plot(xpoints, y_must[:num], color=must_color, label='$\mathrm{MUST}_{\mathrm{adaptive}}$')
    ax1.set_xlim(0, max_lim)

    ax1.set_xticks(kedu) # 设置刻度
    ax1.set_xticklabels(kedu, rotation = angle)

    ax1.set_xlabel('The number of dialogues \n (1) The success rate of dialogue system S \n performing with AgenT.', fontsize=font_size)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('The success rate', fontsize=font_size)
    ax1.legend(loc='lower right', prop={'size': 15})

    # plot 2:
    ax2 = fig.add_subplot(spec[0, 1])

    xpoints = [i * 2000 for i in list(range(1, num+1))]
    y_must, y_uniform = extract_sys_succ('US-AgenR')
    ax2.plot(xpoints, y_uniform[:num], color=uni_corlor, label='$\mathrm{MUST}_{\mathrm{uniform}}$')
    ax2.plot(xpoints, y_must[:num], color=must_color, label='$\mathrm{MUST}_{\mathrm{adaptive}}$')
    ax2.set_xlim(0, max_lim)

    ax2.set_xticks(kedu) # 设置刻度
    ax2.set_xticklabels(kedu, rotation = angle)

    ax2.set_xlabel('The number of dialogues \n (2) The success rate of dialogue system S \n performing with AgenR.', fontsize=font_size)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('The success rate', fontsize=font_size)
    ax2.legend(loc='lower right', prop={'size': 15})

    # plot 3:
    ax3 = fig.add_subplot(spec[1, 0])
    
    xpoints = [i * 2000 for i in list(range(1, num+1))]
    y_must, y_uniform = extract_sys_succ('US-RNNT')
    ax3.plot(xpoints, y_uniform[:num], color=uni_corlor, label='$\mathrm{MUST}_{\mathrm{uniform}}$')
    ax3.plot(xpoints, y_must[:num], color=must_color, label='$\mathrm{MUST}_{\mathrm{adaptive}}$')
    ax3.set_xlim(0, max_lim)

    ax3.set_xticks(kedu) # 设置刻度
    ax3.set_xticklabels(kedu, rotation = angle)

    ax3.set_xlabel('The number of dialogues \n (3) The success rate of dialogue system S \n performing with RNNT.', fontsize=font_size)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('The success rate', fontsize=font_size)
    ax3.legend(loc='lower right', prop={'size': 15})

    # plot 4:
    ax4 = fig.add_subplot(spec[1, 1])
    
    xpoints = [i * 2000 for i in list(range(1, num+1))]
    y_must, y_uniform = extract_sys_succ('US-GPT-MWZ')
    ax4.plot(xpoints, y_uniform[:num], color=uni_corlor, label='$\mathrm{MUST}_{\mathrm{uniform}}$')
    ax4.plot(xpoints, y_must[:num], color=must_color, label='$\mathrm{MUST}_{\mathrm{adaptive}}$')
    ax4.set_xlim(0, max_lim)

    ax4.set_xticks(kedu) # 设置刻度
    ax4.set_xticklabels(kedu, rotation = angle)

    ax4.set_xlabel('The number of dialogues \n (4) The success rate of dialogue system S \n performing with $\mathrm{GPT}_{\mathrm{SL}}$.', fontsize=font_size)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel('The success rate', fontsize=font_size)
    ax4.legend(loc='lower right', prop={'size': 15})

    fig.tight_layout()
    plt.savefig('simulator_gpt_act/visualization/right.png', dpi=800)

# values = [3, 12, 5, 8] 
# labels = ['a', 'b', 'c', 'd'] 

# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         # 同时显示数值和占比的饼图
#         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#     return my_autopct

# plt.pie(values, labels=labels, autopct=make_autopct(values))
# plt.show()

plot_whole()
plot_subfigures()

# accumulate()