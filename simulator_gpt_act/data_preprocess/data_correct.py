# coding:utf-8
import os
import sys
root_path = os.path.abspath(__file__)
for _ in range(3):
    root_path = os.path.dirname(root_path)
sys.path.append(root_path)

import re
import json
import pandas as pd
from pprint import pprint
from simulator_gpt_act.data_preprocess.utils import clean_text
from simulator_gpt_act.data_preprocess import utils


restaurant_da_list = ["inform_type", "inform_type_change", 
                      "ask_info", 
                      "make_reservation", "make_reservation_change_time", 
                      "anything_else", 
                      "goodbye"]
info = ['name', 'area', 'food', 'pricerange']
book = ['people', 'day', 'time']
reqt = ['address', 'postcode', 'phone']

usr_act_path = 'data/multiwoz-master/data/multi-woz/rest_usr_simulator_act.json'
usr_path = 'data/multiwoz-master/data/multi-woz/rest_usr_simulator_goalkey.json'
sys_path = 'data/multiwoz-master/data/multi-woz/rest_sys.json'

usr_act_data = json.loads(open(usr_act_path).read())
usr_data = json.loads(open(usr_path).read())
sys_data = json.loads(open(sys_path).read())
assert len(usr_data) == len(sys_data) == len(usr_act_data)

usr_act_label = pd.read_csv('data/multiwoz-master/data/multi-woz/usr_act_label.csv')

def extract_dials(usr_act_data, usr_data, sys_data):
    dials = []
    for _, (usr_act, usr, sys) in enumerate(zip(usr_act_data, usr_data, sys_data)):
        usr_dial = usr.get('dial')
        sys_dial = sys.get('dial')
        uda_dial = usr_act.get('dial')
        dial = []
        for _, (uda, usr_turn, sys_turn) in enumerate(zip(uda_dial, usr_dial, sys_dial)):
            # pprint(usr_turn)
            # pprint(sys_turn)
            # pprint(uda)
            # print('---'*30)

            new_turn = {'sys': usr_turn['A']['transcript'].replace('\n', ''), 
                        'sys_act': usr_turn['A']['slu'], 
                        'goal': usr_turn['A']['goal'], 
                        'usr_da_intent': uda['B']['sent'].replace('\n', '').lower(),
                        'usr': usr_turn['B']['sent'].replace('\n', '')}
            # print('---'*30)
            # pprint(new_turn)
            dial.append(new_turn)
        dials.append(dial)
        # break

    with open('data/multiwoz-master/data/multi-woz/mwz_restaurant.json', 'w', encoding='utf-8') as fw:
        json.dump(dials, fw, indent=2)
    return dials


def extract_uda_slots():
    dial_slots = {}
    # keep slot consistent
    for items in usr_act_label.values:
        dial_id, char, sent, label_slots = items[1], items[2], items[3], items[4]
        
        if char == 1:
            continue

        if dial_id not in dial_slots:
            dial_slots[dial_id] = []
        
        new_sent = sent.strip().lower()
        if label_slots.lower() != 'no_label':
            # process restaurant_name and restaurant_address in the usr_act_label.csv file
            new_sent, intent, slots = utils.process_rest_slot(dial_id, new_sent, label_slots)
            assert 'restaurant_name' not in new_sent and 'restaurant_address' not in new_sent
            # print(new_sent, intent, slots)
        else:
            # print(sent)
            intent, slots = '', []
        dial_slots[dial_id].append([new_sent, intent, slots])

    dials = extract_dials(usr_act_data, usr_data, sys_data)
    assert len(dials) == len(dial_slots)
    new_dials = utils.add_dial_id(dials, usr_act_label)
    new_dials = utils.correct_uda(new_dials, dial_slots)

    new_dials = clean_text(new_dials)
    with open('data/multiwoz-master/data/multi-woz/mwz_restaurant_with_annatated_da.json', 'w', encoding='utf-8') as fw:
        json.dump(new_dials, fw, indent=2)
    return new_dials
        

# all_dial_slots = extract_uda_slots()


def print_act_dist(path):
    act_seq_dict = {}
    act_seq_list = {}

    data = json.loads(open(path, 'r', encoding='utf-8').read())
    for line in data:
        dials = line.get('dials')

        act_list = []
        for turn in dials:
            sys_act = turn.get('sys_act')
            sys_intent = ' '.join(list(set([act.get('act') for act in sys_act]))).lower() if sys_act else ''

            usr_act = turn.get('usr_act')
            usr_intent = str(list(usr_act.keys())) if usr_act else ''

            # act_list.append(sys_intent)
            act_list.append(usr_intent)

        if not act_list:
            continue

        # print('act_list =', act_list)
        if ' '.join(act_list) not in act_seq_dict:
            act_seq_dict[' '.join(act_list)] = 0
            act_seq_list[' '.join(act_list)] = []
        act_seq_dict[' '.join(act_list)] += 1
        act_seq_list[' '.join(act_list)].append(line.get('ids'))
            # uda = turn.get('usr_da').strip().lower()
            # if uda in ['inform_type', 'inform_type_change']:
            #     if 'request' in usr_act:
            #         pprint(dials)
            
            # if uda in ['ask_info']:
            #     if 'inform' in usr_act:
            #         print(line.get('ids'))
            #         pprint(dials)
            # print(uda, usr_act)
        # print(act_list[-1])
        # if act_list[-1] in ["['inform_type']", "['inform_type_change']"]:
        #     print(line.get('ids'))
    # pprint(act_seq_dict)
    # pprint(act_seq_list)

# print_act_dist('data/multiwoz-master/data/multi-woz/mwz_restaurant_with_annatated_da.json')
