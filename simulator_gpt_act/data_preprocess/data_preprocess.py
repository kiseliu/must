# coding:utf-8
import enum
import os
import sys
from copy import deepcopy
from numpy import copy
root_path = os.path.abspath(__file__)
for _ in range(3):
    root_path = os.path.dirname(root_path)
sys.path.append(root_path)

import re
import json
import pickle
import pandas as pd
from pprint import pprint
from simulator_gpt_act.data_preprocess.utils import clean_goal, clean_text


slot_pattern = re.compile(r'[[].*?[]]', re.S)

restaurant_da_list = ["inform_type", "inform_type_change", "ask_info", "make_reservation", "make_reservation_change_time", 
                      "anything_else", "goodbye"]
info = ['name', 'area', 'food', 'pricerange']
book = ['people', 'day', 'time']
reqt = ['address', 'postcode', 'phone']

usr_act_path = 'data/multi-woz/rest_usr_simulator_act.json'
usr_path = 'data/multi-woz/rest_usr_simulator_goalkey.json'
sys_path = 'data/multi-woz/rest_sys.json'

usr_act_data = json.loads(open(usr_act_path).read())
usr_data = json.loads(open(usr_path).read())
sys_data = json.loads(open(sys_path).read())
assert len(usr_data) == len(sys_data) == len(usr_act_data)

usr_act_label = pd.read_csv('data/multi-woz/usr_act_label.csv')

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

            new_turn = {'sys': usr_turn['A']['transcript'].replace('\n', ''), 'sys_act':usr_turn['A']['slu'], 'goal': usr_turn['A']['goal'], 
                        'usr':usr_turn['B']['sent'].replace('\n', ''), 'usr_act':sys_turn['A']['slu'], 'usr_da':uda['B']['sent'].replace('\n', '')}
            # print('---'*30)
            # pprint(new_turn)
            dial.append(new_turn)
        dials.append(dial)
        # break

    # with open('data/multi-woz/mwz_restaurant.json', 'w', encoding='utf-8') as fw:
    #     json.dump(dials, fw, indent=2)
    return dials


def extract_uda_slots():

    def extract_slot(sent):
        tokens = re.findall(slot_pattern, sent)
        slots = []
        new_sent = sent
        for i, token in enumerate(tokens):
            if '[' in token and ']' in token:
                if '|' not in token:
                    new_token = token[1:-1].replace('*', '')
                else:
                    new_token = token[1:-1].split('|')[0].replace('*', '')
                slots.append(new_token)
                new_sent = new_sent.replace(token, '[' + new_token + ']')

        slots = [item.replace('value_', '') for item in slots]
        # print(sent, slots, new_sent)
        return new_sent, slots

    def sent_delex(sent, label_slots, dial_id):
        act = {}

        new_sent, slots = extract_slot(sent)
        if label_slots.lower() != 'no_label':
            intent = label_slots.lower().split('-')[0]
            all_slots = [slot.strip().replace('*', '').replace('value_', '') for slot in label_slots.split('-')[1:]]

            # print(dial_id, slots, all_slots)
            assert all_slots == slots

            while 'restaurant_name' in slots:
                idx = slots.index('restaurant_name')
                slots[idx] = 'name'
                new_sent = new_sent.replace('[restaurant_name]', '[value_name]')
            
            while 'restaurant_address' in slots:
                idx = slots.index('restaurant_address')
                slots[idx] = 'address'
                new_sent = new_sent.replace('[restaurant_address]', '[value_address]')

            if intent in ['make_reservation', 'make_reservation_change_time'] and 'count' in slots:
                tmp_sent = new_sent
                idx = slots.index('count')
                slots[idx] = 'people'
                new_sent = new_sent.replace('[value_count]', '[value_people]')
            
                if 'count' in slots:
                    print(dial_id, tmp_sent, '[value_count] of them' in new_sent)


            if intent in ['ask_info'] and 'count' in slots:
                # print(dial_id, new_sent, slots, label_slots)
                if 'recommend me [value_count]' in new_sent or 'recommend [value_count]' in new_sent or\
                    '[value_count] of those' in new_sent or '[value_count] of them' in new_sent or '[value_count] of the' in new_sent:
                    # print('---' * 30)
                    new_sent = new_sent.replace('[value_count]', 'one')
                    idx = slots.index('count')
                    slots.pop(idx)

            if intent in ['inform_type', 'inform_type_change'] and 'count' in slots:
                # print(dial_id, new_sent, slots, label_slots)
                if 'about [value_count]' in new_sent or ('how about' in new_sent and '[value_count]' in new_sent) or \
                    ('suggest' in new_sent and '[value_count]' in new_sent) or 'either [value_count]' in new_sent or \
                    'recommend [value_count]' in new_sent or 'find' in new_sent and '[value_count]' in new_sent or \
                    '[value_count] in the' in new_sent:
                    # print('---' * 30)
                    
                    new_sent = new_sent.replace('[value_count]', 'one')
                    idx = slots.index('count')
                    slots.pop(idx)
                    # print(dial_id, new_sent, slots, label_slots)               
                # elif :
                #     print(dial_id, new_sent, slots, label_slots)
                #     pass
                else:
                    print(dial_id, new_sent, slots, label_slots)
                    # pass

            act[intent] = slots

        else:
            if slots:
                intent = 'info_type'
                act[intent] = slots
                # print(sent)
                # print(new_sent, act, dial_id)
            else:
                # print(sent, label_slots, dial_id)
                pass
        return new_sent, act


    dial_slots = {}
    for items in usr_act_label.values:
        dial_id, char, sent, slots = items[1], items[2], items[3], items[4]
        
        if char == 1:
            continue

        if dial_id not in dial_slots:
            dial_slots[dial_id] = []
        delex_sent, act = sent_delex(sent, slots, dial_id)
        dial_slots[dial_id].append([delex_sent, act])
    return dial_slots
        

def merge_goal(dials):
    all_dial_slots = extract_uda_slots()

    goal_path = 'data/multi-woz/detailed_goals_augmented.pkl'
    detailed_goal = pickle.load(open(goal_path, 'rb'))
    detailed_goal_dict = {}
    for goal in detailed_goal:
        goal_id = goal['id']
        assert goal_id

        detailed_goal_dict[goal_id] = clean_goal(goal)


    dial_ids = []
    for id in usr_act_label['id']:
        if id not in dial_ids:
            dial_ids.append(id)

    assert len(dial_ids) == len(dials)

    new_dials = []
    for i, (dial_id, dial) in enumerate(zip(dial_ids, dials)):
        tmp = {}
        tmp['ids'] = dial_id
        tmp['goal'] = detailed_goal_dict[dial_id]

        dial_slots = all_dial_slots.get(dial_id)

        for j, turn in enumerate(dial):
            dial[j]['delex_usr'] = dial_slots[j][0]
            dial[j]['usr_act'] = dial_slots[j][1]

            uda = turn['usr_da'].lower()
            if uda not in turn.get('usr_act'):
                if 'ask_info' in turn.get('usr_act'):
                    slots = turn.get('usr_act').get('ask_info')
                    if slots:    # inform_type: slots
                        # print('====' * 30)
                        # print(dial_id, j, turn.get('usr_da'))
                        dial[j]['usr_act'] = {}
                        dial[j]['usr_act'][uda] = slots
                        # print(dial_id, j, turn)
                    else:        # ask_info: []
                        pass
                elif 'inform_type' in turn.get('usr_act'):
                    slots = turn.get('usr_act').get('inform_type')
                    if {'act': 'inform', 'slots':[['slot', 'nooffer']]} in turn.get('sys_act'):   # inform_type_change
                        # print('---' * 30)
                        # print(dial_id, j, turn)
                        dial[j]['usr_act'] = {}
                        dial[j]['usr_act']['inform_type_change'] = slots
                        # if not slots:
                        #     print('********')
                    elif uda == 'ask_info':   # inform_type
                        if slots:
                            dial[j]['usr_act'] = {}
                            dial[j]['usr_act']['inform_type'] = slots
                        dial[j]['usr_act']['ask_info'] = []
                    elif uda == 'make_reservation':   # make_reservation
                        dial[j]['usr_act'] = {}
                        dial[j]['usr_act'][uda] = slots
                elif 'inform_type_change' in turn.get('usr_act'):
                    slots = turn.get('usr_act').get('inform_type_change')
                    if 'what can i help you' in turn['sys'].lower():
                        dial[j]['usr_act'] = {}
                        dial[j]['usr_act']['inform_type'] = slots
                    elif uda == 'make_reservation':
                        book_slots = []
                        inform_slots = []
                        for slot in slots:
                            if slot in ['people', 'time', 'phone', 'day']:
                                book_slots.append(slot)
                            elif slot != 'restaurant_name':
                                inform_slots.append(slot)
                        dial[j]['usr_act'] = {}
                        if inform_slots:
                            dial[j]['usr_act']['inform_type'] = inform_slots
                        dial[j]['usr_act']['make_reservation'] = book_slots
                    else:
                        dial[j]['usr_act'] = {}
                        dial[j]['usr_act']['inform_type'] = inform_slots
                elif 'make_reservation' in turn.get('usr_act'):
                    pass
                elif 'make_reservation_change_time' in turn.get('usr_act'):
                    pass
                else:
                    pass

            
            if 'make_reservation' in turn.get('usr_act'):
                prev_turn = deepcopy(turn)
                slots = turn.get('usr_act').get('make_reservation')
                book_slots = []
                inform_slots = []
                for slot in slots:
                    if slot in ['people', 'time', 'phone', 'day']:
                        book_slots.append(slot)
                    elif slot != 'restaurant_name':
                        inform_slots.append(slot)
                
                # dial[j]['usr_act'] = {}
                if inform_slots and 'inform_type':
                    if 'inform_type' not in dial[j]['usr_act']:
                        dial[j]['usr_act']['inform_type'] = inform_slots
                    else:
                        dial[j]['usr_act']['inform_type'].extend(inform_slots)
                dial[j]['usr_act']['make_reservation'] = book_slots

                # if not prev_turn == turn:
                #     print('-------------' * 5)
                #     print(dial_id, j, prev_turn)
                #     print()
                #     print(dial_id, j, turn)
            
            if 'make_reservation_change_time' in turn.get('usr_act'):
                slots = turn.get('usr_act').get('make_reservation_change_time')
                book_slots = []
                inform_slots = []
                for slot in slots:
                    if slot in ['people', 'time', 'phone', 'day']:
                        book_slots.append(slot)
                    elif slot != 'restaurant_name':
                        inform_slots.append(slot)
                
                # dial[j]['usr_act'] = {}
                if inform_slots and 'inform_type':
                    if 'inform_type' not in dial[j]['usr_act']:
                        dial[j]['usr_act']['inform_type'] = inform_slots
                    else:
                        dial[j]['usr_act']['inform_type'].extend(inform_slots)
                dial[j]['usr_act']['make_reservation_change_time'] = book_slots
                    

        tmp['dials'] = dial
        new_dials.append(tmp)
        # pprint(tmp)
        # break

        # if dial_id == 'SSNG0181.json':
        #     pprint(tmp)

    new_dials = clean_text(new_dials)
    with open('data/experiment/mwz_restaurant_with_detailed_goal_clean_new.json', 'w', encoding='utf-8') as fw:
        json.dump(new_dials, fw, indent=2)
    return dials


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
            usr_intent = turn.get('usr_da').lower() if usr_act else ''

            act_list.append(sys_intent)
            act_list.append(usr_intent)

        print('act_list =', act_list)
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

    pprint(act_seq_list)

# extract_uda_slots()

dials = extract_dials(usr_act_data, usr_data, sys_data)
merge_goal(dials)

# path = 'data/mwz_restaurant_with_detailed_goal_clean_new.json'
# print_act_dist(path)
