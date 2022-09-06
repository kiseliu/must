# coding:utf-8
import sys
sys.path.append('./')
import json
import random
from copy import deepcopy
from pprint import pprint


at_at_path = 'evaluation_results/simulated_agenda_dataset/dials_at_at.json'
at_ar_path = 'evaluation_results/simulated_agenda_dataset/dials_ar_at.json'  # at_ar --> ar_at
ar_ar_path = 'evaluation_results/simulated_agenda_dataset/dials_ar_ar.json'
ag_ar_path = 'evaluation_results/simulated_agenda_dataset/dials_ar_ag.json'  # ag_ar --> ar_ag
ag_ag_path = 'evaluation_results/simulated_agenda_dataset/dials_ag_ag.json'

slot_mappings = {'postcode': 'post', 'address': 'addr', 'pricerange': 'price', 'phone': 'phone',
                 'food': 'food', 'name': 'name', 'area': 'area', 'reference': 'ref',
                 'people': 'people', 'day': 'day', 'time': 'time'}


def transform_goal(goal):
    new_goal = {}
    if 'cur_book' in goal:
        new_goal['book'] = list(goal['cur_book'].keys())

    if 'cur_info' in goal:
        new_goal['inform'] = list(goal['cur_info'].keys())

    if 'reqt' in goal:
        new_goal['request'] = goal['reqt']
    return new_goal


def delexicalize_sent(sent, da):
    assert 'act' in da
    assert 'parameters' in da

    usr_da = {}
    intent = da['act']
    svs = da['parameters']
    if intent:
        usr_da[intent] = []
        for s, v in svs.items():
            if s != 'goodbye':
                usr_da[intent].append(s)
            if v:
                sent = sent.replace(v, '[value_' + s + ']')
    return usr_da, sent


def extract_dials(dial):
    goal = dial.get('goal')
    dial_id = goal.get('id')

    success = dial.get('success')
    new_dial = {}
    # new_dial['ids'] = dial_id
    if success:
        # print('*****goal: ')
        # pprint(goal)

        info_set = {}
        turns = dial.get('turns')
        new_turns = []
        for i, turn in enumerate(turns):
            '''
            {   "sys": {"transcript": "Hello! What can I help you?",
                "goal": {"inform": ["name"], "book": ["time", "day", "people", "day"]}},
                "sys_act": {"sent": "I am looking for a particular restaurant. Its name is called travellers rest"},
                "usr": 0,
                "usr_act": },
            '''
            new_turn = {'sys': '', 'sys_act': [], 'goal': transform_goal(goal),
                        'usr_act': {}, 'delex_usr': '', 'usr': ''}

            # print('---' * 30)
            # print('usr_act: ', turn.get('usr_act'))
            # print('usr_utter: ', turn.get('usr_utter'))
            # print('sys_nlu: ', turn.get('sys_nlu'))
            # print('sys_act: ', turn.get('sys_act'))
            # print('sys_resp: ', turn.get('sys_resp'))

            new_turn['sys'] = turns[i].get('sys_resp') if i != 0 else "Hello! What can I help you?"
            if i > 0:
                sys_act = turns[i].get('sys_act')
                if sys_act:
                    dst_set = []
                    assert 'act' in sys_act
                    assert 'parameters' in sys_act
                    if sys_act['act'].strip().lower() == 'present_result':
                        for s, v in sys_act['parameters'].items():
                            if s in ["name", "food", "area", "pricerange"]:
                                info_set[s] = v

                    if sys_act['act'].strip().lower() == 'ask_type':
                        for s, v in sys_act['parameters'].items():
                            if s in ["name", "food", "area", "pricerange"]:
                                info_set[s] = v
                            else:
                                s = slot_mappings[s]
                                dst_set.append({'act': 'inform', 'slots': [[s, v]]})

                    for s, v in info_set.items():
                        s = slot_mappings[s]
                        dst_set.append({'act': 'inform', 'slots': [[s, v]]})

                    if sys_act['act'].strip().lower() in ['nomatch_result', 'no_other']:
                        dst_set.append({'act': 'inform', 'slots': [["slot", "nooffer"]]})

                    if sys_act['act'].strip().lower() == 'provide_info':
                        for s, v in sys_act['parameters'].items():
                            if s in ["name", "address", "phone", "postcode"]:
                                s = slot_mappings[s]
                                dst_set.append({'act': 'inform', 'slots': [[s, v]]})

                    ### booking
                    if sys_act['act'].strip().lower() in ['ask_reservation_info']:
                        for s, v in sys_act['parameters'].items():
                            s = slot_mappings[s]
                            dst_set.append({'act': 'request', 'slots': [['slot', s]]})

                    if sys_act['act'].strip().lower() == 'booking_success':
                        for s, v in sys_act['parameters'].items():
                            s = slot_mappings[s]
                            dst_set.append({'act': 'book_inform', 'slots': [[s, v]]})

                    if sys_act['act'].strip().lower() == 'booking_fail':
                        dst_set.append({'act': 'book_inform', 'slots': [["slot", "nobook"]]})

                    new_turn['sys_act'] = deepcopy(dst_set)

            uda = turn.get('usr_act')
            utt = turn.get('usr_utter')
            new_turn['usr_act'], new_turn['delex_usr'] = delexicalize_sent(utt, uda)
            new_turn['usr'] = turn.get('usr_utter')
            new_turns.append(new_turn)

        new_dial['ids'] = dial_id
        new_dial['goal'] = goal
        new_dial['dials'] = new_turns
        # pprint(new_dial)
    return new_dial


def extract_agenda_data(path):
    all_dials = []
    at_at_data = json.loads(open(at_at_path, 'r', encoding='utf-8').read())
    at_ar_data = json.loads(open(at_ar_path, 'r', encoding='utf-8').read())
    ar_ar_data = json.loads(open(ar_ar_path, 'r', encoding='utf-8').read())
    ag_ar_data = json.loads(open(ag_ar_path, 'r', encoding='utf-8').read())
    ag_ag_data = json.loads(open(ag_ag_path, 'r', encoding='utf-8').read())

    # list1 = set([dial.get('goal').get('id') for dial in at_at_data])
    # list2 = set([dial.get('goal').get('id') for dial in at_ar_data])
    # list3 = set([dial.get('goal').get('id') for dial in at_ar_data])
    # list4 = set([dial.get('goal').get('id') for dial in ag_ar_data])
    # list5 = set([dial.get('goal').get('id') for dial in ag_ar_data])
    # for i, j, k, l, m in zip(list1, list2, list3, list4, list5):
    #     print(i, '\t', j, '\t', k, '\t', l, '\t', m)
    print(len(at_ar_data))
    print(len(at_ar_data))
    print(len(ar_ar_data))
    print(len(ag_ar_data))
    print(len(ag_ag_data))
    for dial in at_at_data + ar_ar_data + ag_ag_data:
    # for dial in at_at_data:
        tmp = extract_dials(dial)
        # print(len(tmp))
        if tmp:
            all_dials.append(tmp)
            # pprint(tmp)

    random.shuffle(all_dials)

    print(len(all_dials))
    with open(path, 'w', encoding='utf-8') as fw:
        json.dump(all_dials, fw, indent=4)


def extract_gpt_il_data(path):
    mwz_data = json.loads(open('data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_mwz.json', 'r', encoding='utf-8').read())

    def extract(data):
        random.shuffle(data)

        all_dials = []
        for dial in data:
            tmp = extract_dials(dial)
            if tmp:
                all_dials.append(tmp)
            
            if len(all_dials) > 499:
                break
        return all_dials

    at_at_data = json.loads(open(at_at_path, 'r', encoding='utf-8').read())
    ar_ar_data = json.loads(open(ar_ar_path, 'r', encoding='utf-8').read())
    ag_ag_data = json.loads(open(ag_ag_path, 'r', encoding='utf-8').read())

    all_dials = mwz_data + extract(at_at_data) + extract(ar_ar_data) + extract(ag_ag_data)
    print(len(all_dials))
    random.shuffle(all_dials)

    with open(path, 'w', encoding='utf-8') as fw:
        json.dump(all_dials, fw, indent=4)

# extract_agenda_data('data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_agenda.json')
extract_gpt_il_data('data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_gpt_il.json')
