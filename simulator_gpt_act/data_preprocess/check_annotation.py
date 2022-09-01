# coding:utf-8
import json
import random
from pprint import pprint

path = 'data/experiment/mwz_restaurant_with_detailed_goal_clean_new.json'
new_path = 'data/experiment/mwz_restaurant_with_annatated_da.json'
# agenda_path = 'data/experiment/agenda_200/rest_usr_simulator_goal_agenda.json'
agenda_path = 'data/experiment/agenda_new/rest_usr_simulator_goal_agenda_new.json'

def check_anything_else(path):
    data = json.loads(open(path, 'r').read())
    for dial in data:
        log = dial['dials']
        for turn in log:
            if 'anything_else' in turn['usr_act']:
                utt = turn['usr']
                if 'else' in utt:
                    pass
                elif 'another' in utt or 'other' in utt:
                    pass
                elif 'different' in utt or 'something':
                    pass
                else:
                    print(utt)

def check():
    data = open('data/multiwoz-master/data/multi-woz/usr_act_label.csv').read().split('\n')
    for line in data:
        if 'anything_else' in line.lower():
            items = line.strip().split(',')
            utt = items[3]
            if 'anything else' in utt or 'something else' in utt:
                # print(utt)
                pass
            elif 'else' in utt:
                # print(utt)
                pass
            elif 'any other' in utt or 'other' in utt or 'another' in utt:
                print(utt)
                pass
            elif 'different' in utt:
                # print(items[0], utt)
                pass
            else:
                pass
                # print(items[0], utt)


def extract_act_seq_list(data, condition):
    total = {}
    for dial in data:
        dial_id = dial['ids']
        log = dial['dials']
        for turn in log:
            # delex_usr = turn.get('delex_usr')
            uda = turn.get('usr_act')

            for k, v in uda.items():
                if k not in total:
                    total[k] = []
                total[k].append(dial_id)
    
    all_dials = total.get(condition)
    idx = random.randint(0, len(all_dials))
    pprint(data[idx])

# check_anything_else(path)
# check()

data = json.loads(open(new_path, 'r', encoding='utf-8').read())
agenda = json.loads(open(agenda_path, 'r', encoding='utf-8').read())
# extract_act_seq_list(data, 'anything_else')

def get_dial_idx(data):
    dial_idx_dict = {}
    for i, dial in enumerate(data):
        dial_id = dial['ids']
        dial_idx_dict[dial_id] = i
    return dial_idx_dict

def pprint_dial(dial):
    print(dial['ids'])
    dials = dial['dials']
    for turn in dials:
        print('---' * 30)
        print('goal: ', turn['goal'])
        print('sys : ', turn['sys'])
        print('sys_act : ', turn['sys_act'])
        print('usr_act : ', turn['usr_act'])
        print('usr : ', turn['usr'])
        print('delex_usr : ', turn['delex_usr'])


def compare_dials(mwz, agenda):
    mwz_idx_dict = get_dial_idx(mwz)
    agen_idx_dict = get_dial_idx(agenda)

    inter_set = list(set(list(mwz_idx_dict.keys())).intersection(set(list(agen_idx_dict.keys()))))

    idx = random.randint(0, len(inter_set))
    dial_id = inter_set[idx]

    mwz_idx = mwz_idx_dict[dial_id]
    pprint_dial(mwz[mwz_idx])
    print('*****' * 50)
    age_idx = agen_idx_dict[dial_id]
    pprint_dial(agenda[age_idx])


def merge_dials(mwz, agenda):
    new_dials = []
    new_dials.extend(mwz)
    new_dials.extend(agenda)
    random.shuffle(new_dials)

    print(len(new_dials))
    with open('data/experiment/mwz_restaurant_mix_new.json', 'w', encoding='utf-8') as fw:
        json.dump(new_dials, fw, indent=2)


# compare_dials(data, agenda)
# merge_dials(data, agenda)