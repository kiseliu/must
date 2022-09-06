# coding:utf-8
import re
import copy
import pickle
from pprint import pprint

slot_pattern = re.compile(r'[[].*?[]]', re.S)
info = ['name', 'area', 'food', 'pricerange']
book = ['people', 'day', 'time']
reqt = ['address', 'postcode', 'phone']
anything_else_phrases = ['other', 'else', 'different']

def remove_space(sentence):
    return ' '.join(sentence.lower().strip().split())

def confirm_slot_type(utt, slots):
    info_flag = []
    book_flag = []
    reqt_flag = []

    for s in slots:
        if s in info:
            info_flag.append(s)
        if s in book:
            book_flag.append(s)
        if s in reqt:
            reqt_flag.append(s)

    for token in reqt:
        if token in utt:
            reqt_flag.append(token)
    if 'Phone' in utt:
        reqt_flag.append('phone')
    if 'reference' in utt and 'preference' not in utt:
        reqt_flag.append('reference')
    if 'reservation number' in utt:
        reqt_flag.append('reference')
    if 'post code' in utt:
        reqt_flag.append('postcode')
    if 'some information on the place' in utt:
        reqt_flag.append('address')
    return info_flag, book_flag, reqt_flag

def is_nooffer_or_nobook(dial_id, turn):
    sys = turn['sys']
    sys_act = turn['sys_act']
    nooffer = False
    nobook = False

    for act in sys_act:
        if act['act'] == 'inform' and act['slots']:
            slots = act['slots']
            for slot in slots:
                if slot[1] == 'nooffer':
                    nooffer = True
        elif act['act'] == 'book_inform' and act['slots']:
            slots = act['slots']
            for slot in slots:
                if slot[1] == 'nobook':
                    nobook = True
    return nooffer, nobook

def add_dial_id(dials, usr_act_label):
    goal_path = 'data/multiwoz-master/data/multi-woz/goal/detailed_goals_augmented.pkl'
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

    assert len(dial_ids) == len(dials) == len(detailed_goal_dict)
    new_dials = {}
    for i, (dial_id, dial) in enumerate(zip(dial_ids, dials)):
        tmp = {}
        tmp['ids'] = dial_id
        tmp['goal'] = detailed_goal_dict[dial_id]
        tmp['dials'] = dial
        new_dials[dial_id] = tmp
        # pprint(tmp)
        # break
    return new_dials

def process_rest_slot(dial_id, new_sent, label_slots):
    intent = label_slots.lower().split('-')[0]
    slots = label_slots.lower().split('-')[1:]

    for idx, slot in enumerate(slots):
        assert slot.startswith('restaurant_') or slot.startswith('value_')
        if 'restaurant_name' in slot:
            slots[idx] = 'value_name'
            new_sent = new_sent.replace(slot, 'value_name')
                
        if 'restaurant_address' in slot:
            slots[idx] = 'value_address'
            new_sent = new_sent.replace(slot, 'value_address')
    return new_sent, intent, slots

def extract_slot(sent):
    tokens = re.findall(slot_pattern, sent)
    slots = []
    for i, token in enumerate(tokens):
        if '[' in token and ']' in token:
            if '|' not in token:
                new_token = token[1:-1]
            else:
                new_token = token[1:-1].split('|')[0]
            slots.append(new_token)
    return slots

def process_slot_count(dial_id, utt, utt_delex, slots):
    extracted_slots = extract_slot(utt_delex)
    assert extracted_slots == slots
    for slot in slots:
        assert slot.startswith('value_')

    tokens = re.findall(slot_pattern, utt_delex)
    assert len(tokens) == len(slots)

    new_slots = []
    for i, token in enumerate(tokens):
        if '[' in token and ']' in token:
            placeholder = slots[i].replace('*', '')
            utt_delex = utt_delex.replace(token[1:-1], placeholder)
            new_slots.append(placeholder.split('_')[1])

    if 'count' in new_slots or 'value_count' in utt_delex:
        def replace_span(utt_delex, new_slots):
            flag = False
            # print('\nold = ', utt, new_slots, dial_id)
            if '[value_count] people' in utt_delex:
                utt_delex = utt_delex.replace('[value_count] people', '[value_people] people')
                flag = True
            elif '[value_count] person' in utt_delex:
                utt_delex = utt_delex.replace('[value_count] person', '[value_people] person')
                flag = True
            elif 'a table for [value_count]' in utt_delex:
                utt_delex = utt_delex.replace('a table for [value_count]', 'a table for [value_people]')
                flag = True
            elif 'a reservation for [value_count]' in utt_delex:
                utt_delex = utt_delex.replace('a reservation for [value_count]', 'a reservation for [value_people]')
                flag = True
            elif '[value_count] of us' in utt_delex:
                utt_delex = utt_delex.replace('[value_count] of us', '[value_people] of us')
                flag = True
            elif 'a party of [value_count]' in utt_delex:
                utt_delex = utt_delex.replace('a party of [value_count]', 'a party of [value_people]')
                flag = True
            elif 'book for [value_count]' in utt_delex:
                utt_delex = utt_delex.replace('book for [value_count]', 'book for [value_people]')
                flag = True
            if flag:
                idx = new_slots.index('count')
                new_slots[idx] = 'people'
            # print('new = ', utt_delex, new_slots)
            return utt_delex, new_slots, flag

        def delete_count(utt_delex, new_slots):
            flag = False
            # print('\nold = ', utt, new_slots)
            if 'recommend me [value_count]' in utt_delex or 'recommend [value_count]' in utt_delex:
                utt_delex = utt_delex.replace('[value_count]', 'one')
                flag = True
            elif '[value_count] of those' in utt_delex or '[value_count] of them' in utt_delex or '[value_count] of the' in utt_delex:
                utt_delex = utt_delex.replace('[value_count]', 'one')
                flag = True
            elif 'about [value_count]' in utt_delex or ('how about' in utt_delex and '[value_count]' in utt_delex) or \
                ('suggest' in utt_delex and '[value_count]' in utt_delex) or 'either [value_count]' in utt_delex or \
                ('find' in utt_delex and '[value_count]' in utt_delex):
                utt_delex = utt_delex.replace('[value_count]', 'one')
                flag = True
            if flag:
                idx = new_slots.index('count')
                new_slots.pop(idx)
            # print('new = ', utt_delex, new_slots)
            return utt_delex, new_slots, flag
            
        
        utt_delex, new_slots, r_flag = replace_span(utt_delex, new_slots)
        utt_delex, new_slots, d_flag = delete_count(utt_delex, new_slots)
        # if not r_flag and not d_flag:
        #     print('\nold = ', utt, new_slots, dial_id)
        #     print('new = ', utt_delex, new_slots)
    # else:
    #     print('\n', utt_delex, new_slots)
    return utt_delex, new_slots


def process_da_intent(dial_id, turn, ext_intent, slots, utt_delex):
    correct_uda = {}
    labeled_intent = turn['usr_da_intent']
    utt = turn['usr']

    nooffer, nobook = is_nooffer_or_nobook(dial_id, turn)
    info_slots, book_slots, reqt_slots = confirm_slot_type(utt, slots)
    info_flag = True if info_slots else False
    book_flag = True if book_slots else False 
    reqt_flag = True if reqt_slots else False
    if nooffer:
        if 'inform_type_change' in [labeled_intent, ext_intent]:
            if labeled_intent == ext_intent:
                assert labeled_intent == ext_intent == 'inform_type_change'
                assert not book_flag and info_flag  # all info_slots are not empty
                # only SNG0577.json that s okay . let s try [value_food] food instead , with the other details the same .

                if dial_id in ['WOZ20249.json']:
                    # Okay what is the phone number and postcode of that mediterranean place?
                    correct_uda['inform_type'] = info_slots
                else:
                    correct_uda['inform_type_change'] = info_slots
                    if 'how about' in utt_delex or 'what about' in utt_delex or \
                        'is there anything' in utt_delex or 'is there any' in utt_delex:
                        pass
                if reqt_flag:
                    correct_uda['ask_info'] = reqt_slots
                # print('---' * 30)    
                # print(turn['sys'], '\n')
                # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                # print(dial_id, labeled_intent, slots, '\t', correct_uda)
            else:
                if dial_id in ['SNG01513.json']:
                    # where else could ypou suppose i get american food 
                    correct_uda['anything_else'] = info_slots
                else:
                    if not book_flag and not reqt_flag:
                        # how about
                        correct_uda['inform_type_change'] = info_slots
                    elif dial_id in ['SSNG0162.json']:
                        # Let's try expensive Italian food. I need a table for 6 on wednesday at 18:45, can you help me?
                        correct_uda['inform_type_change'] = ['pricerange', 'food']
                        correct_uda['make_reservation'] = ['people', 'time', 'day']
                    else:
                        # Indian food of course and what is the phone number? 
                        correct_uda['inform_type_change'] = info_slots
                        correct_uda['ask_info'] = reqt_slots
                        # print('---' * 30)    
                        # print(turn['sys'], '\n')
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                        # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
        else:
            if labeled_intent == ext_intent:
                if labeled_intent == 'inform_type':
                    assert info_flag == (True if slots else False)
                    assert not book_flag and not reqt_flag                        

                    if dial_id not in ['SNG01235.json', 'SNG02239.json', 'SNG0505.json', 'SNG0556.json',
                        'SNG0734.json', 'SSNG0153.json', 'SSNG0175.json', 'WOZ20033.json', 'WOZ20620.json']:
                        correct_uda['inform_type_change'] = info_slots
                    else: 
                        # do not change
                        correct_uda['inform_type'] = info_slots
                        # print('---' * 30)    
                        # print(turn['sys'], '\n')
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                        # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                elif labeled_intent == 'anything_else':
                    assert not book_flag and not reqt_flag

                    if dial_id in ['SSNG0022.json', 'SSNG0067.json']:
                        correct_uda['inform_type_change'] = info_slots
                    else:
                        correct_uda['anything_else'] = info_slots
                    # print('---' * 30) 
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                elif labeled_intent == 'make_reservation':
                    if dial_id in ['SNG0542.json']:
                        # Well, can you recommend me another expensive restaurant. I need the address and the phone number to make a reservation.
                        correct_uda['anything_else'] = info_slots
                        correct_uda['ask_info'] = reqt_slots
                    elif dial_id in ['SSNG0164.json']:
                        # I'll take any type of cuisine, then. I just need a reservation for 6 at 13:15 on Saturday.
                        # Okay I guess. Try booking there please.
                        assert not reqt_flag
                        correct_uda['inform_type_change'] = []
                        if book_flag:
                            correct_uda['make_reservation'] = book_slots
                        else:
                            correct_uda['make_reservation'] = []
                    else:
                        #  Then go ahead and book La Raza for me. We have 4 people at 17:00 on Monday.
                        correct_uda['inform_type_change'] = ['name']
                        correct_uda['make_reservation'] = ['people', 'time', 'day']
                        # print('---' * 30)    
                        # print(turn['sys'], '\n')
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                        # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                elif labeled_intent == 'ask_info':
                    assert not book_flag
                    if dial_id in ['SNG0503.json']:
                        correct_uda['inform_type'] = ['food']
                        correct_uda['ask_info'] = ['area']
                    elif dial_id in ['SSNG0117.json']:
                        correct_uda['ask_info'] = info_slots
                        correct_uda['inform_type_change'] = ['area']
                    elif dial_id in ['WOZ20062.json', 'WOZ20507.json', 'WOZ20515.json']: 
                        correct_uda['inform_type_change'] = info_slots
                    if reqt_flag:
                        correct_uda['ask_info'] = reqt_slots
                    # print('---' * 30)    
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                else:
                    assert not slots
                    correct_uda['goodbye'] = []
            else: 
                assert not book_flag and not reqt_flag
                if dial_id in ['SNG0507.json']:
                    # Well, the restaurant should be expensive. Are there any expensive restaurants in the centre of town?
                    correct_uda['anything_else'] = info_slots
                    correct_uda['inform_type'] = info_slots
                else:
                    correct_uda['inform_type_change'] = info_slots
                # print('---' * 30)    
                # print(turn['sys'], '\n')
                # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
    elif nobook:
        if 'make_reservation_change_time' in [labeled_intent, ext_intent]:
            if labeled_intent == ext_intent:  
                assert labeled_intent == ext_intent == 'make_reservation_change_time'
                if book_flag:
                    correct_uda['make_reservation_change_time'] = book_slots
                    if info_flag:
                        assert not reqt_flag
                        correct_uda['inform_type'] = info_slots
                    elif reqt_flag:
                        correct_uda['ask_info'] = reqt_slots   
                        # print('---' * 30)    
                        # print(turn['sys'], '\n')
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                        # print(dial_id, labeled_intent, slots, '\t', correct_uda)
                else:
                    assert not info_flag and not reqt_flag
                    correct_uda['make_reservation_change_time'] = []
                    # print('---' * 30)    
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, slots, '\t', correct_uda)
            else:
                assert not info_flag and not reqt_flag
                correct_uda['make_reservation_change_time'] = book_slots
                # print('---' * 30)    
                # print(turn['sys'])
                # print(turn['sys_act'], '\n')
                # print('utt = ', utt_delex, '\t', utt)
                # print(dial_id, labeled_intent, ext_intent, slots)
                # print(correct_uda, '\t', info_flag, book_flag, reqt_flag)
        else:
            if labeled_intent == ext_intent:
                if labeled_intent == 'make_reservation':
                    if dial_id in ['SSNG0163.json']:
                        correct_uda['anything_else'] = ['time']
                    elif dial_id in ['SNG01776.json', 'SNG02052.json', 'SNG02321.json', 'SNG0594.json']:
                        assert not info_flag
                        correct_uda['make_reservation_change_time'] = book_slots
                        if reqt_flag:
                            correct_uda['ask_info'] = reqt_slots
                        # print('---' * 30)    
                        # print(turn['sys'])
                        # print(turn['sys_act'], '\n')
                        # print('utt = ', utt_delex, '\t', utt)
                        # print(dial_id, labeled_intent, ext_intent, slots)
                        # print(correct_uda, '\t', info_flag, book_flag, reqt_flag)
                    else:
                        correct_uda['make_reservation'] = book_slots
                        if not book_flag:
                            if dial_id in ['SSNG0004.json']:
                                correct_uda['ask_info'] = reqt_slots
                            elif dial_id in ['SSNG0134.json']:
                                pass
                            else:
                                correct_uda['anything_else'] = info_slots
                                if reqt_flag:
                                    correct_uda['ask_info'] = reqt_slots
                        else:  
                            # assert info_flag
                            if info_flag:
                                correct_uda['inform_type_change'] = info_slots
                            if reqt_flag:
                                correct_uda['ask_info'] = reqt_slots
                            # print('---' * 30)    
                            # print(turn['sys'])
                            # print(turn['sys_act'], '\n')
                            # print('utt = ', utt_delex, '\t', utt)
                            # print(dial_id, labeled_intent, ext_intent, slots)
                            # print(correct_uda, '\t', info_flag, book_flag, reqt_flag)
                elif labeled_intent == 'inform_type':
                    assert not book_flag and not reqt_flag
                    if dial_id in ['SSNG0011.json', 'SSNG0023.json', 'SSNG0033.json', 'SSNG0034.json',
                                  'SSNG0046.json', 'SSNG0100.json', 'SSNG0105.json', 'SSNG0108.json', 'SSNG0126.json',
                                  'SSNG0152.json', 'SSNG0168.json', 'SSNG0183.json']:
                        correct_uda['anything_else'] = info_slots
                        # print(turn['sys'], '\n')
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    elif info_flag:
                        correct_uda['inform_type_change'] = info_slots
                    else:
                        correct_uda['inform_type'] = []
                elif labeled_intent == 'inform_type_change':
                    assert not book_flag and not reqt_flag
                    if dial_id in ['SSNG0025.json', 'SSNG0114.json', 'SSNG0168.json', 'SSNG0196.json']:
                        correct_uda['anything_else'] = info_slots
                    else:
                        correct_uda['inform_type_change'] = info_slots
                    # print('---' * 30)    
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                elif labeled_intent == 'anything_else':
                    assert not book_flag and not reqt_flag
                    correct_uda['anything_else'] = info_slots
                    # print('---' * 30) 
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                elif labeled_intent == 'ask_info':
                    assert not book_flag 
                    if info_flag:
                        not reqt_flag
                        if dial_id in ['SSNG0093.json']:
                            # what is the area and price range for the [value_name] , please ? 
                            correct_uda['inform_type_change'] = ['name']
                            correct_uda['ask_info'] = ['area', 'pricerange']
                        else:
                            # I guess that would be fine, I really want modern european food. Could you make sure it is in the same area and price range.
                            #  Yes. Just make sure it is something in the expensive range.
                            correct_uda['inform_type'] = info_slots
                    else:
                        if dial_id in ['SSNG0047.json', 'SSNG0094.json', 'SSNG0107.json', 'SSNG0166.json', 'SSNG0186.json']:
                            # Sure, can you find me something close-by that has comparable prices? And can I get the reference number please?
                            correct_uda['anything_else'] = info_slots
                        else:
                            assert not info_flag
                        if reqt_flag:
                            correct_uda['ask_info'] = reqt_slots
                        # print('---' * 30)    
                        # print(turn['sys'], '\n')
                        # print('utt = ', utt_delex, '\t', utt)
                        # print(correct_uda, '\t', dial_id, info_flag, book_flag, reqt_flag)
                else:
                    assert not slots
                    correct_uda['goodbye'] = []
            else:
                # Yes another Chinese restaurant for six people.
                correct_uda['anything_else'] = info_slots
                correct_uda['make_reservation'] = book_slots
    else:
        if labeled_intent == ext_intent:
            if labeled_intent == 'inform_type':
                if book_flag:
                    # Just two. It's for my wife and I. 
                    correct_uda['inform_type'] = slots
                    pass
                elif 'of course' in utt_delex or 'have you eaten' in utt_delex:
                    return {}
                else:
                    assert not reqt_flag
                    correct_uda['inform_type'] = info_slots
                    # print('---' * 30) 
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                    if 'preference' in utt_delex:
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                        pass
            elif labeled_intent == 'inform_type_change':
                assert not book_flag
                if dial_id in ['SNG0484.json']:
                    if 'other' in utt_delex:
                        correct_uda['anything_else'] = info_slots
                    else:
                        correct_uda['inform_type_change'] = info_slots
                elif dial_id in ['WOZ20615.json']:
                    correct_uda['anything_else'] = info_slots
                elif dial_id in ['SNG01196.json', 'SNG01609.json', 'SNG0487.json', 'SNG0495.json',
                'SNG0529.json', 'WOZ20615.json']:
                    correct_uda['inform_type_change'] = info_slots
                else:
                    correct_uda['inform_type'] = info_slots
                if reqt_flag:
                    correct_uda['ask_info'] = reqt_slots
                    # print('---' * 30) 
                    # print(turn['sys'], '\n')
                    # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                    # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
            elif labeled_intent == 'make_reservation':
                correct_uda['make_reservation'] = book_slots
                if info_flag:
                    correct_uda['inform_type'] = info_slots
                if reqt_flag:
                    correct_uda['ask_info'] = reqt_slots
            elif labeled_intent == 'make_reservation_change_time':
                assert not info_flag
                correct_uda['make_reservation'] = book_slots
                if reqt_flag:
                    correct_uda['ask_info'] = reqt_slots
            elif labeled_intent == 'anything_else':
                assert not book_flag
                correct_uda['anything_else'] = info_slots
                if reqt_flag:
                    correct_uda['ask_info'] = reqt_slots
                # print(dial_id, 'utt_delex = ', utt_delex, '\t', correct_uda, '\n')
            elif labeled_intent == 'ask_info':
                assert not book_flag
                if info_flag:    
                    correct_uda['inform_type'] = info_slots
                    if reqt_flag:   # it should be asking for reqt_slots
                        correct_uda['ask_info'] = reqt_slots
                    else: # it should be asking for info_slots        
                        if ' food ' in utt_delex and ('type' in utt_delex or 'kind' in utt_delex):
                            correct_uda['ask_info'] = ['food']
                        elif 'number' in utt_delex:
                            correct_uda['ask_info'] = ['phone']
                        elif ' address ' in utt_delex or dial_id in ['SNG02071.json', 'SSNG0133.json', 'WOZ20629.json']:
                            correct_uda['ask_info'] = ['address']
                        elif dial_id in ['SNG1325.json', 'WOZ20222.json']:
                            correct_uda['ask_info'] = ['area']
                        elif dial_id in ['SNG0569.json', 'SNG0622.json', 'SSNG0114.json', 'WOZ20019.json', 'WOZ20107.json']:
                            correct_uda['ask_info'] = ['pricerange']
                        elif 'part of town' in utt_delex or 'located' in utt_delex or ' price range ' in utt_delex:
                            correct_uda['ask_info'] = []
                            pass
                        else:
                            correct_uda['ask_info'] = []
                            # print(dial_id, 'utt_delex = ', utt_delex, '\t', correct_uda, '\n')
                else:
                    if reqt_flag:  # it should be asking for reqt_slots
                        correct_uda['ask_info'] = reqt_slots
                        # print(dial_id, 'utt_delex = ', utt_delex, '\t', correct_uda, '\n')
                    else:  # it should be asking for info_slots or reqt_slots
                        if 'reference number' in utt_delex:
                            correct_uda['ask_info'] = ['reference']
                        elif dial_id in ['WOZ20479.json'] or 'area' in utt_delex or 'part of town' in utt_delex:
                            correct_uda['ask_info'] = ['area']
                        elif 'food' in utt_delex:
                            correct_uda['ask_info'] = ['food']
                        elif 'price range' in utt_delex:
                            correct_uda['ask_info'] = ['pricerange']
                        elif 'phone' in utt_delex or 'number' in utt_delex:
                            correct_uda['ask_info'] = ['phone']
                        elif 'address' in utt_delex or 'located' in utt_delex:
                            correct_uda['ask_info'] = ['address']
                        else:
                            correct_uda['ask_info'] = []
                            # print(dial_id, 'utt_delex = ', utt_delex, '\t', correct_uda, '\n')
            else:
                if reqt_flag:
                    if dial_id in ['SNG0477.json', 'SNG0530.json', 'SNG0728.json', 'SNG0730.json', 'SSNG0054.json',
                                   'SSNG0155.json', 'WOZ20216.json']:
                        correct_uda['goodbye'] = []
                    else:
                        correct_uda['ask_info'] = reqt_slots
                        # print('---' * 30) 
                        # print(turn['sys'], '\n')
                        # print('utt_delex = ', utt_delex, '\t', utt, '\n')
                        # print(dial_id, labeled_intent, ext_intent, slots, '\t', correct_uda)
                else:
                    correct_uda['goodbye'] = []
        else:
            if not info_flag and not book_flag:
                if not reqt_flag:
                    if ext_intent == 'goodbye':
                        correct_uda['goodbye'] = []
                    elif ext_intent == 'ask_info':
                        if dial_id in ['SNG0514.json', 'SNG0627.json']:
                            correct_uda['ask_info'] = ['food']
                        elif dial_id in ['WOZ20157.json', 'WOZ20244.json']:
                            correct_uda['ask_info'] = ['pricerange']
                        else:
                            correct_uda['inform_type'] = []
                    else:
                        correct_uda[labeled_intent] = []
                        # print('---' * 30)    
                        # print(turn['sys'])
                        # print('utt = ', utt_delex, '\t', utt, '\t', ext_intent)
                        # print(dial_id, labeled_intent, ext_intent, slots)
                        # print(correct_uda, '\t', info_flag, book_flag, reqt_flag)
                else:
                    correct_uda['ask_info'] = reqt_slots
            else:
                if not info_flag:
                    assert book_flag
                    correct_uda['make_reservation'] = book_slots
                else:
                    assert info_flag
                    correct_uda['inform_type'] = info_slots
                    if dial_id in ['SNG0544.json', 'SNG0531.json']:
                        correct_uda['ask_info'] = []

                if reqt_flag:
                    correct_uda['ask_info'] = reqt_slots
    return correct_uda


def correct_uda(dials, dials_slots):
    for dial_id in dials:
        dial = dials[dial_id]
        dial_turns = dial['dials']
        dial_slots = dials_slots[dial_id]
        for t, (turn, turn_slot) in enumerate(zip(dial_turns, dial_slots)):
            # print('\n', t, '-----')
            # correct count label
            utt_delex, new_slots = process_slot_count(dial_id, turn['usr'], turn_slot[0], turn_slot[2])
            dials[dial_id]['dials'][t]['delex_usr'] = utt_delex
        
            uda_intent = turn_slot[1]
            # correct uda intent
            correct_uda = process_da_intent(dial_id, turn, uda_intent, new_slots, utt_delex)
            if not correct_uda:
                print('----')
            
            dials[dial_id]['dials'][t]['usr_act'] = correct_uda
        # pprint(dials[dial_id])
    return dials


def clean_goal(goal):
    new_goal = {k: v for k, v in goal.items() if k != 'message' and k != "id"}
    details = new_goal['details']
    
    new_goal = {}
    for k, v in details.items():
        if k not in ['book', 'fail_book', 'info', 'fail_info', 'reqt'] or not v:
            continue
        elif k in ['book' or 'fail_book']:
            new_v = {}
            for s in v:
                if s in ['day', 'time', 'people']:
                    new_v[s]=v[s]
            if v:
                new_goal[k] = new_v
        else:
            new_goal[k] = v

    # pprint(goal)
    # pprint(new_goal)
    return new_goal

def clean_text(data):
    new_dials = []
    for d, dials in data.items():
        dial_id = dials.get('ids')
        goal = dials.get('goal')
        turns = dials.get('dials')
        for i, turn in enumerate(turns):
            sys_act = turn.get('sys_act')
            if sys_act:
                for j, act in enumerate(sys_act):
                    slots = act['slots']
                    for m, slot in enumerate(slots):
                        k = slot[0].strip().lower()
                        dials['dials'][i]['sys_act'][j]['slots'][m][0] = k
                        if k == 'area':
                            for n, v in enumerate(slot[1:]):
                                v = v.strip().lower()
                                dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = v
                                if v in ['north', 'south', 'west', 'east', 'center', 'centre']:
                                    # print(dial_id, slots)
                                    pass
                                elif 'north' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'north'
                                elif 'south' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'south'
                                elif 'east' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'east'
                                elif 'west' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'west'
                                elif 'center' in v or 'centre' in v or 'cetre' in v or v in ['centrally located']:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'centre'
                                elif v in ['all over the city', 'throughout the city', 'all of cambridge']:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'any'
                                else:
                                    # print(dial_id, slots)
                                    pass
                        elif k == 'price':
                            for n, v in enumerate(slot[1:]):
                                v = v.strip().lower()
                                dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = v
                                if v in ['cheap', 'moderate', 'expensive', 'cheap to expensive']:
                                    # print(dial_id, slots)
                                    pass
                                elif 'cheap' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'cheap'
                                elif 'moderate' in v or 'moderatre' in v or 'moderatly' in v or 'moderatley priced' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'moderate'
                                elif 'quite expensive' in v or 'pretty expensive' in v or 'more expensive' in v or \
                                    'most expensive' in v or 'very expensive' in v or 'a little expensive' in v or \
                                    'expinsive' in v or 'epensive' in v:
                                    dials['dials'][i]['sys_act'][j]['slots'][m][n+1] = 'expensive'
                                elif 'expensive' in v :
                                    pass
                                elif v == 'vietnamese':
                                    dials['dials'][i]['sys_act'][j]['slots'][m]=['food', 'vietnamese']
                                else:
                                    print(dial_id, k, n, v)
        data[d] = dials
        new_dials.append(dials)
    return new_dials