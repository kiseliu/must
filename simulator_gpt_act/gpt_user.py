import sys, os, re, pdb
root_path = os.path.abspath(__file__)
for _ in range(2):
    root_path = os.path.dirname(root_path)
sys.path.append(root_path)
import logging, random
import torch
import numpy as np
from collections import defaultdict

from simulator_gpt_act.model import Model
from simulator_gpt_act.config import global_config as cfg

import simulator.dialog_config as dialog_config
from simulator.user import User
from simulator.agent.core import Action, SystemAct, UserAct


class GPT_User(User):
    def __init__(self, nlg_sample, nlg_template, model_path=None):
        super().__init__(nlg_sample=nlg_sample, nlg_template=nlg_template)
        self._set_initial_state()

        self._set_initial_goal_dic()

        # # # # # # # # # # # # # # # # 
        # # model configure setting # #

        self.device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
        if cfg.cuda:
            torch.cuda.set_device(cfg.cuda_device)
            logging.info('Device: {}'.format(torch.cuda.current_device()))
        cfg.gpt_path = model_path if model_path else cfg.eval_gpt_path
        self.m = Model(device=self.device)
        self.name = self.set_name(model_path=cfg.gpt_path)
        print('model_path = ', cfg.gpt_path)
        print('model_name = ', self.name)
        self.entity = self.m.reader.entity
        # # # # # # # # # # # # # # # # 

        self.state_list = []

        self._set_initial_model_parameters()


    def set_name(self, model_path):
        if '/at/' in model_path:
            name = 'US-GPT-AGT'
        elif '/ar/' in model_path:
            name = 'US-GPT-AGR'
        elif '/ag/' in model_path:
            name = 'US-GPT-AGG'
        else:
            name = 'US-GPT-MWZ'
        return name

    def _set_initial_state(self):
        self.state = {
                    'informed': {k:0 for k in  self.entity_type['informable_slots']},
                    'asked': {k:0 for k in  self.entity_type['requestable_slots']},
                    'asked_answered': {k:0 for k in  self.entity_type['requestable_slots'] + ['name']},
                    'reservation_informed': {k:0 for k in  self.entity_type['reservation_slots']},
                    'results': [],
                    'no_match_presented': 0,
                    'asked_anything_else': 0,
                    'no_other_presented': 0,
                    'match_presented': 0,
                    'book_fail': 0,

                    'usr_act_sequence': [],
                    'sys_act_sequence': [],

                    'inform': {k:None for k in  self.entity_type['informable_slots']},
                    'book': {k:None for k in  self.entity_type['reservation_slots']}
                    }
        self.check_constrain = []#dialog_config.CONSTRAINT_CHECK_NOTYET
        self.check_info = dialog_config.INFO_CHECK_NOTYET
        self.check_reservation = []#dialog_config.RESERVATION_CHECK_NOTYET
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        self.fail_reason = ''

    def _set_initial_goal_dic(self):
        # # goal transfer into list
        self.goal_dic = defaultdict(list)
        for key in ['cur_info', 'info_second_choice', 'cur_book', 'book_second_choice']:
            if key in self.goal:
                for slot_name in self.goal[key]:
                    self.goal_dic[slot_name] += [self.goal[key][slot_name]]
        if 'reqt' in self.goal:
            for slot_name in self.goal['reqt']:
                self.goal_dic[slot_name] = [slot_name]

        self.goal_list = list(self.goal['cur_info'].keys())
        if 'info_second_choice' in self.goal:
            self.goal_list += list(self.goal['info_second_choice'].keys())
        if 'reqt' in self.goal:
            self.goal_list += list(self.goal['reqt'])
        if 'cur_book' in self.goal:
            self.goal_list += list(self.goal['cur_book'].keys())
        if 'book_second_choice' in self.goal:
            self.goal_list += list(self.goal['book_second_choice'].keys())

    def _set_initial_model_parameters(self):
        self.turn_batch = {
                'dial_id': [0],
                'turn_num': [0],
                'user': [[0]],
                'response': [[0]],
                'bspan_gen': [[0]],
                'supervised': [True],
                'goal': [self.m.tokenizer.convert_tokens_to_ids(self.m.tokenizer.tokenize(' '.join(self.goal_list)) + ['EOS_Z0'])]
        }
        self.prev_z = None
        self.prev_m = None
        self.pv_turn = {}

    def reset(self):
        super().reset()
        self._set_initial_state()
        self._set_initial_goal_dic()
        self._set_initial_model_parameters()
        self.state_list = []

    def respond(self, sys_act, prev_sys=None):
        if sys_act is not None:
            self.state['sys_act_sequence'].append(sys_act.act)

        mode = 'test'
        turn_states = {}
        turn_num = self.turn_batch['turn_num'][0]

        if turn_num != 0:
            self.update_states_from_sys(sys_act)

        if prev_sys is None:
            prev_sys = 'Hello! What can I help you?'.lower()
        else:
            prev_sys = prev_sys.lower()

        # # format input
        utt_tokenized = self.m.tokenizer.tokenize(prev_sys) + ['EOS_U']
        utt_encoded = self.m.tokenizer.convert_tokens_to_ids(utt_tokenized)
        self.turn_batch['user'] = utt_encoded

        first_turn = self.turn_batch['turn_num'] == [0]
        inputs = self.m.reader.convert_turn_eval_URURU(self.turn_batch, self.pv_turn, first_turn)
        inputs = self.m.add_torch_input_eval(inputs)
        
        context_length = len(inputs['context'])
        # predict bspn, then generated resp
        outputs = self.m.model.generate(input_ids=inputs['context_tensor'], max_length=min(context_length+60, 1024), 
                                        pad_token_id=cfg.pad_id, eos_token_id=self.m.tokenizer.encode('EOS_Z2')[0])
        generated_bs = outputs[0].cpu().numpy().tolist()
        generated_bs = generated_bs[context_length:]
        # print('context = ', self.m.tokenizer.decode(inputs['context']))
        print('generated_bs = ', self.m.tokenizer.decode(generated_bs))
        print('goal = ', self.m.tokenizer.decode(self.turn_batch['goal'][0]))

        inputs['context_tensor_bspn'] = torch.tensor([inputs['context'] + generated_bs + self.turn_batch['goal'][0]]).to(self.device)
        context_length = len(inputs['context_tensor_bspn'][0])
        outputs_db = self.m.model.generate(input_ids=inputs['context_tensor_bspn'], max_length=min(context_length+80, 1024), 
                                        pad_token_id=cfg.pad_id, eos_token_id=self.m.tokenizer.encode('EOS_M')[0])
        generated_ar = outputs_db[0].cpu().numpy().tolist()
        generated_ar = generated_ar[context_length:]
        print('generated_ar = ', self.m.tokenizer.decode(generated_ar))

        generated_tokens = [self.m.tokenizer.decode(token) for token in generated_ar]
        # end_idx = generated_tokens.index('EOS_Z6')
        indices = []
        for i, token in enumerate(generated_tokens):
            if token == 'EOS_Z6':
                indices.append(i)
        if indices:
            start_idx = indices[0]
            end_idx = indices[-1]
            generated_act = generated_ar[:start_idx + 1]
            gen_usr_act = self.m.parse_act(self.m.tokenizer.decode(generated_act))

            generated_response = generated_ar[end_idx + 1:-1]
            generated_usr = self.m.tokenizer.decode(generated_response)
            filled_usr, slot_dic = self.fill_sentence(generated_usr)
            filled_usr = filled_usr.replace('!', '') 
            print('filled_usr = ', filled_usr)
            print(gen_usr_act, slot_dic)
        else:
            return None, None

        real_intent = ''
        if len(gen_usr_act) == 0:
            print('User actions break down!!!')
            return None, None
        elif len(gen_usr_act) == 1:
            real_intent = list(gen_usr_act.keys())[0]
        elif 'ask_info' in gen_usr_act:
            real_intent = 'ask_info'
        elif 'anything_else' in gen_usr_act:
            real_intent = 'anything_else'
        elif 'make_reservation_change_time' in gen_usr_act:
            real_intent = 'make_reservation_change_time'
        elif 'make_reservation' in gen_usr_act:
            real_intent = 'make_reservation'
        elif 'inform_type_change' in gen_usr_act:
            real_intent = 'inform_type_change'
        elif 'inform_type' in gen_usr_act:
            real_intent = 'inform_type'
        else:
            real_intent = 'goodbye'

        if real_intent == 'anything_else':
            if 'anything else' not in filled_usr and 'something else' not in filled_usr and \
                'any other' not in filled_usr:
                print('***** revise anything_else intent *****')
                if 'book' in filled_usr:
                    real_intent = 'make_reservation'
                elif 'shoot for the same thing' in filled_usr and 'day' in filled_usr:
                    real_intent = 'make_reservation'
                elif 'a different restaurant' in filled_usr:
                    pass
                gen_usr_act[real_intent] = gen_usr_act['anything_else']

        if slot_dic:
            svs = {}
            for s in gen_usr_act.get(real_intent):
                if s in slot_dic:
                    svs[s] = slot_dic[s]
            real_action = Action(real_intent, svs) if real_intent != 'goodbye' else Action(real_intent, {})
        else:
            real_action = Action(real_intent, {}) if real_intent != 'goodbye' else Action(real_intent, {})

        if turn_num != 0:
            # self.success_or_not(self.pv_turn['response'], prev_sys, filled_usr, sys_act)
            self.success_or_not(self.pv_turn['response'], prev_sys, filled_usr, sys_act)
        
        self.update_states_from_user(filled_usr)
        self.state['usr_act_sequence'].append(real_intent)

        self.turn_batch['bspn_gen'] = generated_bs
        self.turn_batch['response'] = filled_usr

        self.pv_turn['bspn_gen'] = generated_bs
        self.pv_turn['resp'] = self.m.tokenizer.convert_tokens_to_ids(self.m.tokenizer.tokenize(filled_usr))
        self.pv_turn['response'] = filled_usr

        turn_num += 1
        self.turn_batch['turn_num'] = [turn_num]
        return real_action, filled_usr


    def interact(self):
        mode = 'test'
        turn_states = {}
        turn_num = self.turn_batch['turn_num'][0]
        # utterance = input('User:',).lower()
        utterance = 'Hello! What can I help you?'.lower()
        print('Sys: ' + utterance)
        while True:

            if self.turn_batch['turn_num'][0] > 10 or utterance == 'close':
                break;

            # # format input
            utt_tokenized = word_tokenize(utterance) + ['EOS_U']
            utt_encoded   = self.m.reader.vocab.sentence_encode(utt_tokenized)

            if self.turn_batch['turn_num'] == [0]:
                self.turn_batch['user'] = [utt_encoded]
            else:
                self.turn_batch['user'] = [self.m.reader.vocab.sentence_encode(word_tokenize(self.prev_m)) + \
                                     [self.m.reader.vocab.encode('EOS_M')] + \
                                     utt_encoded]

            self.turn_batch['u_len'] = [len(i) for i in self.turn_batch['user']]
            self.turn_batch['m_len'] = [len(i) for i in self.turn_batch['response']]

            u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self.m._convert_batch(self.turn_batch, self.prev_z)

            # # execute tsd-net
            m_idx, z_idx, turn_states = self.m.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                               m_input=m_input,
                                               degree_input=degree_input, u_input_np=u_input_np,
                                               m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                               dial_id=self.turn_batch['dial_id'], **kw_ret)
            
            sent = self.m.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M')
            # print('Usr Simu: ' + sent)

            filled_sent = self.fill_sentence(sent)
            print('Usr Simu: ' + filled_sent)
            # print('Slots: ' + self.m.reader.vocab.sentence_decode(z_idx[0], eos='EOS_Z2'))
            # pdb.set_trace()
            print('Goal:' + ' '.join(self.goal_list))
            print('-------------------------------------------------------\n')
            pdb.set_trace()

            self.prev_z = z_idx
            self.prev_m = filled_sent
            turn_num += 1
            self.turn_batch['turn_num'] = [turn_num]
            # self.turn_batch['bspan'] = self.prev_z


            utterance = input('Sys:',).lower()

    def fill_sentence(self, slot_sent):
        sent = []
        slot_dic = {}

        slot_pattern = re.compile(r'[[].*?[]]', re.S)
        tokens = re.findall(slot_pattern, slot_sent)
        for i, word in enumerate(tokens):
            if '[value_' in word and ']' in word:
                if len(word[1:-1].split('_')) <2:
                    continue
                slot_name = word[1:-1].split('_')[1]
                if slot_name in self.goal_dic:
                    if len(self.goal_dic[slot_name]) > 1:
                        slot_val = self.goal_dic[slot_name].pop(0)
                    else:
                        slot_val = self.goal_dic[slot_name][0]
                elif slot_name in self.entity['informable']:
                    slot_val = random.choice(self.entity['informable'][slot_name])
                    self.goal_dic[slot_name] = [slot_val]
                    # pdb.set_trace()
                else:
                    slot_val = word
                slot_dic[slot_name] = slot_val
                slot_sent = slot_sent.replace(word, slot_val)          
        return slot_sent, slot_dic

    def success_or_not(self, prev_usr, prev_sys, cur_usr, sys_act):

        # # judge whether stop
        stop_flag = 0
        non_stop_pat = re.compile('number|phone|post|address|name|information|value_|restaurant_')
        
        if 'bye' in cur_usr and '?' not in cur_usr:
            stop_flag = 1
        elif 'thank' in cur_usr and '[' not in cur_usr and '?' not in cur_usr:
            stop_flag = 1
        elif re.match('.*have a (good|nice|lovely).*', cur_usr) and '?' not in cur_usr:
            stop_flag = 1
        elif re.match('.*(that is|thats|that s|that will be) all.*', cur_usr):
            stop_flag = 1
        elif not re.findall(non_stop_pat, cur_usr):
            if 'all set' in cur_usr:
                stop_flag = 1
            elif 'i am all i need' in cur_usr:
                stop_flag = 1
            elif 'that s it' in cur_usr:
                stop_flag = 1

        if self.turn_batch['turn_num'][0] > dialog_config.MAX_TURN:
            stop_flag = 1

        if sys_act.act == SystemAct.NOMATCH_RESULT and 'info_second_choice' not in self.goal:
            self.fail_reason = 'RNN User judge fail : sys no offer but there has no second goal'
            stop_flag = 1

        # # system ending

        if sys_act.act == SystemAct.GOODBYE:
            self.fail_reason = 'RNN User judge fail : sys say goodbye first'
            self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        # # # ask info
        # elif re.findall(r'(?<!reference) number|(?<!reservation) number|phone|post *code| address| name|information', prev_usr):
        #     if sys_act.act == SystemAct.PROVIDE_INFO:
        #         self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
        #     else:
        #         self.fail_reason = 'RNN User judge fail : usr ask info but sys do not provide info'
        #         self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            
        # # # reservation
        # prev_usr_slot = self.m.reader.delex_sent(prev_usr)
        elif re.search(r'value_time|value_day|value_people', self.m.reader.delex_sent(prev_usr)) is not None or \
           re.search(r'reference number|reservation number', prev_usr) is not None:
            # # reference number
            if sys_act.act == SystemAct.ASK_RESERVATION_INFO:
                tmp_flag = 1
                for slot_name in ['time','day','people']:
                    if slot_name in prev_sys and self.state['book'][slot_name] is not None:
                        tmp_flag = 0
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.fail_reason = 'RNN User judge fail : sys ASK_RESERVATION_INFO wrong'
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

            elif sys_act.act in [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)

            elif sys_act.act == SystemAct.PRESENT_RESULT:
                prev_sys_slot = self.m.reader.delex_sent(prev_sys)
                constraints = [slot[1:-1].split('|')[1] for slot in re.findall(r'\[.*?\]', prev_sys_slot)]
                tmp_flag = 1
                if self.state['inform']['name'] is not None:
                    tmp_flag = 0
                for slot_name in self.state['inform']:
                    if self.state['inform'][slot_name] is not None and self.state['inform'][slot_name] not in constraints:
                        tmp_flag = 0
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
                self.fail_reason = 'RNN User judge fail : usr make reservation but sys do not ask reservation info, book success or fail, present result'

        elif sys_act.act in [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
            self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            self.fail_reason = 'RNN User judge fail : sys booking but usr do not make reservation'

        elif sys_act.act == SystemAct.ASK_RESERVATION_INFO:
            if 'book' in prev_usr or 'reserv' in prev_usr:
                 self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
                self.fail_reason = 'RNN User judge fail : sys ask reservation but usr do not make reservation'

        # # # inform type

        elif sys_act.act == SystemAct.NOMATCH_RESULT:
            cur_info = {slot_name:slot_val for slot_name, slot_val in self.state['inform'].items() if slot_val is not None}
            match_list = self.query_in_DB(cur_info)
            if not match_list:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
                self.fail_reason = 'RNN User judge fail : sys nomatch wrong'


        elif sys_act.act == SystemAct.NO_OTHER:
            cur_info = {slot_name:slot_val for slot_name, slot_val in self.state['inform'].items() if slot_val is not None}
            match_list = self.query_in_DB(cur_info, skip=self.state['results'])
            if not match_list:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
                self.fail_reason = 'RNN User judge fail : sys noother wrong'

        elif re.search(r'value_area|value_food|value_pricerange', self.m.reader.delex_sent(prev_usr)) is not None:
            if sys_act.act == SystemAct.PRESENT_RESULT:
                prev_sys_slot = self.m.reader.delex_sent(prev_sys)
                constraints = [slot[1:-1].split('|')[1] for slot in re.findall(r'\[.*?\]', prev_sys_slot)]
                tmp_flag = 1
                for slot_name in self.state['inform']:
                    if self.state['inform'][slot_name] is not None and self.state['inform'][slot_name] not in constraints:
                        tmp_flag = 0
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

            elif sys_act.act == SystemAct.ASK_TYPE:
                tmp_flag = 1
                for slot_name in ['area','food','pricerange']:
                    if slot_name in prev_sys and self.state['inform'][slot_name] is not None:
                        tmp_flag = 0
                
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)              

        elif re.search(r'value_name', self.m.reader.delex_sent(prev_usr)) is not None:
            if sys_act.act == SystemAct.NOMATCH_RESULT or sys_act.act == SystemAct.PRESENT_RESULT:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        elif sys_act.act == SystemAct.ASK_TYPE:
            if self.state['inform']['name'] is not None and \
               (self.state['inform']['area'] is None or \
                self.state['inform']['food'] is None or \
                self.state['inform']['pricerange'] is None):

                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        else:
            self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)

        self.dialog_status = self.state_list[-1]

        if stop_flag:
            if dialog_config.TURN_FAIL_FOR_SL not in self.state_list:
                self.dialog_status = dialog_config.SUCCESS_DIALOG
            else:
                self.dialog_status = dialog_config.FAILED_DIALOG

        if dialog_config.SUCCESS_DIALOG:
            self.check_pair(sys_act.act)

    # def success_or_not_new(self, prev_usr, prev_sys, cur_usr, sys_act, usr_act):
    #     if sys_act.act == SystemAct.GOODBYE:
    #         self.evaluate_GOOD_BYE(sys_act)

    #     if usr_act.act == UserAct.GOODBYE:
    #         #self.episode_over = True
    #         self.evaluate_GOOD_BYE(sys_act)

    #     if sys_act is not None:
    #         self.check_pair(sys_act.act)
        
    #     if len(self.state['sys_act_sequence']) >= self.max_turn and usr_act.act != UserAct.GOODBYE:
    #         print("Maximum dialog length reached!")
    #         self.dialog_status = dialog_config.FAILED_DIALOG


    def update_states_from_user(self, cur_usr):
      cur_usr_slot = self.m.reader.delex_sent(cur_usr)
      for slot in re.findall(r'\[.*?\]', cur_usr_slot):
          if len(slot[1:-1].split('|')) < 2:
              continue
          [slot_name, slot_val] = slot[1:-1].split('|')
          slot_name = slot_name.split('_')[1]
          if slot_name in self.state['inform']:
              self.state['inform'][slot_name] = slot_val
          elif slot_name in self.state['book']:
              self.state['book'][slot_name] = slot_val

    def update_states_from_sys(self, sys_act):
        if sys_act.act == SystemAct.PRESENT_RESULT:
            self.state['results'].append(sys_act.parameters)

    def check_presented_result(self, match):
        """
        checke the presented_result/no_match_result
        :return:
        """
        if match == dialog_config.NO_MATCH:
            query_result = self.query_in_DB(self.goal['cur_info'], skip=self.state['results'])
            if len(query_result) == 0:
                return dialog_config.CONSTRAINT_CHECK_SUCCESS
            else:
                print("There is at least one match {}".format(query_result[0]))
                return dialog_config.CONSTRAINT_CHECK_FAILURE
        elif match == dialog_config.NO_OTHER:
            if self.state['usr_act_sequence'][-1] == UserAct.ANYTHING_ELSE and self.state['sys_act_sequence'][-2] == SystemAct.PRESENT_RESULT:
                ###################################
                # can only be the response of ANYTHING_ELSE, and present_result is also the previous response
                # the only correct sequence is sys: present_result -> usr: anything_else -> sys: no_other
                ###################################
                query_result = self.query_in_DB(self.goal['cur_info'])
                if len(query_result) == 1:
                    # indeed there is only one match
                    return dialog_config.CONSTRAINT_CHECK_SUCCESS
                elif len(query_result) == 0:
                    print("There is no match at all for the constrain from the very beginning!")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
                else:
                    print("There are more than one match for the constrain! should present the second result!")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
            else:
                ###################################
                # can only be the response of ANYTHING_ELSE, and present_result already existed before
                # the only correct sequence is sys: present_result -> usr: anything_else -> sys: no_other
                ###################################
                if self.state['usr_act_sequence'][-1] != UserAct.ANYTHING_ELSE:
                    print("FAIL, because the user didn't ask for anything_else")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
                elif self.state['sys_act_sequence'][-2] != SystemAct.PRESENT_RESULT:
                    print("FAIL, because the last sys act is not present_result")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
        else:

            tmp_constraint_check = [(self.goal['cur_info'][entity] == match[entity]) for entity, value in self.state['informed'].items() \
                                    if ((value > 0) and (entity in self.goal['cur_info']))]

            if len(tmp_constraint_check) and np.all(tmp_constraint_check):
                print("Warning, the system hasn't captured all the correct entity but gives the result anyway")
                return dialog_config.CONSTRAINT_CHECK_SUCCESS
            else:
                print("Warning, the system hasn't captured all the correct entity but gives the result anyway, and the result is not correct")
                return dialog_config.CONSTRAINT_CHECK_FAILURE


    def check_pair(self, sys_act_str):
        last_usr_act = self.state['usr_act_sequence'][-1]
        if last_usr_act == UserAct.INFORM_TYPE:
            if sys_act_str not in [SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT]:
                self.fail_reason = "User judge fail : user inform but sys not ask slot, present result, and nomatch result"
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.INFORM_TYPE_CHANGE:
            if sys_act_str not in [SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT]:
                self.fail_reason = "User judge fail : user inform change but sys not ask slot, present result, and nomatch result"
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.ASK_INFO:
            if sys_act_str not in [SystemAct.PROVIDE_INFO]:
                self.fail_reason = "User judge fail : user ask info but sys not provide info"
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.MAKE_RESERVATION:
            if sys_act_str not in [SystemAct.ASK_RESERVATION_INFO, SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                self.fail_reason = "User judge fail : user make reservation but sys not ask reservation info, book success or book fail"
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.MAKE_RESERVATION_CHANGE_TIME:
            if sys_act_str not in [SystemAct.ASK_RESERVATION_INFO, SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                self.fail_reason = "User judge fail : user make reservation change but sys not ask reservation info, book success or book fail"
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.ANYTHING_ELSE:
            if sys_act_str not in [SystemAct.NO_OTHER, SystemAct.PRESENT_RESULT]:
                self.fail_reason = "User judge fail : user ask anything else but sys not present, and no other"
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.GOODBYE:
            if sys_act_str not in [SystemAct.GOODBYE]:
                self.fail_reason = "User judge fail : user says goodbye but sys not say goodbye"
                self.dialog_status = dialog_config.FAILED_DIALOG


    def evaluate_GOOD_BYE(self, sys_act):
        # success conditions: 1) present correct restaurant 2) present correct info/ try to make a reservation
        # failure conditions: 1) check_constrain = FALSE (the result presented is incorrect)
        #                     2) didn't answer ask_info, i.e. value in self.state['asked'] > 0
        #
        print('Entering evaluate goodbye')
        if sys_act.act == SystemAct.GOODBYE and UserAct.GOODBYE not in self.state['usr_act_sequence']:
            self.fail_reason = "User judge fail : sys said googbye but user do not say goodbye"
            self.dialog_status = dialog_config.FAILED_DIALOG
            return

        # 1. check the restaurant info
        if len(self.check_constrain) == 0:
            #didn't present result at all
            self.fail_reason = "User judge fail : didn't present result at all"
            self.dialog_status = dialog_config.FAILED_DIALOG
            return
        else:
            # presented some results
            if UserAct.INFORM_TYPE_CHANGE in self.state['usr_act_sequence']:
                if len(self.check_constrain) < 2:
                    # because there is a second option,
                    self.fail_reason = "User judge fail : INFORM_TYPE_CHANGE presented some results because there is a second option"
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    pass
            elif UserAct.ANYTHING_ELSE in self.state['usr_act_sequence']:
                if len(self.check_constrain) < 2:
                    # because there is a second option,
                    self.fail_reason = "User judge fail : ANYTHING_ELSE presented some results because there is a second option"
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    pass
            else:
                # no second option
                all_constrain = [(c == dialog_config.CONSTRAINT_CHECK_SUCCESS) for c in self.check_constrain]
                if not np.all(all_constrain):
                    self.fail_reason = "User judge fail : not np.all(all_constrain) because there is a second option"
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    # need to check the last one in case of "inform_type_change"
                    pass

        if UserAct.ASK_INFO in self.state['usr_act_sequence']:
            if self.check_info == dialog_config.INFO_CHECK_NOTYET:
                self.fail_reason = "User judge fail : INFO_CHECK_NOTYET"
                self.dialog_status = dialog_config.FAILED_DIALOG
                return
            elif self.check_info == dialog_config.INFO_CHECK_FAILURE:
                self.fail_reason = "User judge fail : INFO_CHECK_FAILURE"
                self.dialog_status = dialog_config.FAILED_DIALOG
                return
            else:
                pass
                # self.dialog_status = dialog_config.SUCCESS_DIALOG

        if UserAct.MAKE_RESERVATION in self.state['usr_act_sequence']:
            if len(self.check_reservation) == 0:
                # didn't present result at all
                self.fail_reason = "User judge fail : UserAct.MAKE_RESERVATION didnt present result at all"
                self.dialog_status = dialog_config.FAILED_DIALOG
                return
            else:
                all_reservation_constrain = [(c == dialog_config.RESERVATION_CHECK_SUCCESS) for c in self.check_reservation]
                if not np.all(all_reservation_constrain):
                    self.fail_reason = "User judge fail : not np.all(all_constrain)"
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    pass

        self.dialog_status = dialog_config.SUCCESS_DIALOG
        return

def main():
    user = GPT_User()
    # user.respond()

    user.interact()

# if __name__ == "__main__":
    # main()
