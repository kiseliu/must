import numpy as np
import os
import sys
sys.path.append('/home/ma-user/work/liuyajiao/DialogueSystems/sequicity/')
import json
import pickle
from simulator_gpt_act.config import global_config as cfg
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
import logging
import random
import re
import csv
import time, datetime
import pdb
import argparse

import torch
from torch.optim import Adam, RMSprop
from torch.autograd import Variable

from transformers import GPT2Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='test')
parser.add_argument('-model', default='tsdf-usr')
parser.add_argument('-cfg', nargs='*')
args = parser.parse_args()

# cfg.init_handler(args.model)
cfg.dataset = args.model.split('-')[-1]

if args.cfg:
    for pair in args.cfg:
        k, v = tuple(pair.split('='))
        dtype = type(getattr(cfg, k))
        if dtype == type(None):
            raise ValueError()
        if dtype is bool:
            v = False if v == 'False' else True
        else:
            v = dtype(v)
        setattr(cfg, k, v)

logging.info(str(cfg))
if cfg.cuda:
    torch.cuda.set_device(cfg.cuda_device)
    logging.info('Device: {}'.format(torch.cuda.current_device()))
cfg.mode = args.mode

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

print('cfg = ', cfg)


def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        idx = s[sidx:].find(r)
        if idx == -1:
            return s, -1
        idx += sidx
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
    return s


class _ReaderBase:
    class LabelSet:
        def __init__(self):
            self._idx2item = {}
            self._item2idx = {}
            self._freq_dict = {}

        def __len__(self):
            return len(self._idx2item)

        def _absolute_add_item(self, item):
            idx = len(self)
            self._idx2item[idx] = item
            self._item2idx[item] = idx

        def add_item(self, item):
            if item not in self._freq_dict:
                self._freq_dict[item] = 0
            self._freq_dict[item] += 1

        def construct(self, limit):
            l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
            print('Actual label size %d' % (len(l) + len(self._idx2item)))
            if len(l) + len(self._idx2item) < limit:
                logging.warning('actual label set smaller than that configured: {}/{}'
                                .format(len(l) + len(self._idx2item), limit))
            for item in l:
                if item not in self._item2idx:
                    idx = len(self._idx2item)
                    self._idx2item[idx] = item
                    self._item2idx[item] = idx
                    if len(self._idx2item) >= limit:
                        break

        def encode(self, item):
            return self._item2idx[item]

        def decode(self, idx):
            return self._idx2item[idx]

    class Vocab(LabelSet):
        def __init__(self, init=True):
            _ReaderBase.LabelSet.__init__(self)
            if init:
                self._absolute_add_item('<pad>')  # 0
                self._absolute_add_item('<go>')  # 1
                self._absolute_add_item('<unk>')  # 2
                self._absolute_add_item('<go2>')  # 3

        def load_vocab(self, vocab_path):
            f = open(vocab_path, 'rb')
            dic = pickle.load(f)
            self._idx2item = dic['idx2item']
            self._item2idx = dic['item2idx']
            self._freq_dict = dic['freq_dict']
            f.close()

        def save_vocab(self, vocab_path):
            f = open(vocab_path, 'wb')
            dic = {
                'idx2item': self._idx2item,
                'item2idx': self._item2idx,
                'freq_dict': self._freq_dict
            }
            pickle.dump(dic, f)
            f.close()

        def sentence_encode(self, word_list):
            return [self.encode(_) for _ in word_list]

        def sentence_decode(self, index_list, eos=None):
            l = [self.decode(_) for _ in index_list]
            if not eos or eos not in l:
                return ' '.join(l)
            else:
                idx = l.index(eos)
                return ' '.join(l[:idx])

        def nl_decode(self, l, eos=None):
            return [self.sentence_decode(_, eos) + '\n' for _ in l]

        def encode(self, item):
            if item in self._item2idx:
                return self._item2idx[item]
            else:
                return self._item2idx['<unk>']

        def decode(self, idx):
            if int(idx) < len(self):
                return self._idx2item[int(idx)]
            else:
                return 'ITEM_%d' % (idx - cfg.vocab_size)

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = self.Vocab()
        self.result_file = ''
        self.set_stats = {}

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
                    if not turn['supervised']:
                        turn['degree'] = [0.] * cfg.degree_size  # unsupervised learning. DB degree should be unknown
        return all_batches

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def get_batches(self, set_name):
        global dia_count
        log_str = ''

        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            batches = self._construct_mini_batch(turn_bucket[k])
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                k, len(turn_bucket[k]), len(batches), len(batches[-1]))

            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        self._mark_batch_as_supervised(all_batches)

        log_str += 'total batch num: %d\n' % len(all_batches)

        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        if set_name == 'train':
            random.shuffle(all_batches)
        return all_batches


    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self._transpose_batch(batch)

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials
        return dial

    def wrap_result(self, turn, gen_m, gen_a, gen_z, eos_syntax=None, prev_z=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """
        results = []
        if eos_syntax is None:
            eos_syntax = {'response': 'EOS_M', 'user': 'EOS_U', 'bspan': 'EOS_Z2', 'uda': 'EOS_Z6'}
        # batch_size = len(turn['user'])
        for i in range(1):
            entry = {}
            if prev_z:
                src = prev_z['bspn_gen'] + turn['user']
            else:
                src = turn['user']
            for key in turn:
                entry[key] = turn[key]
                if key in eos_syntax:
                    entry[key] = self.tokenizer.decode(entry[key])
            entry['generated_response'] = self.tokenizer.decode(gen_m)
            entry['generated_act'] = self.tokenizer.decode(gen_a)
            entry['generated_bspan'] = self.tokenizer.decode(gen_z)
            results.append(entry)

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_act', 'uda', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised']
        for result in results:
            del_k = []
            for k in result:
                if k not in field:
                    del_k.append(k)
            for k in del_k:
                result.pop(k)
        return results

    def db_search(self, constraints):
        raise NotImplementedError('This is an abstract method')

    def db_degree_handler(self, z_samples, *args, **kwargs):
        """
        returns degree of database searching and it may be used to control further decoding.
        One hot vector, indicating the number of entries found: [0, 1, 2, 3, 4, >=5]
        :param z_samples: nested list of B * [T]
        :return: an one-hot control *numpy* control vector
        """
        control_vec = []

        for cons_idx_list in z_samples:
            constraints = set()
            for cons in cons_idx_list:
                if type(cons) is not str:
                    cons = self.vocab.decode(cons)
                if cons == 'EOS_Z1':
                    break
                constraints.add(cons)
            match_result = self.db_search(constraints)
            degree = len(match_result)
            # modified
            # degree = 0
            control_vec.append(self._degree_vec_mapping(degree))
        return np.array(control_vec)

    def _degree_vec_mapping(self, match_num):
        l = [0.] * cfg.degree_size
        l[min(cfg.degree_size - 1, match_num)] = 1.
        return l


class User_Simulator_Reader(_ReaderBase):
    def __init__(self, tokenizer):
        super().__init__()
        self.usr_acts = ['inform_type', 'inform_type_change', 'ask_info', 'make_reservation',
                         'make_reservation_change_time', 'anything_else', 'goodbye']
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens':
                    ['EOS_Z0', 'EOS_Z1', 'EOS_Z2', 'EOS_Z3', 'EOS_Z4', 'EOS_Z5', 'EOS_Z6', 'EOS_U', 'EOS_M',
                    '[value_name]', '[value_address]', 
                    '[value_pricerange]', '[value_price]', '[value_food]', '[value_area]',
                    '[value_time]', '[value_day]', '[value_people]', 
                    '[inform_type]', '[inform_type_change]', '[ask_info]', "[make_reservation]", 
                    "[make_reservation_change_time]", "[anything_else]", "[goodbye]",
                    '<pad>']})
        cfg.pad_id = self.tokenizer.encode('<pad>')[0]
        self.empty_prefix = 'EOS_Z1 EOS_Z3 EOS_Z5 EOS_Z4 EOS_Z2 EOS_M'.split()
        self.empty_prefix = self.tokenizer.convert_tokens_to_ids(self.empty_prefix)
        print('vocab size = ', len(self.tokenizer))
        self._construct(cfg.data, cfg.db, cfg.entity)
        self.result_file = ''

    def _value_key_map(self, db_data):
        requestable_keys = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']
        value_key = {}
        for db_entry in db_data:
            for k, v in db_entry.items():
                if k in requestable_keys:
                    value_key[v] = k
        return value_key

    def _construct(self, data_json_path, db_json_path, entity_json_path):
        """
        construct encoded train, dev, test set.
        :param data_json_path:
        :param db_json_path:
        :return:
        """
        with open(data_json_path) as raw_data_json:
            raw_data = json.loads(raw_data_json.read().lower())
        with open(db_json_path) as db_json:
            db_data = json.loads(db_json.read().lower())
        with open(entity_json_path) as entity_json:
            entity_data = json.loads(entity_json.read().lower())

        self.db = db_data
        self.entity = entity_data

        if os.path.exists(cfg.encoded_file_path):
            encoded_data = json.loads(open(cfg.encoded_file_path, 'r', encoding='utf-8') .read())
        else:
            tokenized_data = self._get_tokenized_data(raw_data, db_data)
            encoded_data = self._get_encoded_data(tokenized_data)
        self.train, self.dev, self.test = self._split_data(encoded_data, cfg.split)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
        raw_data_json.close()
        db_json.close()

    def db_search(self, constraints):
        match_results = []
        for entry in self.db:
            entry_values = ' '.join([str(item) for item in entry.values()])
            match = True
            for c in constraints:
                if c not in entry_values:
                    match = False
                    break
            if match:
                match_results.append(entry)
        return match_results


    def normalize(self, text):
        def insertSpace(token, text):
            sidx = 0
            while True:
                sidx = text.find(token, sidx)
                if sidx == -1:
                    break
                if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                        re.match('[0-9]', text[sidx + 1]):
                    sidx += 1
                    continue
                if text[sidx - 1] != ' ':
                    text = text[:sidx] + ' ' + text[sidx:]
                    sidx += 1
                if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                    text = text[:sidx + 1] + ' ' + text[sidx + 1:]
                sidx += 1
            return text
        # lower case every word
        text = text.lower()

        # replace white spaces in front and end
        text = re.sub(r'^\s*|\s*$', '', text)

        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)

        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                        text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

        # weird unicode bug
        text = re.sub(u"(\u2018|\u2019)", "'", text)


        text = ' ' + text + ' '
        # # replace time and and price
        timepat = re.compile(" \d{1,2}[:]\d{1,2}[ \.,\?]| \d{4}[ \.,\?]| \d{1,2}[ap][m\. ]+| \d{1,2} [ap][m\. ]+| \d{1,2}[:]\d{1,2}[ap]m[ \.,\?]")
        # # some utterances just miss the ":"
        # timepat_noise = re.compile(" at \d{4}[ \.,\?]")
        pricepat = re.compile("\d{1,3}[.]\d{1,2}")

        value_time = re.findall(timepat, text)

        while value_time:
            index = text.find(value_time[0])
            text = text[:index] + \
                   ' [value_time|' + value_time[0][1:-1] + ']' + \
                   text[index+len(value_time[0]) - 1:]
            value_time = re.findall(timepat, text)

        value_price = re.findall(pricepat, text)

        if value_price:
            text = re.sub(pricepat, ' [value_price|' + value_price[0] + '] ', text)

        text = text[1:-1]

        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')

        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\"\<>@\(\)]', '', text)

        # insert white space before and after tokens:
        for token in ['?', '.', ',', '!']:
            text = insertSpace(token, text)

        # insert white space for 's
        text = insertSpace('\'s', text)

        # replace it's, does't, you'd ... etc
        text = re.sub('^\'', '', text)
        text = re.sub('\'$', '', text)
        text = re.sub('\'\s', ' ', text)
        text = re.sub('\s\'', ' ', text)

        fin = open('./data/multi-woz/mapping.pair')
        replacements = []
        for line in fin.readlines():
            tok_from, tok_to = line.replace('\n', '').split('\t')
            replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

        for fromx, tox in replacements:
            text = ' ' + text + ' '
            text = text.replace(fromx, tox)[1:-1]

        # remove multiple spaces
        text = re.sub(' +', ' ', text)

        # concatenate numbers
        tmp = text
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and \
                    re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)

        return text

    def check_balance(self, string):
        # open_tup = tuple('[') 
        # close_tup = tuple(']') 
        # map = dict(zip(open_tup, close_tup)) 
        queue = 0
      
        for i in string: 
            if i == '[': 
                queue += 1
            elif i == ']': 
                if not queue: 
                    return False
                else:
                    queue -= 1
        if not queue:
            return True
        else:
            return False

    def delexicalise(self, utt, dictionary):
        for key, val in dictionary:
            utt = (' ' + utt + ' ')
            if key in utt:
                idx = 0
                while utt[idx:].find(' ' + key + ' ') != -1:
                    idx += utt[idx:].find(' ' + key + ' ')
                    # # to exclude the case that 'ask' is a verb
                    if key == 'ask' and idx > 2 and utt[idx-2:idx] == ' i':
                        idx += 1
                        continue
                    if self.check_balance(utt[:idx]):
                        utt = utt[:idx] + ' ' + val[:-1] + '|' + key + '] ' + utt[idx + len(key) + 2:]
                        idx += len(key) + 4 + len(val[:-1])
                    else:
                        idx += len(key)
            utt = utt[1:-1]

        return utt

    def delex_people_count(self, sent):
        sent = ' ' + sent + ' '
        digitpat = re.compile('(?<!looking for)(?<=for) \d+ (?!of)|(?<=party of) \d+ | \d+ (?=people|person|of us)')
        value_people = re.findall(digitpat, sent)
        while value_people:
            index = sent.find(value_people[0])
            if not self.check_balance(sent[:index]):
                value_people.pop(0)
                continue

            sent = sent[:index] + \
                   ' [value_people|' + value_people[0][1:-1] + '] ' + \
                   sent[index+len(value_people[0]):]
            value_people = re.findall(digitpat, sent)
        sent = sent[1:-1]
        return sent

    def delex_sent(self, response):
        # # replace time, date, specific price
        response = self.normalize(response)

        # # replace info in db
        db_entity_file = open('data/multi-woz/db_entity_file.pkl','rb')
        db_entity_list = pickle.load(db_entity_file)
        db_entity_file.close()
        response = self.delexicalise(response, db_entity_list)

        # # replace # of people for reservation
        response = self.delex_people_count(response)
        return response

    def _replace_entity(self, response, goal):
        response = self.delex_sent(response)

        # # # replace and generate dic
        slotpat = re.compile('\[.*?\]')
        slots = re.findall(slotpat, response)

        for slot in slots:
            [slot_name, slot_val] = slot[1:-1].split('|')
            if slot_name.split('_')[0] == 'restaurant' and (slot_name.split('_')[1]) not in goal:
                response = response.replace(slot, slot_val)
            else:
                response = response.replace(slot, slot_name.split('_')[1] + '_SLOT')

        return response

    def _get_tokenized_data(self, raw_data, db_data):
        tokenized_data = []
        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for i, turn in enumerate(dial['dials']):
                turn_num = i
                constraint = []
                requested = []
                book = []
                recommend = []
                select = []
                goal = []

                for slot in turn['sys_act']:
                    if slot['act'] == 'inform':
                        s = slot['slots'][0][1].strip()
                        if s not in ['dontcare', 'none']:
                            constraint.append(s)
                    elif slot['act'] == 'request':
                        requested.append(slot['slots'][0][1].strip())
                    elif slot['act'] == 'book_inform':
                        book.append(slot['slots'][0][1].strip())
                    elif slot['act'] == 'recommend':
                        recommend.append(slot['slots'][0][1].strip())
                    elif slot['act'] == 'select':
                        for slot_sng in slot['slots'][0][1:]:
                            select.append(slot_sng.strip())
                    else:
                        book.append(slot['slots'][0][1].strip())

                for slot_type in turn['goal']:
                    for slot_val in turn['goal'][slot_type]:
                        if slot_val not in goal:
                            goal.append(slot_val.strip())

                goal.append('EOS_Z0')   # ['EOS_Z0', 'EOS_Z1', 'EOS_Z2', 'EOS_Z3', 'EOS_Z4', 'EOS_Z5', 'EOS_U', 'EOS_M']

                usr_da = []
                for slot, values in turn['usr_act'].items():
                    usr_da.append('[' + slot.strip() + ']')
                    usr_da.extend([v.strip() for v in values])
                usr_da.append('EOS_Z6')

                degree = len(self.db_search(constraint))
                requested = sorted(requested)
                book = sorted(book)
                constraint.append('EOS_Z1')
                book.append('EOS_Z3')
                recommend.append('EOS_Z4')
                select.append('EOS_Z5')
                requested.append('EOS_Z2')
                
                user = self.tokenizer.tokenize(turn['sys']) + ['EOS_U']
                # response = self.tokenizer.tokenize(self._replace_entity(turn['b']['sent'], goal)) + ['EOS_M']
                response = self.tokenizer.tokenize(turn['delex_usr']) + ['EOS_M']
                nodelex_resp = self.tokenizer.tokenize(turn['usr']) + ['EOS_M']

                print('sys = ', turn['sys'])
                print('constraint = ', constraint)
                print('book = ', book)
                print('recommend = ', recommend)
                print('select = ', select)
                print('requested = ', requested)
                # print('degree = ', degree)
                print('goal = ', goal)
                print('usr da = ', usr_da)
                print('resp = ', turn['delex_usr'])

                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'nodelex_resp': nodelex_resp,
                    'uda': self.tokenizer.tokenize(' '.join(usr_da)),
                    'book': self.tokenizer.tokenize(' '.join(book)),
                    'goal': self.tokenizer.tokenize(' '.join(goal)),
                    'select': self.tokenizer.tokenize(' '.join(select)),
                    'recommend': self.tokenizer.tokenize(' '.join(recommend)),
                    'constraint': self.tokenizer.tokenize(' '.join(constraint)),
                    'requested': self.tokenizer.tokenize(' '.join(requested)),
                    'degree': degree,
                })
                # pdb.set_trace()
                print(tokenized_dial[-1])
                # break
            tokenized_data.append(tokenized_dial)
            # break
        return tokenized_data

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.tokenizer.convert_tokens_to_ids(turn['user'])
                usr_da = self.tokenizer.convert_tokens_to_ids(turn['uda'])
                response = self.tokenizer.convert_tokens_to_ids(turn['response'])
                nodelex_resp = self.tokenizer.convert_tokens_to_ids(turn['nodelex_resp'])
                constraint = self.tokenizer.convert_tokens_to_ids(turn['constraint'])
                requested = self.tokenizer.convert_tokens_to_ids(turn['requested'])
                select = self.tokenizer.convert_tokens_to_ids(turn['select'])
                goal = self.tokenizer.convert_tokens_to_ids(turn['goal'])
                recommend = self.tokenizer.convert_tokens_to_ids(turn['recommend'])
                book = self.tokenizer.convert_tokens_to_ids(turn['book'])
                degree = self._degree_vec_mapping(turn['degree'])
                turn_num = turn['turn_num']
                dial_id = turn['dial_id']
                # pdb.set_trace()
                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    # 'user': prev_response + user,
                    'user': user,
                    'uda': usr_da,
                    'response': response,
                    'nodelex_resp': nodelex_resp,
                    'bspan': constraint + book + select + recommend + requested,
                    # 'bspan': goal + constraint + book + select + recommend + requested,
                    'goal': goal,
                    # 'u_len': len(prev_response + user),
                    'u_len': len(user),
                    'm_len': len(response),
                    'degree': degree,
                })
                # modified
                prev_response = response
            # print('encoded_dial = ', encoded_dial)
            encoded_data.append(encoded_dial)
        with open(cfg.encoded_file_path, 'w', encoding='utf-8') as fw:
            json.dump(encoded_data, fw)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[test_thr:]
        return train, dev, test

    def convert_batch_turn(self, turn_batch, pv_batch, first_turn=False):
        """
        URURU
        convert the current and the last turn
        concat [U_0,R_0,...,U_{t-1}, R_{t-1}, U_t, B_t, A_t, R_t]
        firts turn: [U_t, B_t, A_t, R_t]
        try: [user, bspn, db, aspn, resp]

        """
        inputs = {}
        if first_turn:
            contexts = []
            labels = []
            batch_zipped = zip(turn_batch['user'], turn_batch['bspan'], 
                            turn_batch['degree'], turn_batch['goal'], 
                            turn_batch['response'], turn_batch['nodelex_resp'])
            for u, b, db, g, r, ndr in batch_zipped:
                context = u+b+db+g+r
                contexts.append(context)
                label = u + ndr
                labels.append(label)
            inputs['contexts'] = contexts
            inputs['contexts_np'], inputs['lengths'] = self.padSeqs_gpt(inputs['contexts'], 0)

            inputs['labels'] = labels
        else:
            contexts = []
            labels = []
            batch_zipped = zip(pv_batch, turn_batch['user'], turn_batch['bspan'], 
                            turn_batch['degree'], turn_batch['goal'], 
                            turn_batch['response'], turn_batch['nodelex_resp'])
            for ur, u, b, db, g, r, ndr in batch_zipped:
                context = ur + u + b + db + g + r
                contexts.append(context)
                label = ur + u + ndr
                labels.append(label)
            inputs['contexts'] = contexts
            contexts_np, lengths = self.padSeqs_gpt(inputs['contexts'], 0)
            inputs['contexts_np'] = contexts_np
            inputs['lengths'] = lengths

            inputs['labels'] = labels
        return inputs


    def convert_raw_turn(self, turn_batch, pv_batch, first_turn=False):
        """
        URURU
        convert the current and the last turn
        concat [U_0,R_0,...,U_{t-1}, R_{t-1}, U_t, B_t, A_t, R_t]
        firts turn: [U_t, B_t, A_t, R_t]
        try: [user, bspn, db, aspn, resp]

        """
        inputs = {}
        if first_turn:
            contexts = []
            labels = []
            batch_zipped = zip(turn_batch['user'], turn_batch['bspan'], turn_batch['goal'], turn_batch['uda'],
                            turn_batch['response'], turn_batch['nodelex_resp'])
            for u, b, g, da, r, ndr in batch_zipped:
                context = self.empty_prefix + u + b + g + da + r
                contexts.append(context)
                label = b + ndr
                labels.append(label)
            inputs['contexts'] = contexts
            inputs['contexts_np'], inputs['lengths'] = self.padSeqs_gpt(inputs['contexts'], 0)

            inputs['labels'] = labels
        else:
            contexts = []
            labels = []
            batch_zipped = zip(pv_batch, turn_batch['user'], turn_batch['bspan'], turn_batch['goal'], turn_batch['uda'],
                            turn_batch['response'], turn_batch['nodelex_resp'])
            for ur, u, b, g, da, r, ndr in batch_zipped:
                context = ur + u + b + g + da + r
                contexts.append(context)
                label = b + ndr
                labels.append(label)
            inputs['contexts'] = contexts
            contexts_np, lengths = self.padSeqs_gpt(inputs['contexts'], 0)
            inputs['contexts_np'] = contexts_np
            inputs['lengths'] = lengths

            inputs['labels'] = labels
        return inputs


    def convert_turn_eval_URURU(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous U_t, R_t] predict R_t
            firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        R_t  --> B_t + G_t --> U_t
        B_{t-1} U_{t-1} R_t  --> B_t + G_t --> U_t
        """
        inputs = {}
        if first_turn:
            inputs['context'] = self.empty_prefix + turn['user']
        else:
            pv_context =  pv_turn['bspn_gen'] + pv_turn['resp']
            inputs['context'] = pv_context + turn['user']

        if len(inputs['context']) > 900:
            print('len exceeds 900')
            inputs['context'] = inputs['context'][-900:]
         
        return inputs


    def padSeqs_gpt(self, sequences, pad_id, maxlen=None):
        lengths = []
        for x in sequences:
            lengths.append(len(x))

        num_samples = len(sequences)
        seq_mexlen = np.max(lengths)

        # maxlen = 1024
        if seq_mexlen > 1024: # gpt2.n_ctx
            # print('maxlen exceeds 1024')
            maxlen = 1024
        else:
            maxlen = seq_mexlen

        # tokenizer.encode('<|endoftext|>') = ['50256']
        # All labels set to ``-100`` are ignored (masked), the loss is only
        # computed for labels in ``[0, ..., config.vocab_size]`` (from modeling_gpt2.GPT2LMHeadModel)
        
        x = (np.ones((num_samples, maxlen)) * pad_id)
        for idx, s in enumerate(sequences):
            if not len(s):
                print('empty list was found in padSeqs')
            # trunc method = 'pre'
            trunc = s[-maxlen:]
            trunc = np.asarray(trunc)

            # pad method = 'post'
            x[idx, :len(trunc)] = trunc
        return x, lengths


tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
User_Simulator_Reader(tokenizer)
