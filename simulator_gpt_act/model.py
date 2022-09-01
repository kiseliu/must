#torch version 0.4.1
import os
import sys
root_path = os.path.abspath(__file__)
for _ in range(2):
    root_path = os.path.dirname(root_path)
sys.path.append(root_path)

import random
import numpy as np
import logging
import pdb
import csv
# pdb.set_trace()

import torch
import torch.nn as nn
# from torch.optim import Adam, RMSprop
# from torch.autograd import Variable
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from simulator_gpt_act.config import global_config as cfg
from simulator_gpt_act.reader import User_Simulator_Reader
import argparse, time
from simulator_gpt_act.metric import CamRestEvaluator

usr_acts = ["[inform_type]", "[inform_type_change]", "[ask_info]", 
            "[make_reservation]", "[make_reservation_change_time]", "[anything_else]", "[goodbye]"]
usr_slots = ['name', 'area', 'pricerange', 'food', 'people', 'day', 'time', 'address', 'postcode', 'phone', 'reference']

class Model:
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        self.reader = User_Simulator_Reader(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device) 

        self.EV = CamRestEvaluator # evaluator class

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.
        from transformers.Trainer
        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] * cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*0.2)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
        return optimizer, scheduler

    
    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts']:
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor([inputs['context']]).to(self.device)
        return inputs

    def train(self):
        all_batches = self.reader.get_batches('train')
        optimizer, scheduler = self.get_optimizers()

        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
        logging.info("  Learning Rate = %f", cfg.lr)
        logging.info("  Total optimization steps = %d", set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)

        log_inputs = 10
        global_step = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_data_iterator(all_batches)
            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                # if batch_idx > 2:
                #     break
                
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs = self.reader.convert_raw_turn(turn_batch, pv_batch, first_turn)
                    pv_batch = inputs['labels']
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            self.log_first_inputs(inputs)
                            log_inputs -= 1

                        inputs = self.add_torch_input(inputs)
                        outputs = self.model(inputs['contexts_tensor'])
                        loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                            # end of an epoch
                            (epoch_step + 1) == set_stats['num_training_steps_per_epoch']):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1

                            logs = {}  # for tb writer
                            # logging: loss, lr... after certain amount of steps
                            if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                                loss_scalar = (tr_loss - logging_loss) / cfg.report_interval
                                logging_loss = tr_loss
                                logs['loss'] = loss_scalar
                                logging.info('Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                        global_step, epoch_step, loss_scalar))
                                # validate
                                # add to tensorboard...
                                if cfg.evaluate_during_training and loss_scalar < 10:
                                    results = self.validate()
                                    for k, v in results.items():
                                        eval_key = "eval_{}".format(k)
                                        logs[eval_key] = v

                                # if self.tb_writer:
                                #     for k, v in logs.items():
                                #         self.tb_writer.add_scalar(k, v, global_step)
                                # save model... 

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format((time.time()-btm)/60, tr_loss))
            self.save_model(epoch, tr_loss/epoch_step)

    def save_model(self, epoch, loss):
        save_path = os.path.join(cfg.exp_path, 'b{}_g{}_lr{}'.format(cfg.batch_size, cfg.gradient_accumulation_steps, cfg.lr))
        save_path = os.path.join(save_path, 'epoch{}_trloss{:.4f}_gpt2'.format(epoch+1, loss))        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg


    def validate_URURU(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()

        eval_data = self.reader.get_eval_data(data)

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = []
        with torch.no_grad():
            eval_pbar = eval_data
            for dial_idx, dialog in enumerate(eval_pbar):
                print(dial_idx)
                # if dial_idx > 2:
                #     break
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval_URURU(turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    # predict bspn, then generated resp
                    outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=min(context_length + 60, 1024), 
                                                    pad_token_id=cfg.pad_id, 
                                                    eos_token_id=self.tokenizer.encode('EOS_Z2')[0])
                    generated_bs = outputs[0].cpu().numpy().tolist()
                    generated_bs = generated_bs[context_length:]
                    # print('context  ', self.tokenizer.decode(inputs['context']))
                    # print('generated_bs = ', self.tokenizer.decode(generated_bs))

                    inputs['context_tensor_bspn'] = torch.tensor([inputs['context'] + generated_bs + turn['goal']]).to(self.device)
                    context_length = len(inputs['context_tensor_bspn'][0])
                    outputs_db = self.model.generate(input_ids=inputs['context_tensor_bspn'],
                                                    max_length=min(context_length + 80, 1024), temperature=0.7, 
                                                    pad_token_id=cfg.pad_id, 
                                                    eos_token_id=self.tokenizer.encode('EOS_M')[0])
                    generated_ar = outputs_db[0].cpu().numpy().tolist()
                    generated_ar = generated_ar[context_length:]
                    print('generated_ar = ', self.tokenizer.decode(generated_ar))

                    generated_tokens = [self.tokenizer.decode(token) for token in generated_ar]
                    end_idx = generated_tokens.index('EOS_Z6')
                    generated_act = generated_ar[:end_idx]
                    print('generated_act = ', self.tokenizer.decode(generated_act), self.parse_act(self.tokenizer.decode(generated_act)))
                    
                    generated_response = generated_ar[end_idx + 1:-1]
                    # print('generated_ar = ', self.tokenizer.decode(generated_response))
                    turn['bspn_gen'] = generated_bs
                    turn['uda_gen'] = generated_act
                    turn['resp_gen'] = generated_response
                    
                    res = self.reader.wrap_result(turn, generated_response, generated_act, generated_bs, prev_z=pv_turn)
                    result_collection += res

                    # pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    # pv_turn['bspn_gen'] = generated_bs
                    pv_turn['bspn_gen'] = turn['bspan']
                    pv_turn['resp'] = turn['nodelex_resp']

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_act', 'uda', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised']
        with open(cfg.result_path, 'w', newline='') as csvfile:
            csvfile.write(str(cfg))
            writer = csv.DictWriter(csvfile, fieldnames=field)
            csvfile.write('START_CSV_SECTION\n')
            writer.writeheader()
            writer.writerows(result_collection)
        ev = self.EV(result_path=cfg.result_path)
        res = ev.run_metrics()
        return res

    def parse_act(self, act):
        acts = {}

        items = act.split()
        if len(items) == 1:
            acts[items[0]] = []
            return acts
        
        intent = ''
        for i, item in enumerate(items):
            if item in usr_acts:
                intent = item[1:-1]
                if intent not in acts:
                    acts[intent] = []

            if intent and item in usr_slots:
                acts[intent].append(item)
        return acts


    def eval_result_file(self, data='test'):
        ev = self.EV(result_path=cfg.result_path)
        res = ev.run_metrics()
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train')
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

    if args.mode == 'test':
        cfg.gpt_path = cfg.eval_gpt_path

    m = Model('cuda:' + str(cfg.cuda_device))
    if args.mode == 'train':
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
    elif args.mode == 'test':
        # m.load_model()
        m.validate_URURU()
        # m.eval_result_file()
    elif args.mode == 'rl':
        m.load_model('models/multi_woz_simulator911_goal.pkl')
        # m.load_model()
        m.reinforce_tune()
    elif args.mode == 'interact':
        m.load_model()
        m.interactive()


if __name__ == '__main__':
    main()
