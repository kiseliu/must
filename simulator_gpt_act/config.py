import logging
import time

class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 0
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.6

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'
        self.dataset = 'unknown'

        self.seed = 11
        self.cuda = True

        self.split = (9, 1, 1)
        self.root_dir = "."
        # self.gpt_path = '/home/liuyajiao/pretrained-models/distilgpt2/'
        self.gpt_path = 'simulator_gpt_act/models/agenda/b4_g16_lr0.001/epoch50_trloss0.1173_gpt2'
        self.eval_gpt_path = './simulator_gpt_act/models/mwz/b4_g16_lr0.001/epoch58_trloss0.2602_gpt2'
        # self.eval_gpt_path = './simulator_gpt_act/models/mwz/b4_g16_lr0.001/epoch60_trloss0.2484_gpt2'
        self.model_path = self.root_dir + '/models/multi_woz_simulator911_goal.pkl'
        self.result_path = self.root_dir + '/results/multi_woz_simulator_gpt_oracle.csv'
        self.vocab_path = self.root_dir + '/vocab/vocab-multi_woz_simulator911_goal.pkl'
        self.exp_path = './simulator_gpt_act/models/mwz/'

        # self.data = 'evaluation_results/simulated_agenda_dataset/rest_usr_simulator_goal_agenda.json'
        self.data = 'data/multiwoz-master/data/multi-woz/rest_usr_simulator_goal_mwz.json'
        self.encoded_file_path = './data/multiwoz-master/user_simulator_mwz.json'
        self.entity = './data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = './data/multiwoz-master/db/restaurant_db.json'

        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = True
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

        # training settings
        self.lr = 1e-3
        self.warmup_steps = 2000
        self.weight_decay = 0.0 
        self.gradient_accumulation_steps = 16
        self.batch_size = 4

        self.label_smoothing = .0
        self.lr_decay = 0.5
        self.epoch_num = 60
        self.early_stop_count = 5
        self.weight_decay_count = 3
        self.teacher_force = 100
        self.multi_acts_training = False
        self.multi_act_sampling_num = 1
        self.valid_loss = 'score'
        self.report_interval = 200 # 485 for bs 128
        self.evaluate_during_training = False # evaluate during training

        self.degree_size = 5

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

