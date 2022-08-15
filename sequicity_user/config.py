import logging
import time
import os

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

        self.seed = 0
  
    def init_handler(self, m):
        init_method = {
            'tsdf-sys': self._sys_tsdf_init,
            'tsdf-usr': self._usr_tsdf_init,
            'tsdf-usr_act' : self._usr_act_tsdf_init
        }
        init_method[m]()


    def _sys_tsdf_init(self):
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True

        self.split = (9, 1, 1)
        self.model_path = './sequicity_user/models/multiwoz_sys911.pkl'
        self.result_path = './sequicity_user/results/multiwoz_sys.csv'
        self.vocab_path = './sequicity_user/vocab/vocab-multiwoz_sys.pkl'

        self.data = './data/multiwoz-master/data/multi-woz/rest_sys.json'
        self.entity = './data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = './data/multiwoz-master/data/multi-woz/restaurant_db.json'


        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '/home/liuyajiao/pkgs/glove.6B/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 5
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def _usr_tsdf_init(self):
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True
        self.degree_size = 1

        self.split = (9, 1, 1)
        self.root_dir = "./sequicity_user"
        self.model_path = self.root_dir + '/models/multi_woz_simulator911_goal.pkl'
        self.result_path = self.root_dir + '/results/multi_woz_simulator911_goal.csv'
        self.vocab_path = self.root_dir + '/vocab/vocab-multi_woz_simulator911_goal.pkl'

        self.data = './data/multiwoz-master/data/multi-woz/rest_usr_simulator_goalkey.json'
        self.entity = './data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = './data/multiwoz-master/data/multi-woz/restaurant_db.json'

        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '/home/liuyajiao/pkgs/glove.6B/glove.6B.50d.txt'
        self.batch_size = 32
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def _usr_act_tsdf_init(self):
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True
        self.degree_size = 1

        self.split = (9, 1, 1)
        self.root_dir = "./sequicity_user"
        self.model_path = self.root_dir + '/models/multi_woz_simulator911_act3.pkl'
        self.result_path = self.root_dir + '/results/multi_woz_simulator911_act.csv'
        self.vocab_path = self.root_dir + '/vocab/vocab-multi_woz_simulator911_act3.pkl'

        self.data = './data/multiwoz-master/data/multi-woz/rest_usr_simulator_act.json'
        self.entity = './data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = './data/multiwoz-master/data/multi-woz/restaurant_db.json'

        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '/home/liuyajiao/pkgs/glove.6B/glove.6B.50d.txt'
        self.batch_size = 32
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        log_dir = './sequicity_user/log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        file_handler = logging.FileHandler('./sequicity_user/log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

