# coding:utf-8
import os
import torch

class Config(object):
    INTERACTIVE = False

    # cuda
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # user simulator components
    # NLU model
    rule_base_sys_nlu = "./simulator/nlu_model/model/model-test-30-new.pkl"

    # NLG model
    use_sl_simulator = True

    nlg_template = False
    nlg_sample = False
    use_sl_generative = True
    csv_for_generator = './data/multiwoz-master/data/multi-woz/nlg/for_generator.csv'
    generator_debug = True
    topk = 20

    # Policy model
    # save_dir = '/home/liuyajiao/UserSimulator/user-simulator-master/model/save/sl_simulator/oneHot_oldReward_bitMore/' # save_dir = '/home/wyshi/simulator/model/save/sl_simulator/retrieval/oneHot_oldReward_bitMore/'#'/home/wyshi/simulator/model/save/sl_simulator/generative/oneHot_oldReward_bitMore/'
    save_dir = 'model/save/gpt_simulator/must/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    resume = False
    resume_rl_model_dir = './model/save/sl_simulator/oneHot_oldReward_bitMore/best/0_2019-5-19-3-27-15-6-139-1.pkl'

    use_sequicity_for_rl_model = False
    with_bit = True
    with_bit_rep_only = False
    with_bit_more = True
    with_bit_all = False
    if with_bit:
        assert sum([with_bit_rep_only, with_bit_more, with_bit_all]) == 1
    else:
        assert sum([with_bit_rep_only, with_bit_more, with_bit_all]) == 0

    bit_not_used_in_update = True
    use_sent = False
    use_multinomial = False
    use_sent_one_hot = True

    # rl setting
    use_new_reward = False

    warm_start_episodes = 0
    n_episodes = 200000
    uniform_episodes = 40000
    reset_episodes = 40000

    seed = 0
    lr = 1e-4
    discrete_act = True
    init_exp = 0.5 if discrete_act else 0
    reset_exp = 0.1
    final_exp = 0 if discrete_act else 0
    discounted_factor = 0.9 #0.99#0.9
    loose_agents = True
    small_value = 0
    replay = True
    batch_size = 64
    update_every = 64

    # policy model par
    hidden_size = 200
    n_layers = 2
    dropout = 0.3
    max_utt_len = 25
    num_epochs = 30

    # sequicity parameters
    vocab_size = 800
    pretrained_dir = './sequicity_user/models/multiwoz_sys911.pkl'

