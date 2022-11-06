# coding:utf-8

def parse_log(path, mab_path):

    def parse_mab_dict(line):
        mab_dict = {}
        line = line.replace("mab_dict =  {'", '').replace('}', '')
        items = line.split(", '")
        for item in items:
            us, rate = item.split("':")
            mab_dict[us]= rate.strip()
        return mab_dict

    def parse_multi_armed_bandits_dict(line):
        mab_dict = {}
        line = line.replace("multi_armed_bandits_dict =  {'", '').replace('}', '')
        items = line.split(", '")
        for item in items:
            us, rate = item.split("':")
            mab_dict[us]= rate.strip()
        return mab_dict

    success_rates = []
    sample_times = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip()

            if '999/200000' in line:
                print(line)
            if 'multi_armed_bandits_dict = ' in line:
                print('-----'*20)
                multi_armed_bandits_dict = parse_multi_armed_bandits_dict(line)
                times = [multi_armed_bandits_dict['US-AgenT'], multi_armed_bandits_dict['US-AgenR'], 
                         multi_armed_bandits_dict['US-RNNT'], multi_armed_bandits_dict['US-GPT-MWZ']]
                sample_times.append(times)
                print(line)
                # print(times)
            if 'mab_dict = ' in line:
                print(line)
                mab_dict = parse_mab_dict(line)
                succ_rate = [mab_dict['US-AgenT'], mab_dict['US-AgenR'], mab_dict['US-RNNT'], mab_dict['US-GPT-MWZ']]
                # print(mab_dict)
                # print(succ_rate)
                success_rates.append(succ_rate)
            if 'mab_dist = ' in line:
                print(line)

    with open(mab_path, 'w', encoding='utf-8') as fw:
        fw.write(','.join(['US-AgenT', 'US-AgenR', 'US-RNNT', 'US-GPT-MWZ']) + '\n')
        for line in success_rates:
            fw.write(','.join(line) + '\n')

    if 'must' in mab_path:
        mab_path = mab_path.replace('must', 'must_sample_times')
        with open(mab_path, 'w', encoding='utf-8') as fw:
            fw.write(','.join(['US-AgenT', 'US-AgenR', 'US-RNNT', 'US-GPT-MWZ']) + '\n')
            for line in sample_times:
                fw.write(','.join(line) + '\n')

must_log_path = 'model/save/gpt_simulator/old-4k-sd42/rl_must.log'
mab_path = 'simulator_gpt_act/visualization/succ_rates/must.csv'
parse_log(must_log_path, mab_path)

must_log_path = 'model/save/gpt_simulator/old-4k-sd42/rl_uniform.log'
mab_path = 'simulator_gpt_act/visualization/succ_rates/uniform.csv'
parse_log(must_log_path, mab_path)