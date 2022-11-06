# coding:utf-8
import numpy as np

def calc_avg_std(ls):
    for l in ls:
        avg = sum(l)/len(l)
        squre = [(i-avg)*(i-avg) for i in l]
        std = np.sqrt(sum(squre)/len(l))
        print(avg, std)


in_domain = [[97.5, 54.0, 98.5, 78.0],
            [96.0, 90.0, 98.5, 80.5],
            [30.5, 23.0, 99.0, 75.5],
            [60.5, 51.5, 97.0, 82.0],
            [97.5, 83.5, 94.5, 80.5],
            [97.5, 89.5, 97.0, 82.5]]
out_of_domain = [[72.5, 92.5, 77.0],
                 [97.5, 97.5, 82.0],
                 [35.5, 97.5, 84.0],
                 [59.5, 94.0, 92.0],
                 [97.5, 94.0, 82.5],
                 [96.5, 97.5, 90.0]]

all = [a+b for a, b in zip(in_domain, out_of_domain)]
print(all)

calc_avg_std(in_domain)
print('---'*30)
calc_avg_std(out_of_domain)
print('---'*30)
calc_avg_std(all)
