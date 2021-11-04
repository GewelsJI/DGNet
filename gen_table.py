import os

root = './result/'


for _data in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    print('+---------+---------+----------+-----------+--------+-------+--------+--------+--------+--------+-------+')
    print('|  Method | Dataset | Smeasure | wFmeasure |  MAE   | adpEm | meanEm | maxEm  | adpFm  | meanFm | maxFm |')
    print('+---------+---------+----------+-----------+--------+-------+--------+--------+--------+--------+-------+')
    for exp_name in os.listdir(root):
        txt_root = os.path.join(root, exp_name, 'evaluation_results.log')

        with open(txt_root) as f:
            lines = f.readlines()
            for _line in lines:
                if _data in _line and exp_name in _line and 'prediction' not in _line:
                    print(_line)