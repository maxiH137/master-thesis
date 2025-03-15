import neptune
import argparse
import pandas as pd
import numpy as np
from itertools import product
import sys

# Define the combinations
models = ['deepconvlstm', 'tinyhar']
training_status = [True, False]
numbers = [1, 10, 100]
datasets = ['wear', 'wetlab']
sampling = ['shuffle', 'unbalanced', 'balanced', 'sequential']

# Generate all combinations
combinations = list(product(models, training_status, numbers, datasets, sampling))

class Evaluation():
    def __init__(self):
        pass

    def calculate_accuracy_metrics(self, parser_args):
        self.parser = parser_args
        run_id = parser_args.run_id
        
        old_run = neptune.init_run(
            project=parser_args.project,
            api_token=parser_args.api_token,
            with_id=run_id, 
            mode="read-only"
        )

        self.attacks = old_run["label_attack"].fetch()
        self.args = old_run["args"].fetch()
        
        correct_combination = None
        for combination in combinations:
            if combination[0] == self.args['model'] and combination[1] == self.args['trained'] and combination[2] == self.args['datapoints'] and combination[3] == self.args['dataset'] and combination[4] == self.args['sampling']:
                correct_combination = combination
                break

        number_of_subjects = 18 if self.args['dataset'] == 'wear' else 22
        
        data = []
        
        for attack in self.attacks:
            if attack in parser_args.label_strat_array:
                for sbjs in self.attacks[attack]:
                    if "loso_sbj_" in sbjs:
                        try:
                            lnAcc = round(self.attacks[attack][sbjs]["final_lnAcc"], 2)
                            leAcc = round(self.attacks[attack][sbjs]["final_leAcc"], 2)
                        except:
                            print(f'Error final_lnAcc missing: {attack} {sbjs}')
                            lnAcc = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                            values = np.array(lnAcc.values[:, 1], dtype=float)
                            lnAcc = round(values.mean(), 2)
                            old_run[f"label_attack/{attack}/{sbjs}/final_lnAcc"] = lnAcc
                        
                        config_values = self.args.copy()
                        config_values.update({"Run": run_id, "Attack": attack, "Subject": sbjs, "LnAcc": lnAcc, "LeAcc": leAcc})
                        data.append(config_values)
        
        df = pd.DataFrame(data)
        df.to_csv("output_excel.csv", mode='a', header=not pd.io.common.file_exists("output_excel.csv"), index=False, float_format='%.2f')
        #print("Data saved to output.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_strat_array', nargs='+', default=['llbgSGD', 'bias-corrected', 'iRLG', 'gcd', 'ebi', 'wainakh-simple', 'wainakh-whitebox', 'iDLG', 'analytic', 'yin', 'random'], type=str)
    parser.add_argument('--runs', nargs='+', default=['TAL-168', 'TAL-169', 'TAL-170', 'TAL-244'], type=str)
    parser.add_argument('--run_id', default='TAL-244', type=str)
    parser.add_argument('--project', default='master-thesis-MH/tal', type=str)
    parser.add_argument('--api_token', default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTIwZWQ4Mi03NzUwLTQ0MDUtYmY2Yi1jZDJkNjQyMWY5ZDgifQ==', type=str)
    args = parser.parse_args()
    
    args.runs = ['TAL-304', 'TAL-309', 'TAL-299', 'TAL-302', 'TAL-162', 'TAL-163', 'TAL-164', 'TAL-230', 'TAL-289', 'TAL-292', 'TAL-301', 'TAL-294', 'TAL-165', 'TAL-166', 'TAL-235', 'TAL-167', 'TAL-253', 'TAL-258', 'TAL-274', 'TAL-264', 'TAL-168', 'TAL-169', 'TAL-244', 'TAL-170', 'TAL-314', 'TAL-315', 'TAL-311', 'TAL-312', 'TAL-171', 'TAL-172', 'TAL-255', 'TAL-173', 'TAL-303', 'TAL-306', 'TAL-310', 'TAL-308', 'TAL-215', 'TAL-218', 'TAL-272', 'TAL-219', 'TAL-280', 'TAL-282', 'TAL-288', 'TAL-286', 'TAL-174', 'TAL-175', 'TAL-265', 'TAL-176', 'TAL-300', 'TAL-316', 'TAL-318', 'TAL-307', 'TAL-177', 'TAL-178', 'TAL-278', 'TAL-179', 'TAL-263', 'TAL-266', 'TAL-277', 'TAL-273', 'TAL-180', 'TAL-220', 'TAL-275', 'TAL-221', 'TAL-252', 'TAL-254', 'TAL-257', 'TAL-256', 'TAL-183', 'TAL-184', 'TAL-267', 'TAL-198', 'TAL-320', 'TAL-322', 'TAL-324', 'TAL-323', 'TAL-199', 'TAL-200', 'TAL-283', 'TAL-214', 'TAL-279', 'TAL-281', 'TAL-285', 'TAL-284', 'TAL-222', 'TAL-227', 'TAL-276', 'TAL-228', 'TAL-259', 'TAL-260', 'TAL-262', 'TAL-261', 'TAL-270', 'TAL-271', 'TAL-268', 'TAL-269'] # DeepConv, SGD, Trained, WEAR
    
    args.runs = ['TAL-304', 'TAL-309', 'TAL-299', 'TAL-302', 'TAL-162', 'TAL-163', 'TAL-164', 'TAL-230', 'TAL-289', 'TAL-292', 'TAL-301', 'TAL-294', 'TAL-165', 'TAL-166', 'TAL-235', 'TAL-167', 'TAL-253', 'TAL-258', 'TAL-274', 'TAL-264', 'TAL-168', 'TAL-169', 'TAL-244', 'TAL-170', 'TAL-314', 'TAL-315', 'TAL-311', 'TAL-312', 'TAL-171', 'TAL-172', 'TAL-255', 'TAL-173', 'TAL-303', 'TAL-306', 'TAL-310', 'TAL-308', 'TAL-215', 'TAL-218', 'TAL-272', 'TAL-219', 'TAL-280', 'TAL-282', 'TAL-288', 'TAL-286', 'TAL-174', 'TAL-175', 'TAL-265', 'TAL-176', 'TAL-300', 'TAL-316', 'TAL-318', 'TAL-307', 'TAL-177', 'TAL-178', 'TAL-278', 'TAL-179', 'TAL-263', 'TAL-266', 'TAL-277', 'TAL-273', 'TAL-180', 'TAL-220', 'TAL-275', 'TAL-221', 'TAL-252', 'TAL-254', 'TAL-257', 'TAL-256', 'TAL-183', 'TAL-184', 'TAL-267', 'TAL-198', 'TAL-320', 'TAL-322', 'TAL-324', 'TAL-323', 'TAL-199', 'TAL-200', 'TAL-283', 'TAL-214', 'TAL-279', 'TAL-281', 'TAL-285', 'TAL-284', 'TAL-222', 'TAL-227', 'TAL-276', 'TAL-228', 'TAL-259', 'TAL-260', 'TAL-262', 'TAL-261', 'TAL-270', 'TAL-271', 'TAL-268', 'TAL-269'] # DeepConv, SGD, Trained, WEAR
    for run in args.runs:
        eval = Evaluation()
        args.run_id = run
        eval.calculate_accuracy_metrics(args)
