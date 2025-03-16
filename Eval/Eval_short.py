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

# Print combinations
for idx, combination in enumerate(combinations):
    print(f"{idx}: {combination}")

class lnAccSubjects():
    def __init__(self, lnAcc_val, name, single_subject_values):
        self.lnAcc = lnAcc_val
        self.attack = name
        self.single_subject_values = single_subject_values

class Evaluation():
    def __init__(self):
        pass

    def calculate_accuracy_metrics(self, parser_args):
        
        # Load old run 
        self.parser = parser_args
        run_id = parser_args.run_id
        
        old_run = neptune.init_run(
            project=parser_args.project,
            api_token=parser_args.api_token,
            with_id=run_id, 
           # mode="read-only"
        )

        # Access parameters, metadata, or logged data
        self.attacks = old_run["label_attack"].fetch()
        
        # Print parameters
        self.args = old_run["args"].fetch()
        print('Params:', self.args)

        correct_combination = None
        for combination in combinations:
            if combination[0] == self.args['model'] and combination[1] == self.args['trained'] and combination[2] == self.args['datapoints'] and combination[3] == self.args['dataset'] and combination[4] == self.args['sampling']:
                correct_combination = combination
                break
        

        # Redirect stdout to a file
        original_stdout = sys.stdout
        with open('output.txt', 'a') as f:
            sys.stdout = f

            print('Run:', run_id)
            if correct_combination:
                print(f"Combination: {correct_combination}")
            else:
                print("No matching combination found.")
            print(f"{'Attack':<20} | {'LnAcc':<10} | {'LeAcc':<10}")
            print('-' * 45)
            # Calculate label number accuracy over all subjects for each attack
            # Add final_lnAcc to the log, if it is missing
            self.ln_all = 0
            self.le_all = 0
            sbjs_number = 0
            for attack in self.attacks:
                if attack in parser_args.label_strat_array:
                    for sbjs in self.attacks[attack]:
                        if "loso_sbj_" in sbjs:
                            try:
                                self.ln_all += self.attacks[attack][sbjs]["final_lnAcc"]
                                sbj_le = np.array(old_run[f"label_attack/{attack}/{sbjs}/leAcc"].fetch_values().values[:, 1], dtype=float)
                                sbj_le = sbj_le.mean() * 100
                                old_run[f"label_attack/{attack}/{sbjs}/final_leAcc"] = round(sbj_le, 1)
                                self.le_all += self.attacks[attack][sbjs]["final_leAcc"]
                                sbjs_number += 1
                            except:
                                print(f'Error final_lnAcc missing: {attack} {sbjs}')
                                self.lnAcc = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                                self.values = np.array(self.lnAcc.values[:, 1], dtype=float)
                                old_run[f"label_attack/{attack}/{sbjs}/final_lnAcc"] = round(self.values.mean(), 1)

                    print(f"{attack:<20} | {self.ln_all / sbjs_number:<10.2f} | {self.le_all / sbjs_number:<10.2f}")
                    sbjs_number = 0
                    self.ln_all = 0 
                    self.le_all = 0 
            print('---------------------------------------------------------------------------------------------------------')

            # Restore stdout
            sys.stdout = original_stdout

        """
        # Load the attck results again, to consider possible changes
        self.attacks = old_run["label_attack"].fetch()

        # Validate leAcc calculations 
        self.leAcc = 0
        for attack in self.attacks:
            if attack in parser_args.label_strat_array:
                for sbj in self.attacks[attack]:
                    if "loso_sbj_" in sbj:
                        self.leAcc = old_run[f"label_attack/{attack}/{sbj}/leAcc"].fetch_values()
                        self.values = np.array(self.leAcc.values[:, 1], dtype=float)
                        print('Attack: ' + attack + ' leAcc for ' + sbj + ': ' + str(self.values.mean()))
                        
        
        # Validate lnAcc calculations
        self.lnAcc = 0
        self.lnAcc_allSubjects = 0
        self.lnAcc_allSubjects_list = []
        self.single_subject_values = []
        for attack in self.attacks:
            if attack in parser_args.label_strat_array:
                for sbjs in self.attacks[attack]:
                    if "loso_sbj_" in sbjs:
                        self.lnAcc = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                        self.values = np.array(self.lnAcc.values[:, 1], dtype=float)
                        self.mean = round(self.values.mean(), 1)
                        self.final_lnAcc = round(self.attacks[attack][sbjs]['final_lnAcc'], 1)
                        if (self.mean == self.final_lnAcc):
                            print(f'Attack: {attack} lnAcc for {sbjs}: Correct => {self.final_lnAcc} == {self.mean}')
                        else:
                            print(f'Attack: {attack} lnAcc for {sbjs}: Incorrect => {self.final_lnAcc} != {self.mean}')
                        self.lnAcc_allSubjects += self.mean
                        self.single_subject_values.append(self.mean)
                        
                self.lnAcc_allSubjects /= (len(self.attacks[attack]) - 1)
                self.lnAcc_allSubjects_list.append(lnAccSubjects(self.lnAcc_allSubjects, attack, self.single_subject_values))
                self.single_subject_values = []
            self.lnAcc_allSubjects = 0

        # Close the run
        # Print the correct combination of this run
        correct_combination = None
        for combination in combinations:
            if combination[0] == self.args['model'] and combination[1] == self.args['trained'] and combination[2] == self.args['datapoints'] and combination[3] == self.args['dataset'] and combination[4] == self.args['sampling']:
                correct_combination = combination
                break

        if correct_combination:
            print(f"Correct combination: {correct_combination}")
        else:
            print("No matching combination found.")
        """

        old_run["sys/failed"] = False
        old_run.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_strat_array', nargs='+', default=['llbgSGD', 'bias-corrected', 'iRLG', 'gcd', 'ebi', 'wainakh-simple', 'wainakh-whitebox', 'iDLG', 'analytic', 'yin', 'random'], type=str)
   # parser.add_argument('--runs', nargs='+', default=['TAL-253', 'TAL-258', 'TAL-264', 'TAL-274'], type=str) # DeepConv, SGD, Untrained, WEAR
    parser.add_argument('--runs', nargs='+', default=['TAL-168', 'TAL-169', 'TAL-170', 'TAL-244'], type=str) # DeepConv, SGD, Trained,   WEAR
    parser.add_argument('--run_id', default='TAL-244', type=str)
    parser.add_argument('--project', default='master-thesis-MH/tal', type=str)
    parser.add_argument('--api_token', default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTIwZWQ4Mi03NzUwLTQ0MDUtYmY2Yi1jZDJkNjQyMWY5ZDgifQ==', type=str)
    args = parser.parse_args()
    
    eval_results = []
     
    args.runs = ['TAL-304', 'TAL-309', 'TAL-299', 'TAL-302', 'TAL-162', 'TAL-163', 'TAL-164', 'TAL-230', 'TAL-289', 'TAL-292', 'TAL-301', 'TAL-294', 'TAL-165', 'TAL-166', 'TAL-235', 'TAL-167', 'TAL-253', 'TAL-258', 'TAL-274', 'TAL-264', 'TAL-168', 'TAL-169', 'TAL-244', 'TAL-170', 'TAL-314', 'TAL-315', 'TAL-311', 'TAL-312', 'TAL-171', 'TAL-172', 'TAL-255', 'TAL-173', 'TAL-303', 'TAL-306', 'TAL-310', 'TAL-308', 'TAL-215', 'TAL-218', 'TAL-272', 'TAL-219', 'TAL-280', 'TAL-282', 'TAL-288', 'TAL-286', 'TAL-174', 'TAL-175', 'TAL-265', 'TAL-176', 'TAL-300', 'TAL-316', 'TAL-318', 'TAL-307', 'TAL-177', 'TAL-178', 'TAL-278', 'TAL-179', 'TAL-263', 'TAL-266', 'TAL-277', 'TAL-273', 'TAL-180', 'TAL-220', 'TAL-275', 'TAL-221', 'TAL-252', 'TAL-254', 'TAL-257', 'TAL-256', 'TAL-183', 'TAL-184', 'TAL-267', 'TAL-198', 'TAL-320', 'TAL-322', 'TAL-324', 'TAL-323', 'TAL-199', 'TAL-200', 'TAL-283', 'TAL-214', 'TAL-279', 'TAL-281', 'TAL-285', 'TAL-284', 'TAL-222', 'TAL-227', 'TAL-276', 'TAL-228', 'TAL-259', 'TAL-260', 'TAL-262', 'TAL-261', 'TAL-270', 'TAL-271', 'TAL-268', 'TAL-269'] # DeepConv, SGD, Trained, WEAR
    for run in args.runs:
        eval = Evaluation()
        args.run_id = run
        #args.label_strat_array = ['llbgSGD', 'bias-corrected', 'iRLG', 'gcd', 'wainakh-simple', 'wainakh-whitebox']
        eval.calculate_accuracy_metrics(args)
        eval_results.append(eval)