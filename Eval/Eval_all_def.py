import neptune
import argparse
import pandas as pd
import numpy as np
from itertools import product
import sys
import torch

# Define the combinations
models = ['deepconvlstm', 'tinyhar']
training_status = [True, False]
numbers = [100]
datasets = ['wear', 'wetlab']
sampling = ['shuffle', 'unbalanced', 'balanced', 'sequential']
grad_noise = [0.1]
clipping = [0.5, 1.0, 1.5]

# Generate all combinations
combinations = list(product(models, training_status, numbers, datasets, sampling, grad_noise, clipping))

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

    def fix_csv(self, csv, run, label_strat, split_name):
        correct = 0
        wrong = 0   

        recovered_labels_all = csv.iloc[:, 1]
        batchLabels_all = csv.iloc[:, 0]
        lnAcc_list = []
        leAcc_list = []
        for i in range(0, len(csv), 100):
            predicted_labels = recovered_labels_all.values[i:i+100]
            batchLabels = batchLabels_all.values[i:i+100]

            for label in batchLabels:
                label = int(label)
                if label in predicted_labels:
                    correct += 1
                    index = np.where(predicted_labels == label)[0][0]

                    # Delete the first occurrence
                    predicted_labels = np.delete(predicted_labels, index)
                else:
                    wrong +=1
                    
            # Calculate Label Leakage Accuracy for label existence
            batchLabels = torch.tensor(batchLabels)
            predicted_labels = torch.tensor(predicted_labels)

            unique_labelsGT = torch.unique(batchLabels)
            unique_labelsPD = torch.unique(predicted_labels)
            leAcc = 0
            leAccWrong = 0  
            for label in unique_labelsPD:
                if label in unique_labelsGT:
                    leAcc += 1
                elif label not in unique_labelsGT:
                    leAccWrong += 1
            leAcc = leAcc / unique_labelsGT.size()[0] # Label Existence Accuracy
            leAccWrong = leAccWrong / unique_labelsPD.size()[0] # Label Existence Prediction that were predicted, but not actually in the batch
                    
            lnAcc = int((batchLabels.size()[0] - wrong) / batchLabels.size()[0] * 100)

            lnAcc_list.append(lnAcc)
            leAcc_list.append(leAcc)    
            
            correct = 0
            wrong = 0

        run['label_attack' + '/' + str(label_strat) + '/' + split_name + '/leAcc'] = leAcc_list
        run['label_attack' + '/' + str(label_strat) + '/' + split_name + '/lnAcc'] = lnAcc_list

    def calculate_accuracy_metrics(self, parser_args):
        
        # Load old run 
        self.parser = parser_args
        run_id = parser_args.run_id
        
        old_run = neptune.init_run(
            project=parser_args.project,
            api_token=parser_args.api_token,
            with_id=run_id, 
            #mode="read-only"
        )

        # Access parameters, metadata, or logged data
        self.attacks = old_run["label_attack"].fetch()
        
        # Print parameters
        self.args = old_run["args"].fetch()
        self.args['grad_noise'] = old_run[f"label_attack/{'gcd'}/{'loso_sbj_0'}/grad_noise"].fetch()
        self.args['clipping'] = old_run[f"label_attack/{'gcd'}/{'loso_sbj_0'}/clipping"].fetch()
        
        print('Params:', self.args)

        correct_combination = None
        for combination in combinations:
            if combination[0] == self.args['model'] and combination[1] == self.args['trained'] and combination[2] == self.args['datapoints'] and combination[3] == self.args['dataset'] and combination[4] == self.args['sampling'] and combination[5] == self.args['grad_noise'] and combination[6] == self.args['clipping']:
                correct_combination = combination
                break
        
        if correct_combination == None:
            print(f"No matching combination found for run {run_id}.")
            return

        for attack in self.attacks:
            if attack in parser_args.label_strat_array:
                for sbjs in self.attacks[attack]:
                    if "loso_sbj_" in sbjs:
                        le_check = 0
                        ln_check = 0
                        try:
                            ln_check += self.attacks[attack][sbjs]["final_lnAcc"]
                            ln_listCheck = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()

                            timestamps = ln_listCheck.values[:,2]
                            current_time = pd.Timestamp.now()
                            timestamp = pd.to_datetime(timestamps[0])

                            if (current_time - timestamp).days < 2:
                                raise Exception('Error')
                        except:
                            print(f'Error final_lnAcc missing: {attack} {sbjs} {run_id}')
                            try:
                                old_run[f"label_attack/{attack}/{sbjs}/data/label-csv"].download('neptuneTmpLabels.csv')
                                with open('neptuneTmpLabels.csv', 'r') as file:
                                    data = file.readlines()

                                data[0] = data[0].strip() + ',Batch\n'
                                with open('neptuneTmpLabels.csv', 'w') as file:
                                    file.writelines(data)

                                csv = pd.read_csv('neptuneTmpLabels.csv')
                                self.fix_csv(csv, old_run, attack, sbjs)
                                print(f'Fixed: {attack} {sbjs} {run_id}')
                            except:
                                print(f'Multiple Errors in lnAcc missing: {attack} {sbjs} {run_id}')

                            old_run.wait()
                            self.lnAccFix = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                            print(f'Error lnAcc missing: {attack} {sbjs}')

                            self.attacks = old_run["label_attack"].fetch()    
                            self.ln_values = np.array(self.lnAccFix.values[:, 1], dtype=float)
                            new_final_lnAcc = round(self.ln_values.mean(), 1)
                            old_run[f"label_attack/{attack}/{sbjs}/final_lnAcc"] = new_final_lnAcc
                            
                        try:
                            le_check += self.attacks[attack][sbjs]["final_leAcc"]
                            ln_listCheck = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()

                            timestamps = ln_listCheck.values[:,2]
                            current_time = pd.Timestamp.now()
                            timestamp = pd.to_datetime(timestamps[0])

                            if (current_time - timestamp).days < 2:
                                raise Exception('Error')
                        except:
                            print(f'Error final_leAcc missing: {attack} {sbjs} {run_id}')
                            self.leAccFix = old_run[f"label_attack/{attack}/{sbjs}/leAcc"].fetch_values()
                            self.le_values = np.array(self.leAccFix.values[:, 1], dtype=float)
                            old_run[f"label_attack/{attack}/{sbjs}/final_leAcc"] = round(self.le_values.mean(), 1)                   

        old_run.wait()
        self.attacks = old_run["label_attack"].fetch()
        # Redirect stdout to a file
        original_stdout = sys.stdout
        with open('output_sgd_defense.txt', 'a') as f:
            sys.stdout = f

            print('Run:', run_id)
            if correct_combination:
                print(f"Combination: {correct_combination}")
            else:
                print("No matching combination found.")

            number_of_subjects = 0
            if (self.args['dataset'] == 'wear'):
                number_of_subjects = 18
            else:
                number_of_subjects = 22
            
            subjects = ""
            for i in range(number_of_subjects):
                sbj = f"sbj_{i}"
                subjects += f"{sbj:<10} | "

            print(f"{'Attack':<20} | {'LnAcc':<10} | {'LeAcc':<10} | {subjects}")
            print('-' * 400)
            # Calculate label number accuracy over all subjects for each attack
            # Add final_lnAcc to the log, if it is missing
            self.ln_all = 0
            self.le_all = 0
            sbjs_number = 0
            self.ln_subjects = []
            self.le_subjects = []
            for attack in self.attacks:
                if attack in parser_args.label_strat_array:
                    for sbjs in self.attacks[attack]:
                        if "loso_sbj_" in sbjs:
                            try:
                                self.ln_all += self.attacks[attack][sbjs]["final_lnAcc"]
                                self.le_all += self.attacks[attack][sbjs]["final_leAcc"]
                                self.ln_subjects.append(f"{self.attacks[attack][sbjs]['final_lnAcc']:<10.2f}")
                                self.le_subjects.append(f"{self.attacks[attack][sbjs]['final_leAcc']:<10.2f}")
                                sbjs_number += 1
                            except:
                                print(f'Error final_lnAcc missing: {attack} {sbjs}')
                                self.lnAcc = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                                self.values = np.array(self.lnAcc.values[:, 1], dtype=float)
                                old_run[f"label_attack/{attack}/{sbjs}/final_lnAcc"] = round(self.values.mean(), 1)
                                

                    print(f"{attack:<20} | {self.ln_all / sbjs_number:<10.2f} | {self.le_all / sbjs_number:<10.2f} | {' | '.join(map(str, self.ln_subjects)):<10}")
                    
                   
                    self.ln_all = 0 
                    self.le_all = 0 
                    sbjs_number = 0
                    self.ln_subjects = []
                    self.le_subjects = []
            print('-' * 400)

            # Restore stdout
            sys.stdout = original_stdout

        old_run["sys/failed"] = False
        old_run.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_strat_array', nargs='+', default=['llbgSGD', 'bias-corrected', 'iRLG', 'gcd', 'ebi', 'wainakh-simple', 'wainakh-whitebox', 'iDLG', 'analytic', 'yin', 'random'], type=str)
   # parser.add_argument('--runs', nargs='+', default=['TAL-253', 'TAL-258', 'TAL-264', 'TAL-274'], type=str) # DeepConv, SGD, Untrained, WEAR
    parser.add_argument('--runs', nargs='+', default=[], type=str) # DeepConv, SGD, Trained,   WEAR
    parser.add_argument('--run_id', default='TAL-244', type=str)
    parser.add_argument('--project', default='master-thesis-MH/tal', type=str)
    parser.add_argument('--api_token', default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTIwZWQ4Mi03NzUwLTQ0MDUtYmY2Yi1jZDJkNjQyMWY5ZDgifQ==', type=str)
    args = parser.parse_args()
    
    eval_results = []
     
    
    
    args.runs = [f'TAL-{i}' for i in range(430, 521)]
    args.runs += ['TAL-528', 'TAL-546', 'TAL-551', 'TAL-565', 'TAL-570', 'TAL-583', 'TAL-598', 'TAL-605','TAL-1447',]
    if 'TAL-450' in args.runs:
        args.runs.remove('TAL-450')
    #if 'TAL-425' in args.runs:
    #    args.runs.remove('TAL-425')
    if 'TAL-487' in args.runs:
        args.runs.remove('TAL-487')
    if 'TAL-497' in args.runs:
        args.runs.remove('TAL-497')
    if 'TAL-519' in args.runs:
        args.runs.remove('TAL-519')

    for run in args.runs:
        eval = Evaluation()
        args.run_id = run
        #args.label_strat_array = ['llbgSGD', 'bias-corrected', 'iRLG', 'gcd', 'wainakh-simple', 'wainakh-whitebox']
        eval.calculate_accuracy_metrics(args)
        eval_results.append(eval)