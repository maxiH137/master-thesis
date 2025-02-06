import neptune
import argparse
import pandas as pd
import numpy as np
from itertools import product


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


def main(args):
    
    # Load old run 
    run_id = args.run_id
     
    old_run = neptune.init_run(
        project=args.project,
        api_token=args.api_token,
        with_id=run_id, 
        #mode="read-only"
    )

    # Access parameters, metadata, or logged data
    attacks = old_run["label_attack"].fetch()
    
    # Print parameters
    args = old_run["args"].fetch()
    print('Params:', args)

    # Calculate label number accuracy over all subjects for each attack
    # Add final_lnAcc to the log, if it is missing
    ln_all = 0
    sbjs_number = 0
    for attack in attacks:
        for sbjs in attacks[attack]:
            if "loso_sbj_" in sbjs:
                try:
                    ln_all += attacks[attack][sbjs]["final_lnAcc"]
                    sbjs_number += 1
                except:
                    print(f'Error final_lnAcc missing: {attack} {sbjs}')
                    lnAcc = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                    values = np.array(lnAcc.values[:, 1], dtype=float)
                    old_run[f"label_attack/{attack}/{sbjs}/final_lnAcc"] = round(values.mean(), 1)

        print('Attack: ' + attack + ' ' +  str(ln_all / sbjs_number))    


    # Load the attck results again, to consider possible changes
    attacks = old_run["label_attack"].fetch()

    # Validate leAcc calculations 
    leAcc = 0
    for attack in attacks:
        for sbj in attacks[attack]:
            if "loso_sbj_" in sbj:
                leAcc = old_run[f"label_attack/{attack}/{sbj}/leAcc"].fetch_values()
                values = np.array(leAcc.values[:, 1], dtype=float)
                print('Attack: ' + attack + ' leAcc for ' + sbj + ': ' + str(values.mean()))
                    
    
    # Validate lnAcc calculations
    lnAcc = 0
    for attack in attacks:
        for sbjs in attacks[attack]:
            if "loso_sbj_" in sbjs:
                lnAcc = old_run[f"label_attack/{attack}/{sbjs}/lnAcc"].fetch_values()
                values = np.array(lnAcc.values[:, 1], dtype=float)
                mean = round(values.mean(), 1)
                final_lnAcc = round(attacks[attack][sbjs]['final_lnAcc'], 1)
                if (mean == final_lnAcc):
                    print(f'Attack: {attack} lnAcc for {sbjs}: Correct => {final_lnAcc} == {mean}')
                else:
                    print(f'Attack: {attack} lnAcc for {sbjs}: Incorrect => {final_lnAcc} != {mean}')

    # Fetch the csv data
    for attack in attacks:
        for sbjs in attacks[attack]:
            if "loso_sbj_" in sbjs:
                if(args.trained):
                    old_run[f"label_attack/{attack}/{sbjs}/data/labelT-csv"].download('neptuneTmpLabels.csv')
                else: 
                    old_run[f"label_attack/{attack}/{sbjs}/data/label-csv"].download('neptuneTmpLabels.csv')
                csv = pd.read_csv('neptuneTmpLabels.csv')

                print(csv.columns)
                if 'idx' not in csv.columns:
                    datapoints = args['datapoints']
                    csv['idx'] = np.repeat(range(len(csv) // datapoints), datapoints)
                    csv.to_csv('neptuneTmpLabels.csv', index=False)


    # Close the run
    # Print the correct combination of this run
    correct_combination = None
    for combination in combinations:
        if combination[0] == args['model'] and combination[1] == args['trained'] and combination[2] == args['datapoints'] and combination[3] == args['dataset'] and combination[4] == args['sampling']:
            correct_combination = combination
            break

    if correct_combination:
        print(f"Correct combination: {correct_combination}")
    else:
        print("No matching combination found.")
        
    old_run["sys/failed"] = False
    old_run.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default='TAL-244', type=str)
    parser.add_argument('--project', default='master-thesis-MH/tal', type=str)
    parser.add_argument('--api_token', default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTIwZWQ4Mi03NzUwLTQ0MDUtYmY2Yi1jZDJkNjQyMWY5ZDgifQ==', type=str)
    args = parser.parse_args()
    
    main(args)
