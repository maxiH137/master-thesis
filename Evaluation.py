import neptune
import argparse
import pandas as pd
import numpy as np

def main(args):
    
    # Load old run 
    run_id = args.run_id
     
    old_run = neptune.init_run(
        project=args.project,
        api_token=args.api_token,
        with_id=run_id, 
        mode="read-only"
    )

    # Access parameters, metadata, or logged data
    attacks = old_run["label_attack"].fetch()
    
    # Print parameters
    args = old_run["args"].fetch()
    print('Params:', args)

    # Calculate label number accuracy over all subjects for each attack
    ln_all = 0
    sbjs_number = 0
    for attack in attacks:
        for sbjs in attacks[attack]:
            if "loso_sbj_" in sbjs:
                ln_all += attacks[attack][sbjs]["final_lnAcc"]
                sbjs_number += 1
        print('Attack: ' + attack + ' ' +  str(ln_all / sbjs_number))    

    # Validate calculations 
    leAcc = 0
    for attack in attacks:
        for sbj in attacks[attack]:
            if "loso_sbj_" in sbj:
                leAcc = old_run[f"label_attack/{attack}/{sbj}/leAcc"].fetch_values()
                values = np.array(leAcc.values[:, 1], dtype=float)
                print('Attack: ' + attack + ' leAcc for ' + sbj + ': ' + str(values.mean()))
                    
    
    
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

    # Close the run
    old_run["sys/failed"] = False
    old_run.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', default='TAL-165', type=str)
    parser.add_argument('--project', default='master-thesis-MH/tal', type=str)
    parser.add_argument('--api_token', default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTIwZWQ4Mi03NzUwLTQ0MDUtYmY2Yi1jZDJkNjQyMWY5ZDgifQ==', type=str)
    args = parser.parse_args()
    
    main(args)
