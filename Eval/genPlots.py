import matplotlib.pyplot as plt
import numpy as np
import re
import os
import collections
import csv

def parse_output_file(filename):
    """Parses the output.txt file and extracts TAL runs with their accuracy values, subject-wise data, and configurations."""
    runs = {}
    current_run = None
    current_config = None
    
    with open(filename, 'r') as file:
        for line in file:
            run_match = re.match(r'Run: (TAL-\d+)', line)
            config_match = re.match(r'Combination: (\(.*\))', line)
            
            if run_match:
                current_run = run_match.group(1)
                runs[current_run] = {'attacks': [], 'ln_acc': [], 'le_acc': [], 'subjects': collections.defaultdict(list), 'config': current_config}
            
            elif config_match:
                current_config = config_match.group(1)
                if current_run:
                    runs[current_run]['config'] = current_config
            
            elif current_run and '|' in line:
                parts = [x.strip() for x in line.split('|')]
                if len(parts) > 3 and parts[0] != "Attack":  # Skip header row
                    attack, ln_acc, le_acc, *subjects = parts
                    runs[current_run]['attacks'].append(attack)
                    runs[current_run]['ln_acc'].append(round(float(ln_acc), 1))
                    runs[current_run]['le_acc'].append(round(float(le_acc), 1))
                    for idx, subj_acc in enumerate(subjects):
                        runs[current_run]['subjects'][f'sbj_{idx}'].append(round(float(subj_acc), 1))
    
    return runs

def plot_tal_run(run_name, data, output_dir):
    """Generates and saves a bar plot for a given TAL run in a specified folder."""
    x = np.arange(len(data['attacks']))
    
    plt.figure(figsize=(20, 10))  # Increased vertical size
    bars1 = plt.bar(x - 0.4, data['ln_acc'], width=0.4, label="LnAcc", color='b', alpha=0.9)
    bars2 = plt.bar(x, data['le_acc'], width=0.4, label="LeAcc", color='r', alpha=0.9)
    
    # Add values on top of each bar
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', ha='center', fontsize=10)
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', ha='center', fontsize=10)
    
    plt.xticks(ticks=x, labels=data['attacks'], rotation=45, ha="right", fontweight='bold')
    plt.yticks(np.arange(0, 105, 10), fontweight='bold')  # Set y-axis ticks with step of 10
    plt.ylabel("Accuracy (%)", fontweight='bold')
    
    # Set title including configuration
    config_text = data['config'] if data['config'] else "No Config Found"
    plt.title(f"{run_name}\nConfig: {config_text}", fontweight='bold')
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    plt.savefig(os.path.join(output_dir, f"{run_name}.png"))  # Save the plot in output folder
    plt.close()

def plot_subjects(run_name, data, output_dir):
    """Generates and saves bar plots for each subject's accuracy values."""
    for subject, acc_values in data['subjects'].items():
        x = np.arange(len(data['attacks']))
        
        plt.figure(figsize=(20, 10))
        try:
            bars = plt.bar(x, acc_values, width=0.5, color='g', alpha=0.9)
        except:
            print(f"Error plotting {run_name} - {subject}: {acc_values}")
            continue
        # Add values on top of each bar
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', ha='center', fontsize=10)
        
        plt.xticks(ticks=x, labels=data['attacks'], rotation=45, ha="right", fontweight='bold')
        plt.yticks(np.arange(0, 105, 10), fontweight='bold')
        plt.ylabel(f"Accuracy (%)", fontweight='bold')
        plt.title(f"{run_name} - {subject}\nConfig: {data['config']}", fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, f"{run_name}_{subject}.png"))
        plt.close()

        # Save data to a .txt file
        with open(os.path.join(data_output_dir, f"{run_name}.csv"), 'a') as f:
            f.write("Run,Subject,Attack,Accuracy\n")
            for attack, acc in zip(data['attacks'], acc_values):
                f.write(f"{run_name},{subject},{attack},{acc}\n")

def group_and_average(runs, filter_terms):
    """Groups runs by matching specific terms in their configuration and averages accuracy values, including subjects."""
    try:
        grouped = collections.defaultdict(lambda: {'attacks': [], 'ln_acc': [], 'le_acc': [], 'subjects': collections.defaultdict(list), 'config': None})
        
        for run, data in runs.items():
            split = data['config'].split(',')
            split = [re.sub(r'[^a-zA-Z0-9]', '', term) for term in split]
            if all(term in split for term in filter_terms):
                key = data['config']
                if key not in grouped:
                    grouped[key]['attacks'] = data['attacks']
                    grouped[key]['config'] = key
                
                if len(data['ln_acc']) == len(grouped[key]['attacks']):
                    grouped[key]['ln_acc'].append(np.array(data['ln_acc']))
                    grouped[key]['le_acc'].append(np.array(data['le_acc']))
                    for subject, values in data['subjects'].items():
                        grouped[key]['subjects'][subject].append(np.array(values))
        
        # Compute averages
        for key, values in grouped.items():
            if values['ln_acc']:  # Ensure data is present before computing mean
                values['ln_acc'] = np.mean(values['ln_acc'], axis=0).tolist()
                values['le_acc'] = np.mean(values['le_acc'], axis=0).tolist()
                for subject in values['subjects']:
                    x = values['subjects'][subject]
                    values['subjects'][subject] = np.mean(values['subjects'][subject], axis=0).tolist()
        
        
        all_ln_acc = [0] * len(grouped[key]['attacks'])
        all_le_acc = [0] * len(grouped[key]['attacks'])
        all_subjects = collections.defaultdict(lambda: [0] * len(grouped[key]['attacks']))

        for key, values in grouped.items():
            all_ln_acc = [a + (b / len(grouped)) for a, b in zip(all_ln_acc, values['ln_acc'])]
            all_le_acc = [a + (b / len(grouped)) for a, b in zip(all_le_acc, values['le_acc'])]
            for subject in values['subjects']:
                all_subjects[subject] = [a + (b / len(grouped)) for a, b in zip(all_subjects[subject], values['subjects'][subject])]
        
        grouped['All']['attacks'] = grouped[key]['attacks']
        grouped['All']['ln_acc'] = all_ln_acc
        grouped['All']['le_acc'] = all_le_acc
        grouped['All']['subjects'] = all_subjects
        grouped['All']['config'] = filter_terms
    except:
        print(f"Error while processing {filter_terms}")
    
    return grouped

import pandas as pd

def save_grouped_data_to_csv(data, filter, filename):
    """Saves the grouped and averaged data to a CSV file using Pandas for Excel compatibility."""
    
    if 'attacks' in data and data['attacks']:  # Ensure data exists
        rows = []
        for i, attack in enumerate(data['attacks']):
            try:
                row = {
                    "Config": filter,
                    "Attack": attack,
                    "LnAcc": f"{data['ln_acc'][i]:.2f}",
                    "LeAcc": f"{data['le_acc'][i]:.2f}"
                }
                for subject in data['subjects']:
                    row[subject] = f"{data['subjects'][subject][i]:.2f}"
                rows.append(row)
           
            except Exception as e:
                print(f"Error while processing {filter} - {attack}: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(f"{filename}_{'_'.join(filter).replace(',', '|')}.csv", index=False)

# Main execution
output_file = "outputV2.txt"
output_dir = "TAL_Run_Plots"
avg_output_dir = "TAL_Avg_Plots"
data_output_dir = "DATA_CSV"

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(avg_output_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

csv_output_file = os.path.join(data_output_dir, "grouped_data")

runs_data = parse_output_file(output_file)

# Generate individual plots
#for run, data in runs_data.items():
    #plot_tal_run(run, data, output_dir)
    #plot_subjects(run, data, output_dir)

# Define filter terms for grouping and process them
filter_sets = [
    ["deepconvlstm","100", 'True', "wear"],
    #["1", "wetlab"],
   # ["deepconvlstm", "100", "wear"],
   # ["deepconvlstm", "100", "wetlab"],
   # ["tinyhar", "100", "wear"],
   # ["tinyhar", "100", "wetlab"],

   # ["1", "wear", 'True'],
   # ["1", "wetlab", 'True'],

   # ["1", "wear", 'False'],
    #["1", "wetlab", 'False'],

   # ["10", "wear", 'True'],
   # ["10", "wetlab", 'True'],

    #["10", "wear", 'False'],
    #["10", "wetlab", 'False'],

    #["100", "wear", 'True'],
    #["100", "wetlab", 'True'],

    #["100", "wear", 'False'],
    #["100", "wetlab", 'False'],
    
    #["100", "wear", 'sequential'],
    #["100", "wear", 'balanced'],
    #["100", "wear", 'unbalanced'],
    #["100", "wear", 'shuffle'],

    #["100", "wetlab", 'sequential'],
    #["100", "wetlab", 'balanced'],
    #["100", "wetlab", 'unbalanced'],
    #["100", "wetlab", 'shuffle'],

    #["deepconvlstm", "True", "100", "wear"],
    #["deepconvlstm", "False", "100", "wear"],
    #["deepconvlstm", "True", "100", "wetlab"],
    #["deepconvlstm", "False", "100", "wetlab"],
    #["tinyhar", "True", "100", "wear"],
    #["tinyhar", "False", "100", "wear"],
    #["tinyhar", "True", "100", "wetlab"],
    #["tinyhar", "False", "100", "wetlab"],
    #["deepconvlstm", "True", "10", "wear"],
    #["deepconvlstm", "False", "10", "wear"],
    #["deepconvlstm", "True", "10", "wetlab"],
    #["deepconvlstm", "False", "10", "wetlab"],
    #["tinyhar", "True", "10", "wear"],
    #["tinyhar", "False", "10", "wear"],
    #["tinyhar", "True", "10", "wetlab"],
    #["tinyhar", "False", "10", "wetlab"],
    #["deepconvlstm", "True", "1", "wear"],
    #["deepconvlstm", "False", "1", "wear"],
    #["deepconvlstm", "True", "1", "wetlab"],
    #["deepconvlstm", "False", "1", "wetlab"],
    #["tinyhar", "True", "1", "wear"],
    #["tinyhar", "False", "1", "wear"],
    #["tinyhar", "True", "1", "wetlab"],
    #["tinyhar", "False", "1", "wetlab"]
]

for filter_terms in filter_sets:
    grouped_data = group_and_average(runs_data, filter_terms)
    for config, data in grouped_data.items():
        if config == 'All':  # Ensure there is valid data before plotting
            try:
                plot_tal_run(f"Avg_{filter_terms}", data, avg_output_dir)
                save_grouped_data_to_csv(data, filter_terms, csv_output_file)
            except Exception as e:
                print(f"Error while plotting {filter_terms}: {e}")
            plot_subjects(f"Avg_{filter_terms}", data, avg_output_dir)

print(f"Plots have been saved in the '{output_dir}' and '{avg_output_dir}' folders.")
