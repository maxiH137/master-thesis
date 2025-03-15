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
            # Remove non-alphanumeric characters except dots and commas
            cleaned_config = re.sub(r'[^a-zA-Z0-9.,]', '', data['config'])
            split = cleaned_config.split(',')

            # Ensure we have at least 4 elements
            if len(split) >= 4:
                last_part = split[-4:]  # Take only the last 4 values
                cleaned_config = ','.join(last_part)
                joined_terms = ','.join(filter_terms)

                # Ensure the exact match (order and presence)
                if cleaned_config == joined_terms:
                    key = data['config']

                    # Initialize the key in grouped if not present
                    if key not in grouped:
                        grouped[key]['attacks'] = data['attacks']
                        grouped[key]['config'] = key

                    # Ensure length consistency before appending
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
output_file = "output_multFix.txt"
output_dir = "TAL_Run_Plots_multFix"
avg_output_dir = "TAL_Avg_Plots_multFix"
data_output_dir = "DATA_CSV_multFix"

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
    ['5', '5', '0.0', '0.0'],
    ['5', '5', '0.1', '0.5'],
    ['5', '5', '0.1', '1.0'],
    ['5', '5', '0.1', '1.5'],

   # ['2', '2', '0.1', '0.5'],
   # ['2', '5', '0.1', '0.5'],
   # ['5', '2', '0.1', '0.5'],
   # ['5', '5', '0.1', '0.5'],

   # ['2', '2', '0.1', '1.0'],
   # ['2', '5', '0.1', '1.0'],
   # ['5', '2', '0.1', '1.0'],
   # ['5', '5', '0.1', '1.0'],

   # ['2', '2', '0.1', '1.5'],
   # ['2', '5', '0.1', '1.5'],
   # ['5', '2', '0.1', '1.5'],
   # ['5', '5', '0.1', '1.5'],


   # ["wear", '2', '2', '0.0', '0.0'],
   # ["wear", '5', '2', '0.0', '0.0'],
   # ["wear", '5', '2', '0.0', '0.0'],
   # ["wear", '5', '5', '0.0', '0.0'],
   # ["wetlab", '2', '2', '0.0', '0.0'],
   # ["wetlab", '5', '2', '0.0', '0.0'],
   # ["wetlab", '5', '2', '0.0', '0.0'],
   # ["wetlab", '5', '5', '0.0', '0.0'],
   # ["wear", '2', '2', '0.1', '0.5'],
   # ["wear", '5', '2', '0.1', '0.5'],
   # ["wear", '5', '2', '0.1', '0.5'],
   # ["wear", '5', '5', '0.1', '0.5'],

   # ["wetlab", '2', '2', '0.1', '0.5'],
   # ["wetlab", '5', '2', '0.1', '0.5'],
   # ["wetlab", '5', '2', '0.1', '0.5'],
   # ["wetlab", '5', '5', '0.1', '0.5'],
   # ["wear", '2', '2', '0.1', '1.0'],
   # ["wear", '5', '2', '0.1', '1.0'],
   # ["wear", '5', '2', '0.1', '1.0'],
   # ["wear", '5', '5', '0.1', '1.0'],

   # ["wetlab", '2', '2', '0.1', '1.0'],
   # ["wetlab", '5', '2', '0.1', '1.0'],
   # ["wetlab", '5', '2', '0.1', '1.0'],
   # ["wetlab", '5', '5', '0.1', '1.0'],
   # ["wear", '2', '2', '0.1', '1.5'],
   # ["wear", '5', '2', '0.1', '1.5'],
   # ["wear", '5', '2', '0.1', '1.5'],
   # ["wear", '5', '5', '0.1', '1.5'],

   # ["wetlab", '2', '2', '0.1', '1.5'],
   # ["wetlab", '5', '2', '0.1', '1.5'],
   # ["wetlab", '5', '2', '0.1', '1.5'],
   # ["wetlab", '5', '5', '0.1', '1.5'],
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
