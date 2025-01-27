"""
Example script to run attacks in this repository directly without simulation.
This can be useful if you want to check a model architecture and model gradients computed/defended in some shape or form
against some of the attacks implemented in this repository, without implementing your model into the simulation.

All caveats apply. Make sure not to leak any unexpected information.
"""
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import breaching
import neptune
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import sys
from pprint import pprint
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from utils.os_utils import Logger, load_config
from utils.torch_utils import fix_random_seed
from models.DeepConvLSTM import DeepConvLSTM
from models.TinyHAR import TinyHAR
from utils.torch_utils import init_weights, worker_init_reset_seed, InertialDataset
from torch.utils.data import DataLoader
from opacus import layers, optimizers
from Samplers import UnbalancedSampler, BalancedSampler
from Defense_Sampler import DefenseSampler
from DPrivacy import DPrivacy
from neptune.types import File

import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Opt-in to the future behavior to prevent the warning
pd.set_option('future.no_silent_downcasting', True)

class data_config_inertial:
    modality = "vision"
    size = (1_281_167,)
    classes = 19
    shape = (1, 50, 12)
    normalize = True
    
    def __init__(self):
        self.attributes = {}  
    
    def __getitem__(self, key):
        return self.attributes.get(key)
    
    def __setitem__(self, key, value):
        self.attributes[key] = value

class Leakage():
    def __init__(self, args = None, run = None, config = None):
        self.args = args
        self.run = run
        self.config = config
        self.dpri = DPrivacy(multiplier=0.1, clip=0.1)
    
    def main(self, args):
        
        if args.neptune:
            run = neptune.init_run(
            project="master-thesis-MH/tal",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTIwZWQ4Mi03NzUwLTQ0MDUtYmY2Yi1jZDJkNjQyMWY5ZDgifQ=="
            )
        else:
            run = None

        config = load_config(args.config)
        config['init_rand_seed'] = args.seed
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.neptune:
            run_id = run["sys/id"].fetch()
        else:
            run_id = args.run_id
        
        
        log_dir = os.path.join('logs', config['name'], '_' + run_id)
        sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

        # save the current cfg
        with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
            pprint(config, stream=fid)
            fid.flush()
        
        if args.neptune:
            run['config_name'] = args.config
            run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
            run['args/trained'] = args.trained
            run['args/model'] = config['name']
            run['args/dataset'] = config['dataset_name']
            run['args/sampling'] = args.sampling
            run['args/datapoints'] = config['loader']['train_batch_size']
            run['args/classes'] = config['dataset']['num_classes']
            run['args/label_strat_array'] = args.label_strat_array
        
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup = dict(device=torch.device("cpu"), dtype=torch.float)
        
        trained = args.trained
        strat_array = args.label_strat_array
        grads = []

        lnAcc_all = 0
        for label_strat in strat_array:
            for i, anno_split in enumerate(config['anno_json']):
                with open(anno_split) as f:
                    file = json.load(f)
                anno_file = file['database']
                config['labels'] = ['null'] + list(file['label_dict'])
                config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
                train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
                val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

                print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
                if args.eval_type == 'split':
                    name = 'split_' + str(i)
                elif args.eval_type == 'loso':
                    name = 'sbj_' + str(i)
                config['dataset']['json_anno'] = anno_split
                if config['name'] == 'tadtr':
                    config['dataset']['json_info'] = config['info_json'][i]
            
                split_name = config['dataset']['json_anno'].split('/')[-1].split('.')[0]
                # load train and val inertial data
                train_data, val_data = np.empty((0, config['dataset']['input_dim'] + 2)), np.empty((0, config['dataset']['input_dim'] + 2))
                for t_sbj in train_sbjs:
                    t_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'],  t_sbj + '.csv'), index_col=False).replace({"label": config['label_dict']}).infer_objects(copy=False).fillna(0).to_numpy()
                    train_data = np.append(train_data, t_data, axis=0)
                for v_sbj in val_sbjs:
                    v_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'],  v_sbj + '.csv'), index_col=False).replace({"label": config['label_dict']}).infer_objects(copy=False).fillna(0).to_numpy()
                    val_data = np.append(val_data, v_data, axis=0)

                # define inertial datasets
                train_dataset = InertialDataset(train_data, config['dataset']['window_size'], config['dataset']['window_overlap'])
                test_dataset = InertialDataset(val_data, config['dataset']['window_size'], config['dataset']['window_overlap'])

                # define dataloaders
                unbalanced_sampler = UnbalancedSampler(test_dataset, random.randint(0, config['dataset']['num_classes']), random.randint(0, config['dataset']['num_classes']))
                balanced_sampler = BalancedSampler(test_dataset)
                defense_sampler = DefenseSampler(test_dataset, random.randint(0, config['dataset']['num_classes']))
                
                config['init_rand_seed'] = args.seed
                rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True) 
                train_loader = DataLoader(train_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                val_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                unbalanced_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=unbalanced_sampler, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                balanced_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=balanced_sampler, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                defense_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=defense_sampler, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                
                defense_loader.name = 'defense'
                unbalanced_loader.name = 'unbalanced' 
                balanced_loader.name = 'balanced'
                train_loader.name = 'train'
                val_loader.name = 'val'
                                
                if 'tinyhar' in config['name'] and trained:
                    args.resume = 'saved_models/' + config['dataset_name'] + f'/tinyhar/epoch_100_loso_sbj_{i}.pth.tar'
                if 'deepconvlstm' in config['name'] and trained:
                    args.resume = 'saved_models/' + config['dataset_name'] + f'/deepconvlstm/epoch_100_loso_sbj_{i}.pth.tar'

                if 'deepconvlstm' in config['name']:
                    model = DeepConvLSTM(
                        config['dataset']['input_dim'], config['dataset']['num_classes'] + 1, train_dataset.window_size,
                        config['model']['conv_kernels'], config['model']['conv_kernel_size'], 
                        config['model']['lstm_units'], config['model']['lstm_layers'], config['model']['dropout']
                        )
                    print("Number of learnable parameters for DeepConvLSTM: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    
                    # define criterion and optimizer
                    opt = torch.optim.Adam(model.parameters(), lr=config['train_cfg']['lr'], weight_decay=config['train_cfg']['weight_decay'])
                    
                    if args.resume and trained:
                        if os.path.isfile(os.getcwd() + '\\' + args.resume):
                            checkpoint = torch.load(args.resume, map_location = device) # loc: storage.cuda(config['device'])
                        
                            model.load_state_dict(checkpoint['state_dict'])
                            opt.load_state_dict(checkpoint['optimizer'])
                            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(args.resume, checkpoint['epoch']))
                            del checkpoint
                        else:
                            print("=> no checkpoint found at '{}'".format(args.resume))
                            return
                    else:
                        model = init_weights(model, config['train_cfg']['weight_init'])
                        pass
                    
                if 'tinyhar' in config['name']:
                    model = TinyHAR((config['loader']['train_batch_size'], 1, train_dataset.window_size, config['dataset']['input_dim']), config['dataset']['num_classes'] + 1, 
                                     config['model']['conv_kernels'], 
                                     config['model']['conv_layers'], 
                                     config['model']['conv_kernel_size'], 
                                     dropout=config['model']['dropout'], feature_extract=config['model']['feature_extract'])
                    
                    if args.resume and trained:
                        if os.path.isfile(os.getcwd() + '\\' + args.resume):
                            checkpoint = torch.load(args.resume, map_location = device) # loc: storage.cuda(config['device'])
                        
                            model.load_state_dict(checkpoint['state_dict'])
                            #opt.load_state_dict(checkpoint['optimizer'])
                            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(args.resume, checkpoint['epoch']))
                            del checkpoint
                        else:
                            print("=> no checkpoint found at '{}'".format(args.resume))
                            return
                    else:
                        model = init_weights(model, config['train_cfg']['weight_init'])
                
                
                if config['name'] == 'ResNet':
                    model = torchvision.models.resnet18(pretrained=trained)
                    model.conv1 = nn.Conv2d(1, 
                                            model.conv1.out_channels, 
                                            kernel_size = model.conv1.kernel_size, 
                                            stride = model.conv1.stride, 
                                            padding = model.conv1.padding, 
                                            bias = model.conv1.bias)
                    
                    model.fc = nn.Linear(model.fc.in_features, config['dataset']['num_classes'] + 1) 
                    model.name = 'ResNet'
                    model = init_weights(model, config['train_cfg']['weight_init'])
                    
                model.train()
                    
                loss_fn = torch.nn.CrossEntropyLoss()
                
                
                if args.neptune:
                    run['label_attack' + '/' + str(label_strat) + '/attack_config_name'] = args.attack
                
                if trained:
                    with open('CSV/' + 'labelsT_' + str(config['name']) + '_' + str(label_strat) + '.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['GT', 'Pred', 'Sbj'])

                    with open('CSV/' + 'gradientsT_' + str(config['name']) + '_' + str(label_strat) + '.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            for g in range(config['dataset']['num_classes'] + 1):
                                grads.append('G' + str(g))
                            writer.writerow(grads)
                else:
                    with open('CSV/' + 'labels_' + str(config['name']) + '_' + str(label_strat) + '.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['GT', 'Pred', 'Sbj'])

                    with open('CSV/' + 'gradients_' + str(config['name']) + '_' + str(label_strat) + '.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            for g in range(config['dataset']['num_classes'] + 1):
                                grads.append('G' + str(g))
                            writer.writerow(grads)

                # This is the attacker:
                #cfg_case = breaching.get_case_config('4_fedavg_small_scale_har') # user with local_updates
                cfg_case = breaching.get_case_config('8_industry_scale_fl_har') #  multiuser_aggregate
                
                if args.sampling == 'shuffle':
                    cfg_case['data']['shuffle'] = 'shuffle'
                elif args.sampling == 'balanced':
                    cfg_case['data']['shuffle'] = 'balanced'
                elif args.sampling == 'unbalanced':
                    cfg_case['data']['shuffle'] = 'unbalanced'
                elif args.sampling == 'sequential':
                    cfg_case['data']['shuffle'] = 'sequential'
                
                
                cfg_attack = breaching.get_attack_config(args.attack)
                cfg_attack['optim']['max_iterations'] = int(config['attack']['iterations'])
                cfg_attack['label_strategy'] = label_strat
                
                
                #model_br, loss_fn_br = breaching.cases.construct_model(breaching.config.case.model, breaching.cfg.case.data, False)
                server_br = breaching.cases.construct_server(model, loss_fn, cfg_case, setup)
                model = server_br.vet_model(model)

                # Instantiate user and attacker
                user = breaching.cases.construct_user(model, loss_fn, cfg_case, setup, dataset=test_dataset)
                 
                if args.neptune:
                    log_dir_atk = os.path.join('logs', args.attack, '_' + run_id)
                    sys.stdout = Logger(os.path.join(log_dir_atk, 'log.txt'))
                    with open(os.path.join(log_dir_atk, 'cfg_atk.txt'), 'w') as fid:
                        pprint(cfg_attack, stream=fid)
                        fid.flush()
                    
                    run['label_attack' + '/' + str(label_strat) + '/attack_config'].upload(os.path.join(log_dir_atk, 'cfg_atk.txt'))
                    
                attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

                # ## Simulate an attacked FL protocol
                # Server-side computation:
                
                metadata = data_config_inertial()
                metadata.shape = (1, 50, config['dataset']['input_dim'])
                metadata.classes = config['dataset']['num_classes'] + 1
                metadata['task'] = 'classification'
                
                
                recovered_labels_all = []
                batchLabels_all = []
                    
                # Summarize startup:
                breaching.utils.overview(server_br, user, attacker)

                # Simulate a simple FL protocol
                shared_user_data, payloads, true_user_data = server_br.run_protocol(user)
                
                
                #Subset = torch.utils.data.Subset(train_dataset, shared_user_data['indices'])
                
                # Run an attack using only payload information and shared data
                
                for idx, data_block in enumerate(user.dataloader):
                    
                    shared_user_data, payloads, true_user_data = server_br.run_protocol(user)
                
                    user.num_data_points = 100
                    user.num_data_per_local_update_step = 10
                    user.num_local_updates = 10
                    shared_user_data[0]['metadata']['num_data_points'] = user.num_data_points
                    shared_user_data[0]['metadata']['local_hyperparams']['steps'] = user.num_local_updates
                    shared_user_data[0]['metadata']['local_hyperparams']['data_per_step'] = user.num_data_per_local_update_step
                    
                    labels = data_block['labels']
                    data = data_block['inputs']
                    onedimdata = data.unsqueeze(1)
                    
                    output = model(onedimdata)
                    loss = loss_fn(output, labels)
                    
                    gradients = torch.autograd.grad(loss, model.parameters())
                    #shared_user_data[0]['gradients'] = gradients
                    
                    reconstructed_user_data, stats = attacker.reconstruct(payloads, shared_user_data, server_br.secrets, dryrun=False)

                # How good is the reconstruction?
                metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, payloads, model, cfg_case=cfg_case, setup=setup, order_batch=False)
                
                
                recovered_labels = reconstructed_user_data['labels'].sort()[0]
                true_user_data['labels'] = true_user_data['labels'].sort()[0]

                reconstructed_user_data = reconstructed_user_data['data']
                reconstructed_user_data = (reconstructed_user_data - reconstructed_user_data.min()) / (reconstructed_user_data.max() - reconstructed_user_data.min())
                
                    
                correct = 0
                wrong = 0
                predicted_labels = recovered_labels.clone()

                for label in true_user_data['labels']:
                    if label in predicted_labels:
                        correct += 1
                        index = np.where(predicted_labels == label)[0][0]

                        # Delete the first occurrence
                        predicted_labels = np.delete(predicted_labels, index)
                    else:
                        wrong +=1
                        
                # Calculate Label Leakage Accuracy for label existence
                unique_labelsGT = torch.unique(true_user_data['labels'])
                unique_labelsPD = torch.unique(recovered_labels)
                leAcc = 0
                leAccWrong = 0  
                for label in unique_labelsPD:
                    if label in unique_labelsGT:
                        leAcc += 1
                    elif label not in unique_labelsGT:
                        leAccWrong += 1
                leAcc = leAcc / unique_labelsGT.size()[0] # Label Existence Accuracy
                leAccWrong = leAccWrong / unique_labelsPD.size()[0] # Label Existence Prediction that were predicted, but not actually in the batch
                
                
                lnAcc = int((true_user_data['labels'].size()[0] - wrong) / true_user_data['labels'].size()[0] * 100)


                block2 = ''
                block2 += 'Correct Labels: ' + str(correct) + '\n' 
                block2 += 'Wrong Labels: ' + str(wrong) + '\n' 
                block2 += 'LnAcc: ' + str(lnAcc) + '%' + '\n'
                block2 += 'Metrics Acc: ' + str(metrics['label_acc']) + '\n'
                block2 += '\n'

                block1 = '\nLABEL LEAKAGE RESULTS:'
                print('\n'.join([block1, block2]))
                
                # submit final values to neptune 
                if run is not None:
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name].append({"leAcc": leAcc})
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name].append({"lnAcc": lnAcc})
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name].append({"lMetrics Acc": str(metrics['label_acc'])})
                
                recovered_labels_all.append(recovered_labels)
                batchLabels_all.append(true_user_data['labels']) 
                    
                batchLabels_all = torch.cat(batchLabels_all)
                recovered_labels_all = torch.cat(recovered_labels_all)
                
                # save final raw confusion matrix
                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                ax.set_title('Confusion Matrix: ' + str(label_strat) + ' ' + split_name)
                
                if run is not None:
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name + '/conf_matrices'].append(File.as_image(plt.gcf()), name='all')
                plt.close()

            
                #if run is not None:
                    #run['label_attack' + '/' + str(label_strat) + '/' + split_name +'/final_lnAcc'] = run['label_attack' + '/' + str(label_strat) + '/' + split_name +'/lnAcc'].fetch_values().mean().value
                    #lnAcc_all += run['label_attack' + '/' + str(label_strat) + '/' + split_name +'/lnAcc'].fetch_values().mean().value
                
                if trained and run is not None:
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/data/labelT-csv"].upload('CSV/labelsT_' + str(config['name']) + '_' + str(label_strat) + '.csv')
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/data/gradientT-csv"].upload('CSV/gradientsT_' + str(config['name']) + '_' + str(label_strat) + '.csv')
                elif run is not None:
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/data/label-csv"].upload('CSV/labels_' + str(config['name']) + '_' + str(label_strat) +  '.csv') 
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/data/gradient-csv"].upload('CSV/gradients_' + str(config['name']) + '_' + str(label_strat) + '.csv') 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/leakage/wear_loso_deep.yaml')
    parser.add_argument('--eval_type', default='loso')
    parser.add_argument('--neptune', default=False, type=bool)
    parser.add_argument('--run_id', default='run', type=str)
    parser.add_argument('--seed', default=1, type=int)       
    parser.add_argument('--gpu', default='cuda:0', type=str)
    
    # New arguments
    parser.add_argument('--attack', default='_default_optimization_attack', type=str)
    parser.add_argument('--label_strat_array', nargs='+', default=['llbgAVG', 'bias-corrected', 'iRLG', 'gcd', 'wainakh-simple', 'wainakh-whitebox', 'iDLG', 'analytic', 'yin', 'random'], type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--trained', default=False, type=bool)
    parser.add_argument('--sampling', default='shuffle', choices=['defense', 'balanced', 'unbalanced', 'shuffle'], type=str)
    args = parser.parse_args()
    
    leakage = Leakage()  
    leakage.main(args)
