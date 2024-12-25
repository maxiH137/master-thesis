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
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import sys
import time
from pprint import pprint
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from utils.os_utils import Logger, load_config
from libs.datasets import make_dataset, make_data_loader
from utils.torch_utils import fix_random_seed
from models.DeepConvLSTM import DeepConvLSTM
from models.TinyHAR import TinyHAR
from utils.torch_utils import init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from torch.utils.data import DataLoader
from opacus import layers, optimizers
from Unbalanced_Sampler import UnbalancedSampler
from DPrivacy import DPrivacy

import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Opt-in to the future behavior to prevent the warning
pd.set_option('future.no_silent_downcasting', True)

class data_cfg_default:
    modality = "vision"
    size = (1_281_167,)
    classes = 1000
    shape = (3, 224, 224)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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
            run['args/dataset'] = args.dataset
            run['args/attack'] = args.attack
            run['args/datapoints'] = config['loader']['train_batch_size']
            run['args/classes'] = config['dataset']['num_classes']
            run['args/label_strat'] = args.label_strat
        
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup = dict(device=torch.device("cpu"), dtype=torch.float)
        
        trained = args.trained
        grads = []

        if trained:
            with open('labelsT_' + str(config['name']) + '.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['GT', 'Pred', 'Sbj'])

            with open('gradientsT_' + str(config['name']) + '.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    for g in range(config['dataset']['num_classes'] + 1):
                        grads.append('G' + str(g))
                    writer.writerow(grads)
        else:
            with open('labels_' + str(config['name']) + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['GT', 'Pred', 'Sbj'])

            with open('gradients_' + str(config['name']) + '.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    for g in range(config['dataset']['num_classes'] + 1):
                        grads.append('G' + str(g))
                    writer.writerow(grads)

        percentage_all = 0
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
                t_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'], t_sbj + '.csv'), index_col=False).replace({"label": config['label_dict']}).infer_objects(copy=False).fillna(0).to_numpy()
                train_data = np.append(train_data, t_data, axis=0)
            for v_sbj in val_sbjs:
                v_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'], v_sbj + '.csv'), index_col=False).replace({"label": config['label_dict']}).infer_objects(copy=False).fillna(0).to_numpy()
                val_data = np.append(val_data, v_data, axis=0)

            # define inertial datasets
            train_dataset = InertialDataset(train_data, config['dataset']['window_size'], config['dataset']['window_overlap'])
            test_dataset = InertialDataset(val_data, config['dataset']['window_size'], config['dataset']['window_overlap'])

            # define dataloaders
            unbalanced_sampler = UnbalancedSampler(test_dataset, random.randint(0, config['dataset']['num_classes'] - 1), random.randint(0, config['dataset']['num_classes'] - 1))
            config['init_rand_seed'] = args.seed
            rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True) 
            train_loader = DataLoader(train_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
            val_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
            unbalanced_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=unbalanced_sampler, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
            
            if 'tinyhar' in config['name'] and trained:
                args.resume = f'wear loso models/tinyhar/epoch_100_loso_sbj_{i}.pth.tar'
            if 'deepconvlstm' in config['name'] and trained:
                args.resume = f'wear loso models/deepconvlstm/epoch_100_loso_sbj_{i}.pth.tar'

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
                model = torchvision.models.resnet152(pretrained=trained)
                model.conv1 = nn.Conv2d(1, 
                                        model.conv1.out_channels, 
                                        kernel_size = model.conv1.kernel_size, 
                                        stride = model.conv1.stride, 
                                        padding = model.conv1.padding, 
                                        bias = model.conv1.bias)
                
                model.fc = nn.Linear(model.fc.in_features, config['dataset']['num_classes'] + 1) 
                
                model = init_weights(model, config['train_cfg']['weight_init'])
                
            model.eval()
                
            loss_fn = torch.nn.CrossEntropyLoss()

            # This is the attacker:
            #cfg_attack = breaching.get_attack_config("invertinggradients")
            cfg_attack = breaching.get_attack_config(args.attack)
            cfg_attack['optim']['max_iterations'] = int(args.iterations)
            cfg_attack['label_strategy'] = args.label_strat
            
            if args.neptune:
                log_dir_atk = os.path.join('logs', args.attack, '_' + run_id)
                sys.stdout = Logger(os.path.join(log_dir_atk, 'log.txt'))
                with open(os.path.join(log_dir_atk, 'cfg_atk.txt'), 'w') as fid:
                    pprint(cfg_attack, stream=fid)
                    fid.flush()
                
                run['attack_config_name'] = args.attack
                run['attack_config'].upload(os.path.join(log_dir_atk, 'cfg_atk.txt'))
                
            attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

            # ## Simulate an attacked FL protocol
            # Server-side computation:
            
            metadata = data_config_inertial()
            metadata.shape = (1, 50, config['dataset']['input_dim'])
            metadata.classes = config['dataset']['num_classes'] + 1
            metadata['task'] = 'classification'
            
            server_payload = [
                dict(
                    #parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
                    parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=metadata
                )
            ]

            # LOAD DATA
            for _, (inputs, targets) in enumerate(unbalanced_loader, 0):
                
                if 'deepconvlstm' in config['name']:
                    val_data = inputs
                    labels = targets
                    all_labels = targets
                    
                    if(val_data.shape[0] != config['loader']['train_batch_size'] or 
                    val_data.shape[1] != 50 or 
                    val_data.shape[2] != config['dataset']['input_dim']):
                        break

                    
                if 'tinyhar' in config['name']:
                    val_data = inputs
                    labels = targets
                    all_labels = targets
                    
                    if(val_data.shape[0] != config['loader']['train_batch_size'] or 
                    val_data.shape[1] != 50 or 
                    val_data.shape[2] != config['dataset']['input_dim']):
                        break

                
                if config['name'] == 'ResNet':
                    val_data = inputs
                    labels = targets
                    all_labels = targets
                    onedimdata = val_data
                    
                    if(val_data.shape[0] != config['loader']['train_batch_size'] or 
                    val_data.shape[1] != 50 or 
                    val_data.shape[2] != config['dataset']['input_dim']):
                        break
                
                
                # Normalize data
                onedimdata = val_data.unsqueeze(1)
                onedimdata = (onedimdata - onedimdata.min()) / (onedimdata.max() - onedimdata.min())
                

                # User-side computation:
                #loss = loss_fn(model(datapoint[None, ...]), labels)
                
                output = model(onedimdata)
                loss = loss_fn(output, labels)
                
                gradients=torch.autograd.grad(loss, model.parameters())
                
                # Access gradients
                for (name, _), grad in zip(model.named_parameters(), gradients):
                    print(f"Gradient of {name}: {grad}")
                grad_bias = gradients[-1]
                grad_weight = gradients[-2]
                
                if run is not None:
                    run['gradients/weight'].log(grad_weight)
                    run['gradients/bias'].log(grad_bias)
                                
                shared_data = [
                    dict(
                        gradients=gradients,
                        buffers=None,
                        metadata=dict(num_data_points=config['loader']['train_batch_size'], labels=None, local_hyperparams=None,),
                    )
                ]
                
                # Attack:
                reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)
                
                rlabels = reconstructed_user_data['labels']
                rlabels = rlabels.sort()[0]
                all_labels = all_labels.sort()[0]
                #all_labels = all_labels[all_labels != 0]

                reconstructed_user_data = reconstructed_user_data['data']
                reconstructed_user_data = (reconstructed_user_data - reconstructed_user_data.min()) / (reconstructed_user_data.max() - reconstructed_user_data.min())
                

                OriginalData = onedimdata.squeeze(1).reshape(50 * config['loader']['train_batch_size'], config['dataset']['input_dim'])
                ReconstructedData = reconstructed_user_data.squeeze(1).reshape(50 * config['loader']['train_batch_size'], config['dataset']['input_dim'])
                
                gradient = torch.round(shared_data[0]["gradients"][-1], decimals=6)
                
                appendGradient = {f'G{i}': [gradient[i].item()] for i in range(config['dataset']['num_classes'] + 1)}
                appendGradient = pd.DataFrame(appendGradient)
                
                appendData = pd.DataFrame({
                    'GT': all_labels.numpy(),
                    'Pred': rlabels.numpy(),
                    'Sbj': val_sbjs[0]
                })

                # Append to CSV without writing the header again
                if(not trained):
                    appendData.to_csv('labels_' + str(config['name']) + '.csv', mode='a', header=False, index=False)
                else: 
                    appendData.to_csv('labelsT_' + str(config['name']) + '.csv', mode='a', header=False, index=False)

                if(not trained):
                    appendGradient.to_csv('gradients_' + str(config['name']) + '.csv', mode='a', header=False, index=False, float_format='%.4f')
                else:
                    appendGradient.to_csv('gradientsT_' + str(config['name']) + '.csv', mode='a', header=False, index=False, float_format='%.4f')
                    
                    
                correct = 0
                wrong = 0
                predicted_labels = rlabels.clone()

                for label in all_labels:
                    if label in predicted_labels:
                        correct += 1
                        index = np.where(predicted_labels == label)[0][0]

                        # Delete the first occurrence
                        predicted_labels = np.delete(predicted_labels, index)
                    else:
                        wrong +=1
                    
                percentage = int((all_labels.size()[0] - wrong) / all_labels.size()[0] * 100)

                block2 = ''
                block2 += 'Correct Labels: ' + str(correct) + '\n' 
                block2 += 'Wrong Labels: ' + str(wrong) + '\n' 
                block2 += 'Percentage: ' + str(percentage) + '%' + '\n'
                block2 += '\n'

                block1 = '\nLABEL LEAKAGE RESULTS:'
                print('\n'.join([block1, block2]))
                
                # submit final values to neptune 
                if run is not None:
                    transform = T.ToPILImage()
                    run[split_name].append({"percentage": percentage})
                    run[split_name + "/images/original"].append(transform(OriginalData))
                    run[split_name + "/images/reconstruction"].append(transform(ReconstructedData))
            
            if run is not None:
                run[split_name +'/final_perecentage'] = run[split_name +'/percentage'].fetch_values().mean().value
                percentage_all += run[split_name +'/percentage'].fetch_values().mean().value
            
            if trained and run is not None:
                run[split_name + "/data/labelT-csv"].upload('labelsT_' + str(config['name']) + '.csv')
                run[split_name + "/data/gradientT-csv"].upload('gradientsT_' + str(config['name']) + '.csv')
            elif run is not None:
                run[split_name + "/data/label-csv"].upload('labels_' + str(config['name']) + '.csv') 
                run[split_name + "/data/gradient-csv"].upload('gradients_' + str(config['name']) + '.csv') 

        if run is not None:
            run['final_percentage_all'] = percentage_all / len(config['anno_json'])
            print('Final Percentage: ' + str(percentage_all / len(config['anno_json'])) + '%')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/deepconvlstm/wear_loso.yaml')
    parser.add_argument('--eval_type', default='loso')
    parser.add_argument('--neptune', default=False, type=bool)
    parser.add_argument('--run_id', default='run', type=str)
    parser.add_argument('--seed', default=1, type=int)       
    parser.add_argument('--ckpt-freq', default=100, type=int)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    
    # New arguments
    parser.add_argument('--attack', default='_default_optimization_attack', type=str)
    parser.add_argument('--label_strat', default='llbg', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--iterations', default='1', type=str)
    parser.add_argument('--dataset', default='wear', type=str)
    parser.add_argument('--trained', default=False, type=bool)
    args = parser.parse_args()
    
    leakage = Leakage()  
    leakage.main(args)
