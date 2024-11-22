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
import breaching
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
from utils.os_utils import Logger, load_config
from libs.datasets import make_dataset, make_data_loader
from utils.torch_utils import fix_random_seed
from models.DeepConvLSTM import DeepConvLSTM
from models.TinyHAR import TinyHAR
from utils.torch_utils import init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from torch.utils.data import DataLoader
from opacus import layers, optimizers

import os
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


def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = dict(device=torch.device("cpu"), dtype=torch.float)
    
    config = load_config(args.config)
    trained = False
    grads = []

    if args.model == 'TinyHAR' and trained:
        #args.resume = 'D:/OneDrive/Desktop/Masterarbeit/wear loso models/tinyhar/epoch_100_loso_sbj_0.pth.tar'
        args.resume = 'wear loso models/tinyhar/epoch_100_loso_sbj_0.pth.tar'
    if args.model == 'DeepConvLSTM'and trained:
        #args.resume = 'D:/OneDrive/Desktop/Masterarbeit/wear loso models/deepconvlstm/epoch_100_loso_sbj_0.pth.tar'
        args.resume = 'wear loso models/deepconvlstm/epoch_100_loso_sbj_0.pth.tar'

    if trained:
        with open('labelsT_' + str(args.model) + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['GT', 'Pred', 'Sbj'])

        with open('gradientsT_' + str(args.model) + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for g in range(config['dataset']['num_classes'] + 1):
                    grads.append('G' + str(g))
                writer.writerow(grads)
    else:
        with open('labels_' + str(args.model) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['GT', 'Pred', 'Sbj'])

        with open('gradients_' + str(args.model) + '.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for g in range(config['dataset']['num_classes'] + 1):
                    grads.append('G' + str(g))
                writer.writerow(grads)


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
    config['init_rand_seed'] = args.seed
    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True) 
    train_loader = DataLoader(train_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    val_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
  

    if args.model == 'DeepConvLSTM':
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
        
    if args.model == 'TinyHAR':
        model = TinyHAR((100, 1, train_dataset.features.shape[1], train_dataset.channels), config['dataset']['num_classes'] + 1, 
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
       
      
    if args.model == 'ResNet':
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
    x=0
    for _, (inputs, targets) in enumerate(val_loader, 0):
        
        if args.model == 'DeepConvLSTM':
            val_data = inputs
            labels = targets
            all_labels = targets
            #if(x==50):
            #    break
            #x+=1
            if(val_data.shape[0] != config['loader']['train_batch_size'] or val_data.shape[1] != 50 or val_data.shape[2] != config['dataset']['input_dim']):
                break

            
        if args.model == 'TinyHAR':
            val_data = inputs
            labels = targets
            all_labels = targets
            #if(x==50):
            #    break
            #x+=1
            if(val_data.shape[0] != config['loader']['train_batch_size'] or val_data.shape[1] != 50 or val_data.shape[2] != config['dataset']['input_dim']):
                break

        
        if args.model == 'ResNet':
            val_data = inputs
            labels = targets
            all_labels = targets
            
            onedimdata = val_data
            
            if(val_data.shape[0] != config['loader']['train_batch_size'] or val_data.shape[1] != 50 or val_data.shape[2] != config['dataset']['input_dim']):
                break
        
        
        # Normalize data
        onedimdata = val_data.unsqueeze(1)
        onedimdata = (onedimdata - onedimdata.min()) / (onedimdata.max() - onedimdata.min())
        

        # User-side computation:
        #loss = loss_fn(model(datapoint[None, ...]), labels)
        
        loss = loss_fn(model(onedimdata), labels)
        gradients=torch.autograd.grad(loss, model.parameters())
        
        
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
        

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0,0].imshow(onedimdata.squeeze(1).reshape(5000,config['dataset']['input_dim']))
        
        
        #axs[0,0].title.set_text('Label GT: ' + str(labels.item()))
        #axs[0,1].title.set_text('Label  P: ' + str(reconstructed_user_data['labels'].item()))
        
        axs[0,1].imshow(reconstructed_user_data.squeeze(1).reshape(5000,config['dataset']['input_dim']))
    
        axs[1,0].imshow(all_labels.unsqueeze(0))
    
        axs[1,1].imshow(rlabels.unsqueeze(0))

        #plt.savefig("figure3.png")  # You can specify the filename and format
        #plt.show(block=False)

        
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
            appendData.to_csv('labels_' + str(args.model) + '.csv', mode='a', header=False, index=False)
        else: 
            appendData.to_csv('labelsT_' + str(args.model) + '.csv', mode='a', header=False, index=False)

        if(not trained):
            appendGradient.to_csv('gradients_' + str(args.model) + '.csv', mode='a', header=False, index=False, float_format='%.4f')
        else:
            appendGradient.to_csv('gradientsT_' + str(args.model) + '.csv', mode='a', header=False, index=False, float_format='%.4f')
            
        # print final results to terminal
        y = all_labels.size()[0]
        x = (all_labels-rlabels).count_nonzero().item()

        correct = 0
        wrong = 0
        rlablesTmp = rlabels.clone()

        for aLabel in all_labels:
            if aLabel in rlablesTmp:
                correct += 1
                index = np.where(rlablesTmp == aLabel)[0][0]

                # Delete the first occurrence
                rlablesTmp = np.delete(rlablesTmp, index)
            else:
                wrong +=1
            

        block2 = ''
        block2 += 'Correct Labels: ' + str(correct) + '\n' # str(all_labels.size()[0] - (all_labels-rlabels).count_nonzero().item())
        block2 += 'Wrong Labels: ' + str(wrong) + '\n' # str((all_labels-rlabels).count_nonzero().item())
        block2 += 'Percentage: ' + str(int((all_labels.size()[0] - wrong) / all_labels.size()[0] * 100)) + '%' + '\n'
        block2 += '\n'

        block1 = '\nLABEL LEAKAGE RESULTS:'
        print('\n'.join([block1, block2]))

        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/deepconvlstm/wear_loso.yaml')
    #parser.add_argument('--config', default='./configs/tinyhar/wear_loso.yaml')
    parser.add_argument('--eval_type', default='split')
    parser.add_argument('--neptune', default=False, type=bool)
    parser.add_argument('--run_id', default='run', type=str)
    parser.add_argument('--seed', default=1, type=int)       
    parser.add_argument('--ckpt-freq', default=100, type=int)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    # New arguments
    #parser.add_argument('--attack', default='joint-optimization', type=str)
    parser.add_argument('--attack', default='_default_optimization_attack', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--iterations', default='1', type=str)
    parser.add_argument('--dataset', default='wear', type=str)
    #parser.add_argument('--model', default='ResNet', type=str)
    parser.add_argument('--model', default='DeepConvLSTM', type=str)
    #parser.add_argument('--model', default='TinyHAR', type=str)
    #parser.add_argument('--model', default='ResNet', type=str)
    args = parser.parse_args()
    main(args)  
