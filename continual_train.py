'''
Description: 
Author: Zhang yizhen
Date: 2023-03-07 18:35:29
LastEditors: Zhang yizhen
LastEditTime: 2023-03-21 15:23:09
FilePath: /zyz/Piggyback_GAN/continual_train.py

Copyright (c) 2023 by yizhen_coder@outlook.com, All Rights Reserved. 
'''
from cmath import inf
import os, random
from typing import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.networks import PiggybackConv, PiggybackTransposeConv, load_pb_conv
from model.PiggybackGAN import PiggybackGAN
from model.networks import define_G
import numpy as np

from dataset import ImageRestorationDataset
import utils 

def get_dataset(opt, task_name, task_id, is_baseline=False):
    dataset_path = os.path.join(opt.dataset_path, task_name)

    datasets = {}

    for data_type in ['train', 'val', 'test']:
        dataset = ImageRestorationDataset(os.path.join(dataset_path, data_type), istrain=(data_type=='train'), is_baseline=is_baseline, patch_size=opt.patch_size)
        dataloader = DataLoader(dataset, batch_size=opt.train.batch_size[task_id], shuffle=True, num_workers=4, drop_last=True)
        datasets[data_type] = dataloader

    return datasets

def create_dataset(opt):
    task_seq = opt.task_sequence
    task_dataset = []
    for task_id, task_name in enumerate(task_seq):
        dataset = get_dataset(opt, task_name, task_id)
        task_dataset.append(dataset)
    
    return task_dataset



def train_per_task(opt, task_dataset, model, task_id, visualizer, start_epoch=1):
    ################################
    if not (task_id ==len(opt.train.num_epochs) and start_epoch >opt.train.num_epochs[len(opt.train.num_epochs)]):
    ################################
        if start_epoch != 1:
            print(f'{"-"*78}')
            print(f'==> Resuming training for Task {task_id} from epoch {start_epoch} with learning rate: {model.optimizer.param_groups[0]["lr"]}')
            print(f'{"-"*78}')

        train_dataloader = task_dataset[task_id]['train']
        best_epoch, best_psnr = opt.train.num_epochs[task_id], 0.
        
        # utils.fix_random_seed()

        epoch_loss = OrderedDict()
        ##
        #确保当前是在训练之前先载入之前保存的filter文件
        ###
        # if task_id == 0:
        #     model.net_filter_list = []
        #     model.weights = []
        #     model.bias = []
        # else:
        #     # old_task_folder_name = "Task_"+str(task_id)+"_"+opt.task_sequence[task_id]
        #     # print("Loading ", os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
        #     # filters = torch.load(os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
        #     model.net_filter_list = model.filters["net_filter_list"]
        #     model.weights = model.filters["weights"]
        #     model.bias = model.filters["bias"]
        ##
        #正常训练对应的weights矩阵
        ###
        for epoch in range(start_epoch, opt.train.num_epochs[task_id]+1):
            model.train()
            for batch_idx, input in enumerate(tqdm(train_dataloader,ncols=80)):
                model.set_input(input)
                model.optimize_parameters()

                batch_loss = model.get_current_losses()
                visualizer.visualize_scalars(f'Loss_task{task_id}', batch_loss, (epoch-1)*len(train_dataloader)+batch_idx)
                if len(epoch_loss) == 0:
                    epoch_loss = batch_loss
                else:
                    for key in epoch_loss.keys():
                        epoch_loss[key] += batch_loss[key]

            for key in epoch_loss.keys():
                epoch_loss[key] /= len(train_dataloader)

            print(f"{'='*78}\nTraining for task {task_id}\n")
            print(f"Epoch: {epoch}\tLearningRate {model.optimizer.param_groups[0]['lr']:.8f}")
            for key, value in epoch_loss.items():
                print(f"\t{key}: {value:.4f}")
            print(f'\n{"="*78}\n')

            model.update_learning_rate()
            # print("model.named_parameters()")
            # for n, p in model.restore_net.named_parameters():  
            #     if p.requires_grad:   
            #             print(n)
            #             print(p)

            if epoch % opt.train.val_interval == 0:
                tag = '{}_image_task'+str(task_id)
                visualizer.visualize_images(epoch=epoch,
                    **{
                        tag.format('degraded'):model.degraded_image,
                        tag.format('clean'):model.clean_image,
                        tag.format('restored'):model.restored_image
                    }
                )
                # best_epoch, best_psnr = evaluate_tasks_till_now(opt, task_dataset, model, task_id, epoch, best_epoch, best_psnr, visualizer)
 
        conv_dx = 0
        for name, module in model.restore_net.module.named_modules():
                if isinstance(module, PiggybackConv) or isinstance(module, PiggybackTransposeConv):
                        module.unc_filt.requires_grad = False
                        module.bias.requires_grad = False

                        if model.task_num == 1:
                            model.net_filter_list.append([module.unc_filt.detach().cuda()])
                            model.bias.append([module.bias.detach().cuda()])
                        elif model.task_num == 2:
                            model.net_filter_list[conv_dx].append(module.unc_filt.detach().cuda())
                            model.bias[conv_dx].append(module.bias.detach().cuda())
                            model.weights.append([module.weights_mat.detach().cuda()])
                            conv_dx += 1
                        else:
                            model.net_filter_list[conv_dx].append(module.unc_filt.detach().cuda())
                            model.bias[conv_dx].append(module.bias.detach().cuda())
                            model.weights[conv_dx].append(module.weights_mat.detach().cuda())
                            conv_dx += 1
        
     


                
                

        model.best_epoch = best_epoch
        model.save_model(mode='latest')   

        model.save_model(model_name='target')
        # if (task_id<len(opt.task_sequence)):
        #     model.update_task()
        #     model.set_device('gpu')
        
        

def evaluate_tasks_till_now(opt, task_dataset, model, now_task_id, epoch, best_epoch, best_psnr, visualizer):
    # model.eval()
    # psnr_val = []
    
    # pbar = tqdm(total=sum([len(task['val']) for task in task_dataset[:now_task_id+1]]), desc='',ncols=80)
    # for task_id in range(now_task_id+1):
    #     val_dataloader = task_dataset[task_id]['val']
    #     psnr_val_rgb = []
    #     restored_image = None
    #     pbar.set_description(f'Validateing task {opt.task_sequence[task_id]}')



    #     for batch_idx, input in enumerate(val_dataloader):
    #         model.set_input(input)
    #         restored_image = model.inference()

    #         psnr_val_rgb.append(torch.tensor(model.get_current_psnr()))
    #         pbar.update(1)
    
    #     psnr_val_rgb = torch.stack(psnr_val_rgb).sum().item() / len(val_dataloader.dataset)
    #     psnr_val.append(psnr_val_rgb)

    #     tag = 'val_{}_image_task_'+str(task_id)
    #     visualizer.visualize_images(epoch=epoch,
    #         **{
    #             tag.format('degraded'):model.degraded_image,
    #             tag.format('clean'):model.clean_image,
    #             tag.format('restored'):restored_image
    #         }
    #     )

        # if task_id == now_task_id and psnr_val_rgb>best_psnr and psnr_val_rgb!= inf:
            # best_psnr = psnr_val_rgb
            # best_epoch = epoch
    model.save_model(mode='best', task_id=now_task_id, epoch=epoch)
            # ################################
            # # presever best_params
            # params = {n: p for n, p in model.named_parameters() if p.requires_grad}# 模型的所有参数
            # for n, p in params.items():
            #     model.best_params[n] = p.clone().detach()
            # ################################
            
            
    # pbar.close()
    print(f'{"="*78}\nValidation for epoch {epoch}:\n')
    # for task_id in range(now_task_id+1):
    #     print(f" ------ Task: {opt.task_sequence[task_id]}\tpsnr: {psnr_val[task_id]:.4f}")
    # print(f' ------ Best epoch: {best_epoch}\tBest PSNR: {best_psnr:.4f}')
    print(f'model:task_{now_task_id}_epoch_{epoch}_model has been saved!')
    print(f'\n{"="*78}\n')
    
    # visualizer.visualize_scalars(f"val_PSNR", {f'Task{task_id}':psnr_val[task_id] for task_id in range(now_task_id+1)}, sum([batch*epoch_ for batch,epoch_ in zip([len(data) for data in task_dataset[:now_task_id]], opt.train.num_epochs[:now_task_id])])+epoch*len(task_dataset[now_task_id]))

    model.save_model(mode='epoch', task_id=now_task_id, epoch=epoch)

    return best_epoch, best_psnr

def test_tasks_till_now(opt, task_dataset, model, now_task_id, visualizer):
    ##load_best_epoch
    # print("model.best_epoch",model.best_epoch)
    # folder=f'task_{now_task_id}_epoch_{model.best_epoch}_model'
    folder = 'model_latest' 
    # folder=f'task_{now_task_id}_model_best'
    print(f'task_{now_task_id}_model_best')
    load_path = os.path.join(opt.checkpoints.save_model_dir, folder)
    assert os.path.exists(load_path), 'Load file not exists!'
    load_dict = torch.load(load_path, map_location='cpu')
    # model.restore_net.module.load_state_dict(load_dict['model'])
    model.eval()

    print(f'{"="*78}\nTesting with task{now_task_id} with the best epoch{model.best_epoch}:\n')
    print("loading_pb_conv...")
    
    for task_id in range(now_task_id+1):
        model.restore_net = define_G(model.opt, model.opt.model.net, replace=model.opt.model.replace, task_id=task_id, filter_list=model.net_filter_list)
        model.restore_net = load_pb_conv(model.restore_net,model.net_filter_list,model.weights,model.bias,task_id)
        test_dataloader = task_dataset[task_id]['test']
        total_psnr, total_ssim = 0., 0.
        utils.fix_random_seed()

        for batch_idx, input in enumerate(test_dataloader):
            model.set_input(input)
            model.inference()

            if batch_idx % opt.checkpoints.save_image_interval == 0:
                save_folder = os.path.join(opt.checkpoints.save_image_dir, f'Task{task_id}', f't{now_task_id}')
                visualizer.save_images(save_folder)

            total_psnr += model.get_current_psnr()
            total_ssim += model.get_current_ssim()

        total_psnr /= len(test_dataloader.dataset)
        total_ssim /= len(test_dataloader.dataset)
        
        print(f" ------ Task: {opt.task_sequence[task_id]}, \
                        PSNR: {total_psnr}, \
                        SSIM: {total_ssim}")

    print(f'\n{"="*78}\n')
    ##back_to_last_epoch
    # folder=f'task_{now_task_id}_epoch_{opt.train.num_epochs[now_task_id]}_model'
    # print("shifting with latest")
    # load_path = os.path.join(opt.checkpoints.save_model_dir, folder)
    # assert os.path.exists(load_path), 'Load file not exists!'
    # load_dict = torch.load(load_path, map_location='cpu')
    # model.restore_net.module.load_state_dict(load_dict['model'])
   

# def compute_importance(opt, task_dataset, model, task_id):
#     print(f'{"-"*78}')
#     print("compute_importance")
#     print(f'{"-"*78}')
#     utils.fix_random_seed()
#     #init
#     params = {n: p for n, p in model.named_parameters() if p.requires_grad}
#     precision_matrices = {} 
#     for n, p in params.items():
#         precision_matrices[n] = 0*p.data

#     train_dataloader = task_dataset[task_id]['train']
#     #compute
#     model.eval()
#     for batch_idx, input in enumerate(tqdm(train_dataloader,ncols=80)):
#         model.set_input(input)
#         model.zero_grad()
#         model.forward()
#         model.backward_G()
#         for n,p in model.named_parameters():
#             precision_matrices[n].data += p.grad.data ** 2 /len(train_dataloader)
#     model.importance = precision_matrices

def train_and_evaluate(opt):
    utils.fix_random_seed()

    task_dataset = create_dataset(opt)

    model = PiggybackGAN(opt)

    start_task_id, start_epoch = 0, 1
    if opt.checkpoints.resume:
        model.load_model(mode=opt.checkpoints.resume_mode, task_id=opt.checkpoints.resume_task_id, epoch=opt.checkpoints.resume_epoch)
        start_task_id = model.task_num - 1
        start_epoch = model.scheduler.last_epoch + model.scheduler.after_scheduler.last_epoch + 1

    if torch.cuda.device_count()>=1:
        print(f"Using {torch.cuda.device_count()} GPUs!\n")
        model.set_device('gpu')
        model.cuda()

    # for module in model.restore_net.modules:
    #     print(module)

    visualizer = utils.Visualizer(SummaryWriter(), model, opt.visualize, opt.checkpoints.save_image)

    # train_baseline(opt, model, opt.task_sequence[0], task_id=0)

    for task_id, task_name in enumerate(opt.task_sequence[start_task_id:], start_task_id):
        # print("start_epoch",start_epoch)
        train_per_task(opt, task_dataset, model, task_id, visualizer, start_epoch)
        test_tasks_till_now(opt, task_dataset, model, task_id, visualizer)
        if (task_id<len(opt.task_sequence)):
            model.update_task()
            model.set_device('gpu')
        start_epoch = 1