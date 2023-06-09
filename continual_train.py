'''
Description: 
Author: Zhang yizhen
Date: 2023-03-07 18:35:29
LastEditors: Zhang yizhen
LastEditTime: 2023-03-23 16:56:54
FilePath: /zyz/piggyback_1/continual_train.py

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
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

def get_train_val_dataset(opt, task_name, task_id, is_baseline=False):
    dataset_path = os.path.join(opt.dataset_path, task_name)

    datasets = {}

    for data_type in ['train', 'val']:
        dataset = ImageRestorationDataset(os.path.join(dataset_path, data_type), istrain=(data_type=='train'), is_baseline=is_baseline, patch_size=opt.train_patch_size)
        dataloader = DataLoader(dataset, batch_size=opt.train.batch_size[task_id], shuffle=True, num_workers=4, drop_last=True)
        datasets[data_type] = dataloader

    return datasets

def get_test_dataset(opt, task_name, task_id, is_baseline=False):
    dataset_path = os.path.join(opt.dataset_path, task_name)

    datasets = {}

    for data_type in ['test']:
        dataset = ImageRestorationDataset(os.path.join(dataset_path, data_type), istrain=(data_type=='train'), is_baseline=is_baseline, patch_size=opt.test_patch_size)
        dataloader = DataLoader(dataset, batch_size=opt.train.batch_size[task_id], shuffle=True, num_workers=4, drop_last=True)
        datasets[data_type] = dataloader

    return datasets
def create_dataset(opt):
    task_seq = opt.task_sequence
    train_task_dataset = []
    test_task_dataset = []
    for task_id, task_name in enumerate(task_seq):
        train_dataset = get_train_val_dataset(opt, task_name, task_id)
        test_dataset = get_test_dataset(opt, task_name, task_id)
        train_task_dataset.append(train_dataset)
        test_task_dataset.append(test_dataset)
    
    return train_task_dataset,test_task_dataset



def train_per_task(opt, task_dataset, model, task_id, visualizer, start_epoch=1):
    """_summary_
    the train_per_task function

    Args:
        opt (_type_): the config of the training
        task_dataset (_type_): training and validation task dataset
        model (_type_): the CycleGAN model
        task_id (_type_): the task id of the training task
        visualizer (_type_): visualizer of the training task
        start_epoch (int, optional):  Defaults to 1.
    """    

    if not (task_id ==len(opt.train.num_epochs) and start_epoch >opt.train.num_epochs[len(opt.train.num_epochs)]):

        if start_epoch != 1:
            print(f'{"-"*78}')
            print(f'==> Resuming training for Task {task_id} from epoch {start_epoch} with learning rate: {model.optimizer.param_groups[0]["lr"]}')
            print(f'{"-"*78}')

        train_dataloader = task_dataset[task_id]['train']
        best_epoch, best_psnr = opt.train.num_epochs[task_id], 0.
        model.best_epoch = best_epoch
        # utils.fix_random_seed()

        epoch_loss = OrderedDict()

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
    
            if epoch % opt.train.val_interval == 0:
                tag = '{}_image_task'+str(task_id)
                visualizer.visualize_images(epoch=epoch,
                    **{
                        tag.format('degraded'):model.degraded_image,
                        tag.format('clean'):model.clean_image,
                        tag.format('restored'):model.restored_image
                    }
                )
                model.save_model(mode='epoch', task_id=task_id, epoch=epoch)
 
                # best_epoch, best_psnr = evaluate_tasks_till_now(opt, task_dataset, model, task_id, epoch, best_epoch, best_psnr, visualizer)

        conv_dx = 0
        for name, module in model.restore_net.module.named_modules():
                if isinstance(module, PiggybackConv) or isinstance(module, PiggybackTransposeConv):
                        module.unc_filt.requires_grad = False
                        module.bias.requires_grad = False

                        if model.task_num == 1:
                            model.net_filter_list.append([module.unc_filt.detach().cuda()])
                        elif model.task_num == 2:
                            model.net_filter_list[conv_dx].append(module.unc_filt.detach().cuda())
                            model.weights.append([module.weights_mat.detach().cuda()])
                            conv_dx += 1
                        else:
                            model.net_filter_list[conv_dx].append(module.unc_filt.detach().cuda())
                            model.weights[conv_dx].append(module.weights_mat.detach().cuda())
                            conv_dx += 1
        # conv_dx = 0
        # for name, module in model.netG_B.module.named_modules():
        #         if isinstance(module, PiggybackConv) or isinstance(module, PiggybackTransposeConv):
        #                 module.unc_filt.requires_grad = False
        #                 module.bias.requires_grad = False

        #                 if model.task_num == 1:
        #                     model.netG_B_filter_list.append([module.unc_filt.detach().cuda()])
        #                 elif model.task_num == 2:
        #                     model.netG_B_filter_list[conv_dx].append(module.unc_filt.detach().cuda())
        #                     model.netG_B_weights.append([module.weights_mat.detach().cuda()])
        #                     conv_dx += 1
        #                 else:
        #                     model.netG_B_filter_list[conv_dx].append(module.unc_filt.detach().cuda())
        #                     model.netG_B_weights[conv_dx].append(module.weights_mat.detach().cuda())
        #                     conv_dx += 1

        
        model.save_model(mode='latest')   

            
     
        

# def evaluate_tasks_till_now(opt, task_dataset, model, now_task_id, epoch, best_epoch, best_psnr, visualizer):
#     # model.eval()
#     # psnr_val = []
    
#     # pbar = tqdm(total=sum([len(task['val']) for task in task_dataset[:now_task_id+1]]), desc='',ncols=80)
#     # for task_id in range(now_task_id+1):
#     #     val_dataloader = task_dataset[task_id]['val']
#     #     psnr_val_rgb = []
#     #     restored_image = None
#     #     pbar.set_description(f'Validateing task {opt.task_sequence[task_id]}')



#     #     for batch_idx, input in enumerate(val_dataloader):
#     #         model.set_input(input)
#     #         restored_image = model.inference()

#     #         psnr_val_rgb.append(torch.tensor(model.get_current_psnr()))
#     #         pbar.update(1)
    
#     #     psnr_val_rgb = torch.stack(psnr_val_rgb).sum().item() / len(val_dataloader.dataset)
#     #     psnr_val.append(psnr_val_rgb)

#     #     tag = 'val_{}_image_task_'+str(task_id)
#     #     visualizer.visualize_images(epoch=epoch,
#     #         **{
#     #             tag.format('degraded'):model.degraded_image,
#     #             tag.format('clean'):model.clean_image,
#     #             tag.format('restored'):restored_image
#     #         }
#     #     )

#         # if task_id == now_task_id and psnr_val_rgb>best_psnr and psnr_val_rgb!= inf:
#             # best_psnr = psnr_val_rgb
#             # best_epoch = epoch
#     model.save_model(mode='best', task_id=now_task_id, epoch=epoch)
#             # ################################
#             # # presever best_params
#             # params = {n: p for n, p in model.named_parameters() if p.requires_grad}# 模型的所有参数
#             # for n, p in params.items():
#             #     model.best_params[n] = p.clone().detach()
#             # ################################
            
            
#     # pbar.close()
#     print(f'{"="*78}\nValidation for epoch {epoch}:\n')
#     # for task_id in range(now_task_id+1):
#     #     print(f" ------ Task: {opt.task_sequence[task_id]}\tpsnr: {psnr_val[task_id]:.4f}")
#     # print(f' ------ Best epoch: {best_epoch}\tBest PSNR: {best_psnr:.4f}')
#     print(f'model:task_{now_task_id}_epoch_{epoch}_model has been saved!')
#     print(f'\n{"="*78}\n')
    
#     # visualizer.visualize_scalars(f"val_PSNR", {f'Task{task_id}':psnr_val[task_id] for task_id in range(now_task_id+1)}, sum([batch*epoch_ for batch,epoch_ in zip([len(data) for data in task_dataset[:now_task_id]], opt.train.num_epochs[:now_task_id])])+epoch*len(task_dataset[now_task_id]))

#     model.save_model(mode='epoch', task_id=now_task_id, epoch=epoch)

#     return best_epoch, best_psnr

def test_tasks_till_now(opt, task_dataset, model, now_task_id, visualizer):
    ##load_best_epoch
    # print("model.best_epoch",model.best_epoch)
    folder=f'task_{now_task_id}_epoch_{opt.train.num_epochs[now_task_id]}_model'
    # folder = 'model_latest' 
    # folder=f'task_{now_task_id}_model_best'
    # print(f'task_{now_task_id}_model_best')
    load_path = os.path.join(opt.checkpoints.save_model_dir, folder)
    assert os.path.exists(load_path), 'Load file not exists!'
    load_dict = torch.load(load_path, map_location='cpu')

    count1,count2 = count_parameters(model.restore_net)
    print(f'The model has {count1} trainable parameters and {count2} parameters including the buffers ')
    
    model.restore_net.module.load_state_dict(load_dict['model'])
    model.eval()

    print(f'{"="*78}\nTesting with task{now_task_id} with the best epoch{model.best_epoch}:\n')
    print("loading_pb_conv...")
    
    for task_id in range(now_task_id+1):
        model.restore_net = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        opt.dropout, opt.init_type, opt.init_gain, task_num=task_id+1, filter_list = model.net_filter_list)
        # model.netG_B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 opt.dropout, opt.init_type, opt.init_gain, task_id, model.netG_B_filter_list)
        model.restore_net = load_pb_conv(model.restore_net,model.net_filter_list,model.weights,task_id)
        # model.netG_B = load_pb_conv(model.netG_B,model.netG_B_filter_list,model.netG_B_weights,model.netG_B_bias,task_id)
        model.set_device('gpu')
        count1,count2 = count_parameters(model.restore_net)
        print(f'The model has {count1} trainable parameters and {count2} parameters including the buffers ')
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

def count_parameters(model):
    count_parameters1 = sum(p.numel() for p in model.parameters())
    count_parameters2 = sum(p.numel() for p in model.parameters())+sum(b.numel() for b in model.buffers())
    return count_parameters1,count_parameters2



def train_and_evaluate(opt):
    utils.fix_random_seed()

    train_task_dataset,test_task_dataset = create_dataset(opt)

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
    # print(model.restore_net)
    # summary(model.restore_net, input_size=(3, 256, 256), batch_size=8)
    # summary(model.netG_B, input_size=(3, 256, 256), batch_size=8)
    # summary(model.netD_A, input_size=(3, 256, 256), batch_size=8)
    # summary(model.netD_B, input_size=(3, 256, 256), batch_size=8)
    # for module in model.restore_net.modules:
    #     print(module)

    visualizer = utils.Visualizer(SummaryWriter(), model, opt.visualize, opt.checkpoints.save_image)

    # train_baseline(opt, model, opt.task_sequence[0], task_id=0)

    for task_id, task_name in enumerate(opt.task_sequence[start_task_id:], start_task_id):
        # print("start_epoch",start_epoch)
        train_per_task(opt, train_task_dataset, model, task_id, visualizer, start_epoch)
        test_tasks_till_now(opt, test_task_dataset, model, task_id, visualizer)
        if (task_id<len(opt.task_sequence)):
            model.update_task()
            model.set_device('gpu')
        start_epoch = 1