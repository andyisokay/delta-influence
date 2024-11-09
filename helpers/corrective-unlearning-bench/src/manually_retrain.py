import torch, methods, resnet, timm
import numpy as np
import sys
import os 
from os import makedirs
from os.path import exists
from torch.utils.data.sampler import SubsetRandomSampler
from opts import parse_args 
from utils import seed_everything, SubsetSequentialSampler, get_targeted_classes  
from dataset import load_dataset, DatasetWrapper, manip_dataset, get_deletion_set

if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(seed=3017)
    opt = parse_args()
    print('==> Opts: ',opt)

    if opt.device == 'cuda':
        assert(torch.cuda.is_available())

    # Get model
    # model = getattr(resnet, opt.model)(opt.num_classes).cuda()
    if opt.model == 'vitb16':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=opt.num_classes).cuda()
    elif opt.device == 'cuda':
        model = getattr(resnet, opt.model)(opt.num_classes).cuda()
    else:
        model = getattr(resnet, opt.model)(opt.num_classes)
    
    # Get dataloaders done
    # train_set: with augmentation
    # train_noaug_set: no aug
    train_set, train_noaug_set, test_set, train_labels, max_val = load_dataset(dataset=opt.dataset, root=opt.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    manip_dict, manip_idx, untouched_idx = manip_dataset(dataset=opt.dataset, train_labels=train_labels, method=opt.dataset_method, manip_set_size=opt.forget_set_size, save_dir=opt.save_dir)
    # manip_idx_path = save_dir+'/'+dataset+'_'+method+'_'+str(manip_set_size)+'_manip.npy'
    print('==> Loaded the dataset! (clean)')
    # print(f"manip_idx:/n {manip_idx}")
    # aaaa

    wtrain_noaug_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode='test')
    train_test_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(untouched_idx), num_workers=4, pin_memory=True)
    manip_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(manip_idx), num_workers=4, pin_memory=True)
    eval_loaders = {}
    if opt.dataset_method == 'poisoning':
        corrupt_val = np.array(max_val)
        corrupt_size = opt.patch_size
        wtrain_noaug_adv_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode='test_adversarial', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
        adversarial_train_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # cleanL is for clean label
        untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(untouched_idx), num_workers=4, pin_memory=True)
        manip_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(manip_idx), num_workers=4, pin_memory=True)
        wtest_adv_cleanL_set = DatasetWrapper(test_set, manip_dict, mode='test_adversarial', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
        adversarial_test_loader = torch.utils.data.DataLoader(wtest_adv_cleanL_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        eval_loaders['adv_test'] = adversarial_test_loader
    else:
        adversarial_train_loader, adversarial_test_loader, corrupt_val, corrupt_size = None, None, None, None

    eval_loaders['manip'] = manip_noaug_cleanL_loader
    if opt.dataset_method == 'interclasslabelswap':
        classes = get_targeted_classes(opt.dataset)
        indices = []
        for batch_idx, (data, target) in enumerate(test_loader):
            matching_indices = (target == classes[0]) | (target == classes[1])
            absolute_indices = batch_idx * test_loader.batch_size + torch.where(matching_indices)[0]
            indices.extend(absolute_indices.tolist())
        eval_loaders['unseen_forget'] = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(indices), num_workers=4, pin_memory=True)

    wtrain_manip_set = DatasetWrapper(train_set, manip_dict, mode='pretrain', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
    # pretrain loader: manip examples (examples with triggers and modified labels, i.e. poisons) + cleans

    # retrain loader: remove those detected poisons
    # directly assign forget_idx and retain_idx
    # replace xxx with your path
    forget_idx = np.load('/data/xxx/github/corrective-unlearning-bench/debug_files/detected_poison_indices.npy')
    full_idx = np.arange(len(train_labels))
    retain_idx = np.setdiff1d(full_idx, forget_idx)
    forget_idx, retain_idx = torch.from_numpy(forget_idx), torch.from_numpy(retain_idx) 
    print(f"After remove {len(forget_idx)} detected poisons, there are {len(retain_idx)} examples left")
    print("start retraining from scratch on these filtered data...")

    retrain_loader = torch.utils.data.DataLoader(wtrain_manip_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(retain_idx), num_workers=4, pin_memory=True)
        
    # Stage 1: Pretraining
    opt.pretrain_file_prefix = opt.save_dir+'/'+opt.dataset+'_'+opt.model+'_'+opt.dataset_method+'_'+str(opt.forget_set_size)+'_'+str(opt.patch_size)+'_'+str(opt.pretrain_iters)+'_'+str(opt.pretrain_lr)
    if not exists(opt.pretrain_file_prefix):makedirs(opt.pretrain_file_prefix)

    if not exists(opt.pretrain_file_prefix + '/Manually_retrainmodel/model.pth'):
        opt.max_lr, opt.train_iters, expname, unlearn_method = opt.pretrain_lr, opt.pretrain_iters, opt.exp_name, opt.unlearn_method
        
        #We now actually pretrain by calling unlearn(), misnomer
        opt.unlearn_method, opt.exp_name = 'Naive', 'retrainmodel'
        method = getattr(methods, opt.unlearn_method)(opt=opt, model=model)
        method.unlearn(train_loader=retrain_loader, test_loader=test_loader)
        method.compute_and_save_results(train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader)
        opt.exp_name, opt.unlearn_method = expname, unlearn_method  
    else:
        print('==> Loading the retrained model!')
        model.load_state_dict(torch.load(opt.pretrain_file_prefix + '/Manually_retrainmodel/model.pth'))
        model.to(opt.device)
        print('==> Loaded the retrained model!')