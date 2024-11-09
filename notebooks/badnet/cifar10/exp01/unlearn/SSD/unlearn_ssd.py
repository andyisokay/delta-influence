import torch, timm
import numpy as np
import sys
import copy
import os
from os import makedirs
from os.path import exists
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

user_name = "xxx"
repo_dir = f"/data/{user_name}/github/poisoning-gradient-matching/open_source_delta_influence"  

sys.path.append(f'{repo_dir}/helpers/corrective-unlearning-bench/src')
import methods, resnet
from opts import parse_args 
from utils import seed_everything, SubsetSequentialSampler, get_targeted_classes  
from dataset import load_dataset, DatasetWrapper, manip_dataset, get_deletion_set

os.chdir(repo_dir)
print(os.getcwd())
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
import forest

# -------------------------------------  unlearn args  --------------------------------------- #
"""
dataset = 'CIFAR10'
num_classes = 10
unlearn_method = 'SSD' # choices=['Naive', 'EU', 'CF', 'Scrub', 'BadT', 'SSD', 'ActivationClustering', 'SpectralSignature', 'DeltaInfluence']
SSDselectwt = 2 # SSD: alpha aka selection weight, lower leads to more forgetting  [0.1, 1, 10, 50, 100, 500, 1000, 1e4, 1e5, 1e6]
SSDdampening = 0.1 * SSDselectwt # SSD: lambda aka dampening constant, lower leads to more forgetting  [0.1α, 0.5α, α, 5α, 10α]
"""
model_name = 'ResNet18'
dataset_method = None # we prepared the poisoned set already
forget_set_size = None # Number of samples to be manipulated
deletion_size = None # Number of samples to be deleted
k = -1 # 'All layers are freezed except the last-k layers, -1 means unfreeze all layers'
factor = 0.1 # Magnitude to decrease weights
### scrub ###
kd_T = 4 # Knowledge distilation temperature for SCRUB
alpha = 0.001 # KL from og_model constant for SCRUB, higher incentivizes closeness to ogmodel
msteps = 400 # Maximization steps on forget set for SCRUB
### unsure method ###
rsteps = 800 # InfRe when to stop retain set gradient descent
ascLRscale = 1.0 # AL/InfRe: scaling of lr to use for gradient ascent

# Optimizer Params
batch_size = 512 # input batch size for training
pretrain_iters = 40 # number of epochs to train, 7500
train_iters = 40
unlearn_iters = 40 # number of epochs to train (unlearning), 1000
pretrain_lr = 0.025
unlearn_lr = 0.025
wd = 0.0005
# Defaults
data_dir = '../data/'
save_dir = '../logs/'
exp_name = 'unlearn'
device = 'cuda'

class args_specify:
  def __init__(
        self,
        dataset,
        num_classes,
        unlearn_method,
        SSDselectwt,
        SSDdampening,
        # the rest are default values
        model_name=model_name,
        dataset_method=dataset_method,
        forget_set_size=forget_set_size,
        deletion_size=deletion_size,
        k=k,
        factor=factor,
        kd_T=kd_T,
        alpha=alpha,
        msteps=msteps,
        rsteps=rsteps,
        ascLRscale=ascLRscale,
        batch_size=batch_size,
        pretrain_iters=pretrain_iters,
        train_iters=train_iters,
        unlearn_iters=unlearn_iters,
        pretrain_lr=pretrain_lr,
        unlearn_lr=unlearn_lr,
        wd=wd,
        data_dir=data_dir,
        save_dir=save_dir,
        exp_name=exp_name,
        device=device,
            ):
        self.model = model_name
        self.dataset = dataset
        self.dataset_method = dataset_method
        self.unlearn_method = unlearn_method
        self.num_classes = num_classes
        self.forget_set_size = forget_set_size
        self.deletion_size = deletion_size
        self.k = k
        self.factor = factor
        self.kd_T = kd_T
        self.alpha = alpha
        self.msteps = msteps
        self.SSDdampening =SSDdampening
        self.SSDselectwt = SSDselectwt
        self.rsteps = rsteps
        self.ascLRscale = ascLRscale
        self.batch_size = batch_size
        self.pretrain_iters = pretrain_iters
        self.train_iters = train_iters
        self.unlearn_iters = unlearn_iters
        self.pretrain_lr = pretrain_lr
        self.unlearn_lr = unlearn_lr
        self.wd = wd
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.device = device

# --------------------------------------  model args  ---------------------------------------- #
"""
net = ['ResNet18']
dataset = 'CIFAR10'
"""
recipe = 'gradient-matching'
threatmodel = 'single-class'
poisonkey = None
modelkey = None
eps = 16
budget = 0.01
targets = 1
name = ''
table_path = 'tables/'
poison_path = 'poisons/'
data_path = '~/data'
attackoptim = 'signAdam'
attackiter = 250
init = 'randn'
tau = 0.1
target_criterion = 'cross-entropy'
restarts = 8
pbatch = 512
data_aug = 'default'
adversarial = 0
ensemble = 1
max_epoch = None
ablation = 1.0
loss = 'similarity'
centreg = 0
normreg = 0
repel = 0
nadapt = 2
vruns = 1
vnet = None
optimization = 'conservative'
epochs = 40
gradient_noise = None
gradient_clip = None
lmdb_path = None
benchmark = ''
benchmark_idx = 0
save = None
local_rank = None
pretrained = False
noaugment = False
cache_dataset = False
pshuffle = False
dryrun = False

class model_args_specify:
  def __init__(
        self,
        net,
        dataset,
        # the rest are default values
        recipe=recipe,
        threatmodel=threatmodel,
        poisonkey=poisonkey,
        modelkey=modelkey,
        eps=eps,
        budget=budget,
        targets=targets,
        name=name,
        table_path=table_path,
        poison_path=poison_path,
        data_path=data_path,
        attackoptim=attackoptim,
        attackiter=attackiter,
        init=init,
        tau=tau,
        target_criterion=target_criterion,
        restarts=restarts,
        pbatch=pbatch,
        data_aug=data_aug,
        adversarial=adversarial,
        ensemble=ensemble,
        max_epoch=max_epoch,
        ablation=ablation,
        loss=loss,
        centreg=centreg,
        normreg=normreg,
        repel=repel,
        nadapt=nadapt,
        vruns=vruns,
        vnet=vnet,
        optimization=optimization,
        epochs=epochs,
        gradient_noise=gradient_noise,
        gradient_clip=gradient_clip,
        lmdb_path=lmdb_path,
        benchmark=benchmark,
        benchmark_idx=benchmark_idx,
        save=save,
        local_rank=local_rank,
        pretrained=pretrained,
        noaugment=noaugment,
        cache_dataset=cache_dataset,
        pshuffle=pshuffle,
        dryrun=dryrun
            ):
        self.net = net
        self.dataset = dataset
        self.recipe = recipe
        self.threatmodel = threatmodel
        self.poisonkey = poisonkey
        self.modelkey = modelkey
        self.eps = eps
        self.budget = budget
        self.targets = targets
        self.name = name
        self.table_path = table_path
        self.poison_path = poison_path
        self.data_path =data_path
        self.attackoptim = attackoptim
        self.attackiter = attackiter
        self.init = init
        self.tau = tau
        self.target_criterion = target_criterion
        self.restarts = restarts
        self.pbatch = pbatch
        self.data_aug = data_aug
        self.adversarial = adversarial
        self.ensemble = ensemble
        self.max_epoch = max_epoch
        self.ablation = ablation
        self.loss = loss
        self.centreg = centreg
        self.normreg = normreg
        self.repel = repel
        self.nadapt = nadapt
        self.vruns = vruns
        self.vnet = vnet
        self.optimization = optimization
        self.epochs = epochs
        self.gradient_noise = gradient_noise
        self.gradient_clip = gradient_clip
        self.lmdb_path = lmdb_path
        self.benchmark = benchmark
        self.benchmark_idx = benchmark_idx
        self.save = save
        self.local_rank = local_rank
        self.pretrained = pretrained
        self.noaugment = noaugment
        self.cache_dataset = cache_dataset
        self.pshuffle = pshuffle
        self.dryrun = dryrun

# -------------------------------------------------------------------------------------------- #
def getVictimModel(exp_name, net, dataset, attack_method):
    # get the poisoned model
    model_args = model_args_specify(net=net, dataset=dataset)
    model_setup = forest.utils.system_startup(model_args)
    model_exp = exp_name
    load_model_name = 'victim.pth'
    models_dir = f'{repo_dir}/notebooks/{attack_method}/{model_exp}/models/'
    victim_dir = models_dir + load_model_name
    model = forest.Victim(model_args, setup=model_setup)
    model = model.load_model(victim_dir)
    model = model.to(device)
    print(f"victim model loaded... ({victim_dir})")
    return model, model_args, model_setup, victim_dir

def check_detection_acc(detected_idx, manip_idx):
    set1 = set(detected_idx.tolist())
    set2 = set(manip_idx.tolist())

    common_elements = set1.intersection(set2)
    common_tensor = torch.tensor(list(common_elements))

    print(f" The algorithm returns {len(detected_idx)} poisons...\n")
    print(f" among them {len(common_tensor)} are true poisons (there're {len(manip_idx)} poisons in total -> [{(len(common_tensor)/(len(manip_idx)/100)):.2f}%] detected) \n\n their indices: {common_tensor.tolist()}\n")
    print(f" the other {len(detected_idx) - len(common_tensor)} are actually cleans...")

# (model_exp, victim_model, model_args, model_setup, victim_dir, dataset, num_classes, unlearn_method, SSDselectwt, SSDdampening, retain_loader, test_loader, delete_loader, delete_idx, victim_class_testloader, full_clean_testloader, manip_robust_vc_testloader)
def unlearn_and_eval(model_exp, attack_method, detect_method, model, model_args, model_setup, victim_dir, dataset, num_classes, unlearn_method, SSDselectwt, SSDdampening, retain_loader, test_loader, delete_loader, delete_idx, victim_class_testloader, full_clean_testloader, manip_robust_vc_testloader):
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(seed=3017)
    # get new opt
    opt = args_specify(
        dataset=dataset,
        num_classes=num_classes,
        unlearn_method=unlearn_method,
        SSDselectwt=SSDselectwt,
        SSDdampening=SSDdampening)
    if opt.device == 'cuda':
        assert(torch.cuda.is_available())
    
    opt.save_dir = f'{repo_dir}/notebooks/{attack_method}/{model_exp}/unlearn_logs'
    # opt.dataset_method = 'badnet' # xxx
    # detection_name = "Control_Group" # xxx
    opt.dataset_method = attack_method
    detection_name = detect_method 
    print(f"{opt.dataset_method}--{detection_name}--{model_exp}--{opt.model}--{opt.unlearn_method}--{opt.SSDselectwt}--{opt.SSDdampening}")
    opt.pretrain_file_prefix = opt.save_dir+'/'+opt.dataset+'_'+opt.model+'_'+opt.dataset_method+'_'+detection_name
    if not exists(opt.pretrain_file_prefix):makedirs(opt.pretrain_file_prefix)
    opt.exp_name = 'unlearn'
    opt.deletion_size = len(delete_idx)
    opt.train_iters = len(retain_loader) + len(delete_loader) # unlearn iters would be this according to SSD

    # unlearned_save_path = opt.pretrain_file_prefix+'/'+str(opt.deletion_size)+'_'+opt.unlearn_method+'_'+opt.exp_name+'_'+str(opt.train_iters)
    unlearned_save_path = opt.pretrain_file_prefix+'/'+str(opt.deletion_size)+'_'+opt.unlearn_method+'_'+opt.exp_name+'_'+str(opt.train_iters)+ '_' + str(opt.SSDselectwt) + '_' + str(opt.SSDdampening)
    if not exists(unlearned_save_path):
        opt.max_lr = opt.pretrain_lr
        method = getattr(methods, opt.unlearn_method)(opt=opt, model=model)
        # method.unlearn(train_loader=poisoned_train_loader, test_loader=test_loader, eval_loaders=None) # <---- BadT
        method.unlearn(train_loader=retain_loader, test_loader=test_loader, forget_loader=delete_loader, eval_loaders=None) # <---- Scrub
        # method.unlearn(train_loader=retain_loader, test_loader=test_loader, forget_loader=delete_loader)
        # method.compute_and_save_results(train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader)
        method.save_results(test_loader)
    else:
        print(f"unlearned model already found at: {unlearned_save_path}")
    
    # eval
    # load saved unlearned model
    unlearned_model = forest.Victim(model_args, setup=model_setup)
    unlearned_model = unlearned_model.load_model(victim_dir)
    unlearned_model.load_state_dict(torch.load(unlearned_save_path + '/unlearned_model.pth'))
    unlearned_model.to(device)
    unlearned_adv_acc, unlearned_clean_acc, unlearned_robust_acc = eval_unlearn_performance(unlearned_model, victim_class_testloader, full_clean_testloader, manip_robust_vc_testloader)
    return unlearned_adv_acc, unlearned_clean_acc, unlearned_robust_acc

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(f"correct: {correct}")
    # print(f"total: {total}")
    return 100.0 * correct / total

def eval_unlearn_performance(unlearned_model, victim_class_testloader, full_clean_testloader, manip_robust_vc_testloader):
    unlearned_adv_acc = calculate_accuracy(unlearned_model, victim_class_testloader)
    # print(f"unlearned_adv_acc: {unlearned_adv_acc}%")
    # clean acc
    unlearned_clean_acc = calculate_accuracy(unlearned_model, full_clean_testloader)
    # print(f"unlearned_clean_acc: {unlearned_clean_acc}%")
    # manip_robust
    unlearned_robust_acc = calculate_accuracy(unlearned_model, manip_robust_vc_testloader)
    # print(f"unlearned_robust_acc: {unlearned_robust_acc}%")
    return unlearned_adv_acc, unlearned_clean_acc, unlearned_robust_acc









    
    
    
