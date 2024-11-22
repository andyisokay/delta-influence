# delta-influence
This repository contains code implementation for the paper: 

***Delta-Influence: Unlearning Poisons via Influence Functions***

You can follow the below step-by-step guideline to replicate our experiments on "cifar10+badnet" which includes all code for attack, detection, unlearning and eval. 

Notebooks for other "{dataset}+{attack}" will be updated in the future, (currently we provide "cifar10+badnet", "cifar100+frequency attack, "imagenette+witches' brew") but essentially they are similar so you can definitely try some different datasets, attack methods and unlearn algorithms:)

## Example
### Setup
```
conda create -n delta-influence-env python=3.12  
conda activate delta-influence-env  
pip install -e .
```

*Credits: We utilize the [Kronfluence](https://github.com/pomonam/kronfluence) to calculate influence matrix and the [Corrective-Unlearning-Bench](https://github.com/drimpossible/corrective-unlearning-bench) for unlearning, so please make sure you have them installed before moving on*

### Prepare the poisoned dataset
"poison_dataset.ipynb" shows how to inject badnet poison into the cifar10 dataset and also provides training scripts to get the victim model

### Detect poisons 
"delta_influence.ipynb" implements the delta-influence algorithm, which will return you the most responsible examples for the poisoning behavior

Besides, we also provide implementations of other popular detection methods, as well as the threshold baseline mentioned in the paper: 
- [*Activation Clustering*](https://arxiv.org/abs/1811.03728)
- [*Spectral Signature*](https://arxiv.org/abs/1811.00636)
- [*Frequency Analysis*](https://github.com/YiZeng623/frequency-backdoor)
- *Influence Threshold*

To check the ablation studies, relavant notebooks can be found named "modify_images.ipynb" and "modify_labels.ipynb"

### Unlearn
For each combination of "{dataset}+{attack}+{detection}", we compare the unlearning effectiveness of 5 different corrective unlearning methods:

- [*Exact Unlearning (EU)*](https://arxiv.org/abs/2201.06640)
- [*Catastrophic Forgetting (CF)*](https://arxiv.org/abs/2201.06640)
- [*Selective Synaptic Dampening (SSD)*](https://arxiv.org/abs/2308.07707)
- [*SCRUB*](https://arxiv.org/abs/2003.02960)
- [*Bad Teacher (BadT)*](https://arxiv.org/pdf/2302.09880) 

