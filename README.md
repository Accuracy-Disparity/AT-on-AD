# Understanding the Impact of Adversarial Training on Accuracy Disparity

This repository offers the official implementation for the NeurIPS 2022 submission "**Understanding the Impact of Adversarial Training on Accuracy Disparity**". Concretely, it contains the *datasets* and *code* for reproducing all the experimental results presented in both the main paper and the appendix.

## Datasets

We provide six groups of datasets, including three groups of real-world datasets---MNIST, FMNIST, CIFAR-10, and three groups of synthetic datasets---mixture of Gaussian, Cauchy, or Holtsmark. The folder names for these datasets in this repository are `data_mnist`, `data_fmnist`, `data_cifar`, `data_syn`, `data_cauchy_2`, and `data_levy_1.5`. 

For each group of datasets, we provide both the balanced one (in `{folder}/data_R_1.pth`) and the imbalanced ones (in `{folder}/data_R_{2,5,10}.pth`).

More details regarding the construction of the dataset can be found in Appendix D.1 Datasets.

## Code

### Step 1. Model Training and Evaluation

We first obtain the **standard models** and **robust models** under certain specifications (details later) and evaluate the performance of these models.

Below, we offer an example via the synthetic mixture of Gaussian. The command arguments are similar for other datasets.

```bash
## Standard Model
# Training
python train_syn.py --seed $1 --n-epochs 500 --R $2 --lr $3
# Evaluation
python test_syn.py  --seed $1 --n-epochs 500 --R $2 --lr $3


## Robust Model
# Training
python train_syn.py 	--seed $1 --n-epochs 500 \
			--adv --norm-type $2 \
			--norm-scale $3 --R $4 --lr $5
# Evaluation
python test_syn.py	--seed $1 --n-epochs 500 \
			--adv --norm-type $2 \
			--norm-scale $3 --R $4 --lr $5
```

Note that when training the robust models, we need to configure three more arguments/options: `--adv`, `--norm-type`, and `--norm-scale`, which are needed for adversarial training. Specifically, we offer two options for ``--norm-type``: $\ell_2$ or $\ell_\infty$. 

### Step 2. Verification of Research Questions

After we evaluate the trained models, we analyze their performance to obtain answers to the research questions. Essentially, we compute the gap of accuracy disparity (1st row in Fig. 1) and the gap of standard accuracy (2nd row in Fig. 1); we also document some more intermediate metrics to help understanding.

The script for analyzing the models is located at [`process_data.py`](./process_data.py). This script is suited to the set of parameters for different datasets used in our experiments (details in Appendix D.1 Training Protocols). 

For readers that want to experiment with their own datasets and parameters, we recommend them to adapt the training and evaluation code, as well as the analysis script for their own needs.
