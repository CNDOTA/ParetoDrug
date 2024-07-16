## ParetoDrug — Official PyTorch Implementation

This repository contains  the **official PyTorch implementation** of the paper: **Enabling Target-Aware Molecule Generation to Follow Multi Objectives with Pareto MCTS**.

## Datasets

- [data/train-val-data.tsv](https://drive.google.com/drive/folders/1myoeLdsOYz8mSvYEhSdMfUszUJlaJR3u?usp=sharing): It contains all sequence pairs for training and validation.

- [data/train-val-split.json](https://github.com/CMACH508/AlphaDrug/blob/main/data/train-val-split.json): It contains the index of the training pairs and test pairs in [train-val-data.tsv](https://drive.google.com/drive/folders/1myoeLdsOYz8mSvYEhSdMfUszUJlaJR3u?usp=sharing).

- [data/testing-proteins-100.txt](https://github.com/CMACH508/AlphaDrug/blob/main/data/testing-proteins-100.txt): It contains all pdbids of the testing proteins which can be downloaded from [PDBbind website](http://www.pdbbind.org.cn/).


## Requirements

### Install
Pleases follow these commands to install the environment:

```
conda create -n paretodrug python=3.7

conda activate paretodrug

conda install -c conda-forge smina=2020.12.10 rdkit=2020.09.5 openbabel=3.1.1

conda install -c bioconda mmseqs2=13.45111

pip install biopython==1.79 pandas==1.3.4

pip install loguru biopython graphviz easydict tqdm scipy

pip3 install torch==1.13.1 torchvision torchaudio
```

## Model Training

- Before training, please make sure train-val-data.tsv is in the data folder.

- There are several key args for training listed as follows:
- 
    | Argument | Description | Default | Type |
    | :-----| :---- | :---- | :---- |
    | --layers | Number of layers in transformer | 4 | int |
    | --bs | Batch size | 32 | int |

- Train Lmser Transformer:

    ```shell
    cd your_project_path
    python train.py --layers 4 --bs 32 --device 0,1,2,3
    ```

## Pretrained Model

### We provide the pretrained model for LT (Lmser Transformer) as follows:
| Model  | Path |
| :----- | :---- | 
| Lmser Transformer | ./experiment/LT/model/30.pt|


## Run Monte Carlo Tree Search (MCTS)

The computational resources to run ParetoDrug normally are 1 GPU and 8 CPU cores.
The running time lasts for several hours as ParetoDrug performs MCTS and inferences with the pretrained generative model.
You can set a smaller 'st' parameter to reduce the running time.

### There are several key args for MCTS listed as follows:
| Argument | Description                        | Default | Type |
|:---------|:-----------------------------------|:--------|:-----|
| -k       | Protein index                      | 0       | int  |
| -st      | Number of simulation times in MCTS | 150     | int  |
| -p       | NN model path                      | LT      | str  |
| --max    | max mode or freq mode              | True    | bool |
| -g       | GPU index                          | 0       | int  |

### Multi-objective SBDD
Here is an example of running ParetoDrug on protein 1a9u (protein index 0 in test proteins) with 150 simulation times using the pretrained model LT in max mode with GPU 0.
```shell
python pareto_mcts.py -k 0 -g 0 -st 150 -p LT --max
```

### Multi-objective SBDD for the specified protein structure
If you want to generate molecules for your own PDB file, please provide the PDB file named #PDBid_protein.pdb and ligand file named #PDBid_ligand.sdf and put them in the "/data/test_pdbs/#PDBid/" folder, then run the following command with the parameter "--protein #PDBid" such as "--protein 1a9u".
Note that this will change the original protein index in "/data/test_pdbs".
```shell
python pareto_mcts_case.py --protein 1a9u -g 0 -st 150 -p LT --max
```

### Multi-target SBDD with case HIV
For the multi-target SBDD case study of finding HIV-related dual-inhibitor molecules, please run the following command.
```shell
python mt_pareto_mcts.py -q HIV
```

### Multi-target multi-objective SBDD with case Lapatinib
For the multi-target multi-objective SBDD case study of the drug Lapatinib, please run the following command.
```shell
python mtmo_pareto_mcts.py -q Lapatinib
```

## Acknowledgements
This repo is built upon the article: **AlphaDrug: protein target specific de novo molecular generation** and its repo https://github.com/CMACH508/AlphaDrug.
We thanks the authors of AlphaDrug for releasing their codes and data. Please also consider to cite it if you use our repo.


