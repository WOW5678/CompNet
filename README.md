CompNet
====

This is the source code of paper "[Order-free Medicine Combination Prediction With Graph
Convolutional Reinforcement Learning](https://github.com/WOW5678/CompNet/blob/master/CompNet_cikm2019.pdf)".

Overview
====
CompNet is an end-to-end model mainly based on graph convolutional networks (GCN) and reinforcement learning (RL). Patient information and  drug-drug interactions knowledge are utilized to provide safe and personalized prediction for  medication combination. CompNet is tested on real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/).

Requirements
=====
pytorch >= 0.4

python >= 3.5

Running the code
=====

Data preprocessing
===
1. download [MIMIC data](https://mimic.physionet.org/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./MIMIC-III/
2. download [DDI data] (https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0) and put it in ./MIMIC-III/
3. run code ./process_MIMIC.py

CompNet
=====
run main_CompNet.py

Cite
====
Please cite our paper if you use this code in your own work:

```
@inproceedings{wang2019CompNet,
  title="{Order-free Medicine Combination Prediction With Graph Convolutional Reinforcement Learning}",
  author={Shanshan Wang and Pengjie Ren and Zhumin Chen and Zhaochun Ren and Jun Ma and Maarten de Rijke},
  Booktitle={{CIKM} 2019},
  year={2019}
}
```
