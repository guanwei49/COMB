# COMB: Interconnected Transformers-Based Autoencoder for Multi-Perspective Business Process Anomaly Detection

This is the source code of our paper '[COMB: Interconnected Transformers-Based Autoencoder for Multi-Perspective Business Process Anomaly Detections](https://ieeexplore.ieee.org/abstract/document/10707475)'.

## Requirements
- [Python==3.7.13](https://pytorch.org)
- [PyTorch==1.13.0](https://pytorch.org)
- [NumPy==1.23.5](https://numpy.org)
- [scikit-learn==1.2.1](https://scikit-learn.org)
- [pandas==1.5.3](https://pandas.pydata.org/)

## Using Our Code
Utilizing the anomalous event logs located in the _**eventlogs**_ folder to obtain evaluation results (For
  reproducibility of the experiments).
```
    python main.py 
```

## Datasets

Five commonly used real-life logs:

i) **_[BPIC12](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)_**: Event log of a loan application process

ii) **_[BPIC13](https://doi.org/10.4121/uuid:a7ce5c55-03a7-4583-b855-98b86e1a2b07)_**: Logs of Volvo IT incident and problem management.

iii)  **_[BPIC17](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)_**: This event log pertains to a loan application process of a Dutch financial institute. The data
contains all applications filed through an online system in 2016 and their subsequent events until February 1st 2017, 15:11.


iv)  **_[Receipt](https://doi.org/10.4121/12709127.v2)_**: This log contains records of the receiving phase of the building permit application process in an
anonymous municipality.

v)  **_[Sepsis](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)_**: This log contains events of sepsis cases from a hospital.

Eight synthetic logs: i.e., **_Paper_**,  _**P2P**_, **_Small_**, **_Medium_**, **_Large_**, **_Huge_**, **_Gigantic_**,
and **_Wide_**.

The summary of statistics for each event log is presented below:

|     Log     | #Activities | #Traces |   #Events   | Max trace length | Min trace length | #Attributes | #Attribute values |
|:-----------:|:-----------:|:-------:|:-----------:|:----------------:|:----------------:|:-----------:|:-----------------:|
|  Gigantic   |    76-78    |  5000   | 28243-31989 |        11        |        3         |     1-4     |      70-363       |
|    Huge     |     54      |  5000   | 36377-42999 |        11        |        5         |     1-4     |      69-340       |
|    Large    |     42      |  5000   | 51099-56850 |        12        |        10        |     1-4     |      68-292       |
|   Medium    |     32      |  5000   | 28416-31372 |        8         |        3         |     1-4     |      66-276       |
|     P2p     |     13      |  5000   | 37941-42634 |        11        |        7         |     1-4     |      39-146       |
|    Paper    |     14      |  5000   | 49839-54390 |        12        |        9         |     1-4     |      36-128       |
|    Small    |     20      |  5000   | 42845-46060 |        10        |        7         |     1-4     |      39-144       |
|    Wide     |    23-34    |  5000   | 29128-31228 |        7         |       5-6        |     1-4     |      53-264       |
|   BPIC12   |     36      |  13087  |   262200    |       175        |        3         |      0      |         0         |
| BPIC13\_C  |      7      |  1487   |    6660     |        35        |        1         |      4      |        638        |
| BPIC13\_I  |     13      |  7554   |    65533    |       123        |        1         |      4      |       2144        |
| BPIC13\_O  |      5      |   819   |    2351     |        22        |        1         |      2      |        251        |
|   BPIC17   |     26      |  31509  |   1202267   |       180        |        10        |      1      |        149        |
|  Receipt   |     27      |  1434   |    8577     |        25        |        1         |      2      |        58         |
|   Sepsis   |     16      |  1050   |    15214    |       185        |        3         |      1      |        26         |

Logs containing artificial anomalies ranging from 5% to 45% are stored in the folder '**_eventlogs_**'. The file names
are formatted as _log_name_-_anomaly_ratio_-_ID_.


## To Cite Our Paper
```
@inproceedings{guan2024comb,
  title={COMB: Interconnected Transformers-Based Autoencoder for Multi-Perspective Business Process Anomaly Detection},
  author={Guan, Wei and Cao, Jian and Yao, Yan and Gu, Yang and Qian, Shiyou},
  booktitle={2024 IEEE International Conference on Web Services (ICWS)},
  pages={1115--1124},
  year={2024},
  organization={IEEE}
}
```