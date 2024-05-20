## model

### EMBSFormer

#### abstract

Traffic flow prediction plays an important role in Intelligent Transportation Systems (ITS) to support traffic management and urban planning. There has been extensive successful work in this area. However, these approaches focus only on modeling the flow transition and ignore the flow generation process, which manifests itself in two ways: (i) The models are based on Markovian assumptions, ignoring the multi-periodicity of the flow generation in nodes. (ii) The same structure is designed to encode both the transition and generation processes, ignoring the differences between them. To address these problems, we propose EMBSformer, which full name is Effective Multi-Branch Similarity  Transformer for Traffic Flow Prediction. Through data analysis, we found that the factors affecting traffic flow include node-level traffic generation and graph-level traffic transition, which describe the multi-periodicity and trend of a node, respectively. Specifically, to capture traffic generation patterns, we propose a similarity analysis module that supports multi-branch encoding to dynamically expand significant cycles. For traffic transition, we employ a temporal and spatial self-attention mechanism to maintain global node interactions, and use GNN and 1d conv to model local node interactions, respectively. Model performance is evaluated on three real-world datasets on both long-term and short-term prediction tasks. Experimental results show that EMBSformer significantly outperforms baselines on the long-term prediction task, with an significant improvement. Moreover, compared to models based on flow transition modeling (e.g. GMAN, 513k) in short-term prediction, the variant of EMBSFormer(93K) uses nearly 20% of the parameters achieving the same performance.

## Datasets

### provide datasets:

PEMS04, PEMS07, PEMS08

1. PEMS-04:
   
   307 detectors  
   Jan to Feb in 2018  
   3 features: flow, occupy, speed.
   
2. PEMS-07:
   
   883 detectors  
   May to Augest in 2018  
   3 features: flow, occupy, speed.

2. PEMS-08:
   
   170 detectors  
   July to Augest in 2016  
   3 features: flow, occupy, speed.
   

## Requirements

| name              | version     |
| ----------------- | ----------- |
| python            |  3.8    |
| pytorch           | 1.13.1 |
| pytorch-lightning | 1.9.0       |
| numpy             | 1.23.5 |
| pandas            | 1.5.3  |
| matplotlib        | 3.7.1  |
| email             |             |
| smtplib           |             |
| configparser      |             |

## Train model

### PEMS08 as example

we can choose whether send email or plot predict figure with the key of "send_email" and "pre_curve_figure"

```python
python main.py --model_name EMBSFormer  --max_epochs 100 --batch_size 16 --data_name PEMS08 --gpus 1 
```

