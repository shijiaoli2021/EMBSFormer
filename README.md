## model

### ASTGFormer

#### abstract

Traffic flow prediction is crucial for traffic management and public property safety. In this paper, we propose ASTGFormer, a long-short historical similarity-informed graph convolution transformer for traffic flow predic-tion.Significantly, we differentiate the interplay between recent history, long-term history and future predictions, and introduce distinct methodologies for each segment. On the one hand, for the recent history data, we employ self-attention mechanisms and graph convolution to capture the dynamic spatio-temporal dependencies and adjacency relationships of the traffic flow. On the other hand, for the long-term historical data, we propose a long-short historical similarity attention mechanism to capture the similarity relationship between the long-term and recent historical patterns, thereby learning the periodic traffic patterns and complementing the prediction results. In addition, we also consider the impact of national statutory holidays in our model. Notably, ASTGFormer not only simplifies the handling of long-term historical data and significantly reduces the number of parameters but also achieves outstanding results compared with proposed base-lines in experiments conducted on three real-world datasets.
![ASTGFormer](./figures/framework.png)

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
python main.py --model_name ASTGFormer  --max_epochs 100 --batch_size 16 --data_name PEMS08 --gpus 1 
```

