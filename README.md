# Contrastive Learning of Adaptive Social Information Fusion for Recommender Systems

This is the PyTorch implementation for **AFCL** proposed in the paper **Contrastive Learning of Adaptive Social Information Fusion for Recommender Systems**.

> Yafang Li, Chenda Li, Caiyan Jia, Baokai Zu, Hongyuan Wang


##  Running environment

We develop our codes in the following environment:

- python==3.9.13
- numpy==1.23.1
- torch==1.11.0
- scipy==1.9.1
- torch-sparse==0.6.17

## Datasets

| Dataset    | # Users  | # Items   | # Inter |# Relation  | Density of $E_I$ |   
|------------|----------|-----------|---------|------------|------------------|
| LastFM     | 1,892    | 17,632    | 92,834  | 25.434     |  0.278%          |
| Ciao       | 7,375    | 105,114   | 284,086 | 53,152     |  0.037%          |
| Douban     | 2,848    | 39,586    | 894,887 | 35,770     |  0.793%          |
| Douban-book| 13,024   | 22,347    | 792,062 | 169,150    |  0.272%          |
| Yelp       | 16,239   | 14,284    | 169,986 | 158,590    |  0.0732%         |
