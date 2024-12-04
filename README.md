# Dirichlet network for microstructure segmentation of aluminum alloy
This is the source code of "Effective Microstructure Segmentation and Measurement Method Based on Uncertainty Estimation for Aluminum Alloy Metallographic Images". 

### Introduction:

This study introduces a novel approach for measuring alloy microstructures by integrating uncertainty estimation technology. 
Leveraging the properties of the Dirichlet distribution, we propose a maximum likelihood loss function combined with an L1 regularization penalty term. 
The maximum likelihood function optimizes the model parameters by targeting the expectation of the categorical distribution. 
Meanwhile, the regularization term employs a truncation strategy to mitigate potential overfitting, thereby enhancing the overall prediction accuracy. 
In our experiments, when combined with the proposed post-processing module, our method demonstrates superior segmentation performance and precise measurement results. 
Additionally, it exhibits advantages in terms of model parameters and prediction time. 
These findings suggest that our approach effectively segments silicon particles while capturing inherent uncertainty, ultimately improving prediction reliability. 
This underscores the potential practical value of the proposed model in real-world applications.

### Prerequisites

- pytorch >= 1.12.1(Our code is based on the 1.12.1)
- numpy >= 1.21.6

### Train and Evaluation
1. Clone this repository to local

2. Aluminum alloy metallographic image dataset, or custom image segmentation dataset.

3. For the Dirichlet network proposed in this paper, set "model.mode" in file train_lmc.py to "dir".

   (Option) For the variational Bayesian network, set "model.mode" in file train_lmc.py to "BBB" or "MCDrop".

   (Option) For the deep ensemble, set "model.mode" in file train_lmc.py to "Ensemble".

4. Execute the training file train_lmc.py until the termination condition is reached (training epochs is 100).

During and after training, the predictions are saved and uncertainty maps are constructed based on entropy and maximum probability.

### Aluminum alloy metallographic image dataset
The metallographic dataset used in this study comes from an alloy manufacturing company. 
For research purposes, the complete dataset can be requested from the author by emailing limingchun_cn@qq.com ([Mingchun Li](https://orcid.org/0000-0001-7780-3213)) or liuyang999@pku.edu.cn (Yang Liu).


### Results on Aluminum alloy Dataset
| Method | mDice  | AUROC(MaxP) |
|:-----|:------:|:-----:| 
| Baseline (UNet) | 0.7908 | 0.9633 |
| Deep ensemble | 0.8037 | 0.9649  |
| Dataset Splitting | 0.7996 | 0.9636 |
| Bayesian Network (Gaussian Prior) | 0.7971 | 0.9657 |
| Bayesian Network (Laplacian Prior) | 0.7988 | 0.9656  |
| MC-Dropout | 0.8072 | 0.9659 |
| Dirichlet network | 0.8198 | 0.9661 |

### Final models
This is the final model and log file in our paper. We used this model to evaluate. You can download by:
https://pan.baidu.com/s/1Ruc0Z7D9_Y9UZ0cWSzwQEg?pwd=as9p code: as9p 

### Notes
The current method is implemented using the UNET model and verified on an aluminum alloy metallographic image dataset. 
Due to corporate interests, it is difficult to publish the complete dataset we used. 
Please do not use the obtained data for commercial purposes.

### References
[1] <a href="https://link.springer.com/article/10.1007/s10462-023-10562-9">A survey of uncertainty in deep neural networks.</a>

[2] <a href="https://arxiv.org/abs/1612.01474">Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.</a>

[3] <a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation.</a>

