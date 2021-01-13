# Bi-Tempered-Logistic-Loss-PyTorch
PyTorch implementation of Robust Bi-Tempered Logistic Loss Based on Bregman Divergences. Using as origin PyTroch loss.


Pytorch implementation of Robust Bi-Tempered Logistic Loss Based on Bregman Divergences

This repository contains the direct translation from Tensorflow to Pytorch of the paper "Robust Bi-Tempered Logistic Loss Based on Bregman Divergences" (https://arxiv.org/abs/1906.03361). The repo translates the official Tensorflow repo which can be found here: https://github.com/google/bi-tempered-loss and other repo can be found here: https://github.com/mlpanda/bi-tempered-loss-pytorch





```python
import torch

inputs = torch.randn(3, 5, requires_grad=True)
targets = torch.empty(3, dtype=torch.long).random_(5)

from bi_tempered_logistic_loss import BiTemperedLogisticLoss
loss_function = BiTemperedLogisticLoss(reduction='sum', t1=0.7, t2=1.3, label_smoothing=0.3)
loss = loss_function(inputs, targets)
```