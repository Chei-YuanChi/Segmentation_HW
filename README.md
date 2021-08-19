# IIR新生訓練 Segmentation_HW
## 改良Unet，使3-fold平均Dice高於baseline(0.71)
### 方式
1. 增加模型深度
2. 改良encoder backbone： Resnet
3. 加入attention機制 ： SE networks
* #### Squeeze-and-Excitation Networks
![](https://i.imgur.com/38UHB2f.png)
```
class SE(nn.Module):
  def __init__(self, kernels, reduction=16):
    super(SE, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential( 
      nn.Linear(kernels, kernels // reduction, bias=False), 
      nn.ReLU(inplace=True),    
      nn.Linear(kernels // reduction, kernels, bias=False),  
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c) 
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)
```
* SE + Resnet
![](https://i.imgur.com/P38ISjS.png)

```
class SELayer(nn.Module):
  def __init__(self, inplanes, kernels, stride, padding, reduction=16):
    super(SELayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Sequential(
      conv3x3_bn_relu(inplanes, kernels, stride, padding),
      conv3x3_bn_relu(kernels, kernels, stride, padding),
    )   // 原模型
    self.fc = nn.Sequential( 
      nn.Linear(kernels, kernels // reduction, bias=False), 
      nn.ReLU(inplace=True),    
      nn.Linear(kernels // reduction, kernels, bias=False),  
      nn.Sigmoid()
    )

  def forward(self, input):
    x = self.conv(input)
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c) 
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x) + input
```
* 增加深度
```
class down(nn.Module):
  def __init__(self, inplanes, kernels, stride, padding = 1, reduction = 16, pooling = True):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(inplanes, kernels, kernel_size = 3, stride = stride, padding = padding),
        SELayer(kernels, kernels, stride, padding, reduction),
        SELayer(kernels, kernels, stride, padding, reduction),
    )
    self.max_pooling = nn.MaxPool2d(2) if pooling else nn.Identity()
    
  def forward(self, x):
    return self.max_pooling(self.conv(x))
```
### 結果比較
* 原模型 3-fold 平均 Dice
![](https://i.imgur.com/7mRCCaE.png)
* 加入 SE 及 Resnet backbone
![](https://i.imgur.com/VwWgnhG.png)
* 加入 SE 及 Resnet backbone 後增加深度
![](https://i.imgur.com/Wv0opXS.png)
