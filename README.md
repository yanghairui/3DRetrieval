# Sequential learning for sketch-based 3D model retrieval

[[Paper]](https://link.springer.com/article/10.1007/s00530-021-00871-w)

[[Data]](https://drive.google.com/drive/folders/19NJdl-4OG0unmjRXxFE_mPBybd0T7C3N?usp=sharing)

## Data Preparation
将3D模型转化成多投影视图表示[1]，./dataset/render目录下有相关的渲染代码。

## Method

方法分成两个阶段：

1. 第一阶段通过下面的度量损失函数学习到较强的3D模型特征和其类别表示向量。
   
   $$\mathcal L_d = \frac{1}{N_m}\sum_{i=1}^{N_m} \max(0, m + D(f_i^m, c_{y_i^m}) - \min_{j\in C, j \neq y_i^m} D(f_i^m, c_{j})) + D(f_i^m, c_{y_i^m})$$ 
   
   其中， $c_{y_i^m}$ 表示所属 $y_i^m$ 的类别表示向量， $f_i^m$ 表示的是样本 $x_i^m$ 经过CNN得到特征向量， $D(,)$ 采用的是余弦距离。所有特征向量均进行 $L_2$ 正则化。

2. 通过相关性损失函数使得sketch尽可能地靠近同类别的3D模型类别表示。
   
   $$\mathcal L_c = \frac{1}{N_s}\sum_{i=1}^{N_s} D(f_i^s, c_{y_i})$$ 
   
   其中 $f_i^s$ 表示的是sketch样本 $x_i^s$ 经过CNN得到特征向量， $c_{y_i}$ 表示类别 $y_i$ 的3D模型类别向量。

## Training and Testing

### Contents

源代码文件结构，其中数据的组织结构参见ycf@210.30.96.136:/data/ycf/data/目录下SHREC_2013、2014、2016数据集。

![NpPytI.png](https://s1.ax1x.com/2020/06/15/NpPytI.png)

以SHREC 2013数据集为例：
 
 1. for 3D Model  
 ```
 python shrec13_alexnet_15_15.py --config experiments/shrec_2013/model/alexnet/config.yaml
 ```
 
 2. for 2D Sketch 
 ```
 python shrec13_im_norm_union_28_alexnet.py --config experiments/shrec_2013/image/alexnet/config.yaml --center_path <1中训练后的权重文件>
 ```
 
## Reference
[1] Hang Su, etc. Multi-view Convolutional Neural Networks for 3D Shape Recognition. ICCV 2015.
