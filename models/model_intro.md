# 模型架构详解 (Model Architectures)

本项目采用了三种经典的轻量级卷积神经网络作为骨干网络（Backbone），用于从双能 X 射线图像（低能 + 高能）中提取特征并回归预测矿石品位。

所有模型均基于 `torchvision` 的预训练模型（ImageNet 权重）进行微调，并针对本项目的双通道输入和多目标回归任务进行了特定的结构修改。

## 1. ResNet18

ResNet18 是一个经典的深度残差网络，以其结构简单、训练稳定著称。

*   **基础架构**: ResNet18 (18层深度)
*   **输入层修改**:
    *   原始 ResNet18 的第一层卷积 (`conv1`) 接受 3 通道 (RGB) 输入。
    *   **本项目修改**: 将 `conv1` 替换为接受 **2 通道** (Low Energy + High Energy) 输入的卷积层。
    *   *参数初始化*: 使用 Kaiming Normal 初始化新卷积层的权重。
*   **输出层 (回归头) 修改**:
    *   原始全连接层 (`fc`) 被移除。
    *   **本项目修改**: 替换为一个新的序列结构：
        1.  `Dropout(p=0.5)`: 用于防止过拟合。
        2.  `Linear(in_features=512, out_features=num_targets)`: 输出预测的品位值。

```python
# 结构示意
self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.backbone.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, num_targets)
)
```

## 2. ResNet18-Spatial (空间感知版)

针对“黄铁矿（弥散分布）”与“方铅矿/闪锌矿（集中分布）”的分类难题，我们在 ResNet18 基础上引入了 **空间金字塔池化 (Spatial Pyramid Pooling, SPP)**。

*   **基础架构**: ResNet18 (去掉了最后的平均池化层和全连接层)
*   **SPP 模块**:
    *   将最后一层特征图（通常为 4x4 大小）分别进行三个尺度的池化：
        1.  **1x1 网格** (全局平均)：捕获整图特征。
        2.  **2x2 网格** (区域平均)：捕获四个象限的特征。
        3.  **4x4 网格** (精细平均)：捕获局部细节特征。
    *   **特征拼接**: 将上述池化结果拉平并拼接，形成一个包含非常丰富的空间分布信息的特征向量。
*   **优势**: 能够有效区分在空间上分布不同的矿物，而不仅仅看总量。

## 3. MobileNetV3-Small

MobileNetV3 是专为移动端和嵌入式设备设计的高效网络，采用了深度可分离卷积和 SE (Squeeze-and-Excitation) 模块。

*   **基础架构**: MobileNetV3-Small
*   **输入层修改**:
    *   原始输入层位于 `features[0][0]`。
    *   **本项目修改**: 替换为 **2 通道** 输入的卷积层，保持原有的 kernel size 和 stride。
*   **输出层 (回归头) 修改**:
    *   原始分类器 (`classifier`) 的最后一层是一个线性层。
    *   **本项目修改**: 将最后一层 (`classifier[3]`) 替换为适应目标数量的线性层。

```python
# 结构示意
self.backbone.features[0][0] = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False)
# Classifier: Linear -> Hardswish -> Dropout -> Linear (Modified)
self.backbone.classifier[3] = nn.Linear(1024, num_targets)
```

## 3. EfficientNet-B0

EfficientNet 通过复合缩放方法（Compound Scaling）平衡了网络深度、宽度和分辨率，旨在以更少的参数量获得更好的性能。

*   **基础架构**: EfficientNet-B0
*   **输入层修改**:
    *   原始输入层位于 `features[0][0]`。
    *   **本项目修改**: 替换为 **2 通道** 输入的卷积层。
*   **输出层 (回归头) 修改**:
    *   原始分类器 (`classifier`) 由 Dropout 和 Linear 层组成。
    *   **本项目修改**: 将末端的线性层 (`classifier[1]`) 替换为适应目标数量的线性层。

```python
# 结构示意
self.backbone.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)
# Classifier: Dropout -> Linear (Modified)
self.backbone.classifier[1] = nn.Linear(1280, num_targets)
```

## 总结

| 模型 | 参数量 | 特点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **ResNet18** | 中等 | 结构成熟，特征提取能力强，训练稳定 | 通用，对计算资源不极度敏感的场景 |
| **ResNet18-Spatial** | 中等+ | **增强了空间分布感知能力** | 区分弥散分布（如黄铁矿）与集中分布（如方铅矿）的难点任务 |
| **MobileNetV3-Small** | 极小 | 速度极快，模型体积小 | 边缘计算，实时性要求极高的场景 |
| **EfficientNet-B0** | 小 | 参数利用率高，在相同计算量下精度通常优于 MobileNet | 平衡精度与速度的最佳选择 |
