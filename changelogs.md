# 更新日志

## 2026-02-03
- **新增分类指标**：基于可配置的阈值，实现了 Fe、Zn、Pb 的准确率（Accuracy）和召回率（Recall）计算。
- **新增加权指标**：利用矿石重量实现了加权准确率和加权召回率。
- **综合品位评估**：添加了综合品位评估逻辑（Pb+Zn > 3.0, Pb+Zn+Fe > 10.0）。
- 实现 `resnet18_spatial` 模型，引入 SPP (Spatial Pyramid Pooling) 模块以提取空间分布特征。
- 增加选矿专业指标 （回收率、抛废率等）
- **代码变更**：
    - `config.yaml`：新增 `metrics` 部分以配置阈值。
    - `data/dataset.py`：更新 `__getitem__` 以返回样本重量。
    - `train.py`：更新训练循环以处理返回的重量数据。
    - `visualize_results.py`：实现了包含综合品位在内的指标计算逻辑。
    - `run_pipeline.py`：更新最终汇总表，以包含 Acc%, Recall%, W_Acc%, W_Rec%；**并将结果保存为 TXT 文件**；**增加了模型选择逻辑**；**强制输出格式保留两位小数**。
    - `visualize_results.py`: 
         - 计算综合品位（Pb+Zn, Pb+Zn+Fe）的 MSE, MAE, R2 回归指标。
         - 将所有准确率和召回率乘以 100（百分比形式）并在列名增加 `%`。
         - 所有指标（包括 MSE/MAE/R2）保留两位小数。
    - `config.yaml`: 增加 `model_selection` 部分，可通过编号选择需要训练的模型。
    - `utils/reproducibility.py`: 新增随机种子设置函数 `set_seed(42)`，确保每次训练结果一致。
    - `models/model_intro.md`: 新增模型架构说明文档，详细描述了 ResNet18, MobileNetV3-Small, EfficientNet-B0 的结构修改。
    - **新功能**:
        - `models/network.py`: 实现 `resnet18_spatial` 模型，引入 SPP (Spatial Pyramid Pooling) 模块以提取空间分布特征。
        - `visualize_results.py`: 增加选矿专业指标：**金属回收率 (Recovery%)**、**废石抛废率 (Yield%)**、**尾矿品位 (Tail_Grade)**。
        - `config.yaml`: 更新模型列表，支持 `resnet18_spatial`。
