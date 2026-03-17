# FourCastNet 部署指南

NVIDIA FourCastNet AI 天气预测模型完整部署指南。

## 📖 内容

- 硬件与环境配置
- ERA5 数据下载与预处理
- 模型定义 (FNO 架构)
- 训练脚本 (含 wandb 监控)
- 推理与可视化
- TensorRT 加速

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/datombc/FourCastNet-Deployment-Guide.git
cd FourCastNet-Deployment-Guide

# 创建环境
conda create -n fourcastnet python=3.10 -y
conda activate fourcastnet

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-modulus numpy xarray netCDF4 h5py tqdm wandb

# 下载数据
python download_era5.py

# 预处理
python preprocess.py

# 训练
python train.py

# 推理
python infer.py
```

## 📄 文档

完整部署指南请查看：[FourCastNet部署指南.md](./FourCastNet部署指南.md)

## 📚 参考资料

- [FourCastNet 论文](https://arxiv.org/abs/2202.11214)
- [NVIDIA Modulus](https://github.com/NVIDIA/modulus)
- [NVIDIA Earth-2](https://nvidia.com/en-us/data-center/earth-2)

## 📝 License

MIT License