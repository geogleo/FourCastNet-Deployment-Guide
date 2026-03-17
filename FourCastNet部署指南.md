# FourCastNet 完整部署指南

---

## 🖥️ 硬件检查

```bash
# 检查 GPU
nvidia-smi

# 需要：CUDA 11.8+，显存 24GB+ (训练需多卡)
```

---

## 📦 环境配置

### 方式一：Conda + Pip (推荐新手)

```bash
# 1. 创建环境
conda create -n fourcastnet python=3.10 -y
conda activate fourcastnet

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 NVIDIA Modulus
pip install nvidia-modulus

# 4. 安装依赖
pip install numpy xarray netCDF4 h5py tqdm wandb

# 5. 验证安装
python -c "import modulus; print(modulus.__version__)"
```

### 方式二：Docker (推荐生产)

```bash
# 拉取官方镜像
docker pull nvcr.io/nvidia/modulus/modulus:24.01

# 运行容器 (挂载数据目录)
docker run --gpus all -it \
  -v ~/fourcastnet:/workspace \
  -v ~/era5_data:/data \
  nvcr.io/nvidia/modulus/modulus:24.01
```

### 方式三：从源码安装

```bash
# 克隆 Modulus
git clone https://github.com/NVIDIA/modulus.git
cd modulus

# 安装
pip install -e .

# 运行测试
pytest tests/
```

---

## 📁 项目结构

```bash
fourcastnet/
├── data/
│   ├── era5/
│   │   ├── train/
│   │   │   ├── 1979.nc
│   │   │   ├── 1980.nc
│   │   │   └── ...
│   │   └── stats/
│   │       ├── mean.nc
│   │       └── std.nc
│   └── processed/
│       └── era5_train.h5
├── configs/
│   └── fourcastnet.yaml
├── models/
│   └── pretrained/
│       └── fourcastnet.ckpt
├── train.py
├── infer.py
├── dataset.py
└── utils.py
```

---

## 📊 数据准备

### Step 1: 下载 ERA5 数据

```bash
# 安装 CDS API
pip install cdsapi

# 创建 API 配置 (~/.cdsapirc)
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF
```

### Step 2: 下载数据脚本

```python
# download_era5.py
import cdsapi
import os
from concurrent.futures import ThreadPoolExecutor

def download_year_month(year, month, variables, output_dir):
    """下载单月数据"""
    c = cdsapi.Client()
    filename = f"{output_dir}/{year}_{month:02d}.nc"
    
    if os.path.exists(filename):
        print(f"已存在: {filename}")
        return
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'variable': variables,
        },
        filename
    )
    print(f"下载完成: {filename}")

def download_era5(output_dir, years, variables):
    """批量下载 ERA5 数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for year in years:
            for month in range(1, 13):
                futures.append(
                    executor.submit(
                        download_year_month, 
                        year, month, variables, output_dir
                    )
                )
        for f in futures:
            f.result()

if __name__ == "__main__":
    # ERA5 单层变量
    variables = [
        '10m_u_component_of_wind',      # 10米风速 u
        '10m_v_component_of_wind',      # 10米风速 v
        '2m_temperature',                # 2米温度
        'surface_pressure',              # 地面气压
        'total_precipitation',           # 总降水
        'mean_sea_level_pressure',        # 海平面气压
    ]
    
    # 下载 1979-2018 训练数据
    download_era5(
        output_dir="./data/era5/train",
        years=range(1979, 2019),
        variables=variables
    )
    
    # 下载 2019-2022 验证数据
    download_era5(
        output_dir="./data/era5/val",
        years=range(2019, 2023),
        variables=variables
    )
```

```bash
# 运行下载
python download_era5.py
# 预计耗时：数天 (取决于网速和服务器负载)
```

### Step 3: 预处理数据

```python
# preprocess.py
import xarray as xr
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm

def compute_statistics(data_path, output_path):
    """计算训练集的均值和标准差"""
    files = sorted(Path(data_path).glob("*.nc"))
    
    # 第一次遍历：计算均值
    print("计算均值...")
    total_sum = None
    total_count = 0
    
    for f in tqdm(files):
        ds = xr.open_dataset(f)
        data = ds.to_array().values  # (var, time, lat, lon)
        
        if total_sum is None:
            total_sum = np.zeros(data.shape[0])  # 每个变量
        
        total_sum += data.sum(axis=(1, 2, 3))
        total_count += data.shape[1] * data.shape[2] * data.shape[3]
        ds.close()
    
    mean = total_sum / total_count
    
    # 第二次遍历：计算标准差
    print("计算标准差...")
    total_sq_sum = None
    
    for f in tqdm(files):
        ds = xr.open_dataset(f)
        data = ds.to_array().values
        
        if total_sq_sum is None:
            total_sq_sum = np.zeros(data.shape[0])
        
        total_sq_sum += ((data - mean[:, None, None, None]) ** 2).sum(axis=(1, 2, 3))
        ds.close()
    
    std = np.sqrt(total_sq_sum / total_count)
    
    # 保存统计量
    np.save(f"{output_path}/mean.npy", mean)
    np.save(f"{output_path}/std.npy", std)
    
    return mean, std

def convert_to_h5(data_path, output_path, mean, std):
    """将 NetCDF 转换为 HDF5 格式"""
    files = sorted(Path(data_path).glob("*.nc"))
    
    with h5py.File(output_path, 'w') as hf:
        # 第一次遍历获取总时间步数
        total_steps = 0
        for f in files:
            ds = xr.open_dataset(f)
            total_steps += ds.dims['time']
            ds.close()
        
        # 创建数据集
        # 假设 6 个变量, 720 lat, 1440 lon
        hf.create_dataset('data', shape=(total_steps, 6, 720, 1440), dtype=np.float32)
        
        # 写入数据
        idx = 0
        for f in tqdm(files):
            ds = xr.open_dataset(f)
            data = ds.to_array().values  # (var, time, lat, lon)
            
            # 标准化
            data = (data - mean[:, None, None, None]) / std[:, None, None, None]
            
            # 写入
            for t in range(data.shape[1]):
                hf['data'][idx] = data[:, t, :, :]
                idx += 1
            
            ds.close()
    
    print(f"预处理完成: {output_path}")

if __name__ == "__main__":
    # 计算统计量
    mean, std = compute_statistics(
        data_path="./data/era5/train",
        output_path="./data/era5/stats"
    )
    
    # 转换训练集
    convert_to_h5(
        data_path="./data/era5/train",
        output_path="./data/processed/era5_train.h5",
        mean=mean,
        std=std
    )
    
    # 转换验证集
    convert_to_h5(
        data_path="./data/era5/val",
        output_path="./data/processed/era5_val.h5",
        mean=mean,
        std=std
    )
```

---

## 🧠 模型定义

```python
# model.py
import torch
import torch.nn as nn
import numpy as np

class SpectralConv2d(nn.Module):
    """傅里叶卷积层"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 傅里叶模态数
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input, weights):
        """复数乘法"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # 傅里叶变换
        x_ft = torch.fft.rfft2(x)
        
        # 频域加权
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                              dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # 逆傅里叶变换
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """FNO 块"""
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()
    
    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        return self.act(x1 + x2)


class FourCastNet(nn.Module):
    """FourCastNet 模型"""
    def __init__(self, in_channels, out_channels, modes=12, width=256, num_layers=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        
        # 输入投影
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO 层
        self.layers = nn.ModuleList([
            FNOBlock(width, modes, modes) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        # x: (batch, in_channels, lat, lon)
        
        # 投影到宽度
        x = x.permute(0, 2, 3, 1)  # (batch, lat, lon, in_channels)
        x = self.fc0(x)  # (batch, lat, lon, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, lat, lon)
        
        # FNO 层
        for layer in self.layers:
            x = layer(x) + x  # 残差连接
        
        # 输出投影
        x = x.permute(0, 2, 3, 1)  # (batch, lat, lon, width)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (batch, out_channels, lat, lon)
        
        return x


class FourCastNetWrapper(nn.Module):
    """包装器：支持自回归预测"""
    def __init__(self, in_channels, out_channels, modes=12, width=256, num_layers=12):
        super().__init__()
        self.model = FourCastNet(in_channels, out_channels, modes, width, num_layers)
        
    def forward(self, x, num_steps=1):
        """自回归预测"""
        predictions = []
        
        for _ in range(num_steps):
            x = self.model(x)
            predictions.append(x)
        
        return predictions
```

---

## 🚂 训练脚本

```python
# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from model import FourCastNet, FourCastNetWrapper

class ERA5Dataset(Dataset):
    """ERA5 数据集"""
    def __init__(self, h5_path, stats_path, pred_step=1):
        self.h5_path = h5_path
        self.pred_step = pred_step
        
        # 加载统计量
        self.mean = np.load(f"{stats_path}/mean.npy")
        self.std = np.load(f"{stats_path}/std.npy")
        
        # 打开文件
        self.h5 = h5py.File(h5_path, 'r')
        self.length = self.h5['data'].shape[0] - pred_step
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 当前时刻
        x = self.h5['data'][idx]
        # 下一时刻
        y = self.h5['data'][idx + self.pred_step]
        
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    
    def close(self):
        self.h5.close()


def train(config):
    """训练函数"""
    # 初始化 wandb
    wandb.init(project="fourcastnet", config=config)
    
    # 设备
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    
    # 数据集
    train_dataset = ERA5Dataset(
        h5_path=config['train_data'],
        stats_path=config['stats_path'],
        pred_step=1
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 模型
    model = FourCastNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        modes=config['modes'],
        width=config['width'],
        num_layers=config['num_layers']
    ).to(device)
    
    # 多 GPU
    if config['multi_gpu']:
        model = nn.DataParallel(model)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            pred = model(x)
            loss = criterion(pred, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # 学习率调度
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        
        # 记录到 wandb
        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'lr': scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{config['save_dir']}/best_model.ckpt")
        
        # 定期保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{config['save_dir']}/checkpoint_epoch_{epoch+1}.ckpt")
    
    # 清理
    train_dataset.close()
    wandb.finish()
    
    print(f"训练完成！最佳损失: {best_loss:.6f}")


if __name__ == "__main__":
    config = {
        # 数据
        'train_data': './data/processed/era5_train.h5',
        'stats_path': './data/era5/stats',
        
        # 模型
        'in_channels': 6,
        'out_channels': 6,
        'modes': 12,
        'width': 256,
        'num_layers': 12,
        
        # 训练
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        
        # 系统
        'gpu': 0,
        'multi_gpu': False,
        'num_workers': 4,
        
        # 保存
        'save_dir': './models',
        'save_interval': 10,
    }
    
    # 创建保存目录
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    train(config)
```

---

## 🔮 推理脚本

```python
# infer.py
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from model import FourCastNet

class FourCastNetInference:
    """推理类"""
    def __init__(self, checkpoint_path, config, stats_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # 加载统计量
        self.mean = np.load(f"{stats_path}/mean.npy")
        self.std = np.load(f"{stats_path}/std.npy")
        
        # 加载模型
        self.model = FourCastNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            modes=config['modes'],
            width=config['width'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def normalize(self, x):
        """标准化"""
        return (x - self.mean[:, None, None]) / self.std[:, None, None]
    
    def denormalize(self, x):
        """反标准化"""
        return x * self.std[:, None, None] + self.mean[:, None, None]
    
    def predict(self, initial_state, num_steps):
        """
        自回归预测
        
        Args:
            initial_state: (var, lat, lon) 初始状态
            num_steps: 预测步数 (每步 6 小时)
        
        Returns:
            predictions: (num_steps, var, lat, lon) 预测序列
        """
        # 标准化
        x = self.normalize(initial_state)
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for step in range(num_steps):
                x = self.model(x)
                pred = x.cpu().numpy()[0]
                
                # 反标准化并保存
                pred_denorm = self.denormalize(pred)
                predictions.append(pred_denorm)
        
        return np.array(predictions)
    
    def predict_from_file(self, nc_path, num_steps):
        """从 NetCDF 文件预测"""
        # 加载初始状态
        ds = xr.open_dataset(nc_path)
        initial_state = ds.to_array().values[:, 0, :, :]  # (var, lat, lon)
        ds.close()
        
        return self.predict(initial_state, num_steps)
    
    def visualize(self, predictions, variable_idx=0, variable_name='Temperature'):
        """可视化预测结果"""
        num_steps = len(predictions)
        fig, axes = plt.subplots(1, min(num_steps, 5), figsize=(20, 4))
        
        for i, ax in enumerate(axes):
            if i < num_steps:
                im = ax.imshow(predictions[i, variable_idx], cmap='RdBu_r')
                ax.set_title(f'T+{(i+1)*6}h')
                plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{variable_name} Prediction')
        plt.tight_layout()
        plt.savefig('prediction.png', dpi=150)
        plt.close()


def main():
    # 配置
    config = {
        'in_channels': 6,
        'out_channels': 6,
        'modes': 12,
        'width': 256,
        'num_layers': 12,
    }
    
    # 初始化推理器
    inferencer = FourCastNetInference(
        checkpoint_path='./models/best_model.ckpt',
        config=config,
        stats_path='./data/era5/stats',
        device='cuda'
    )
    
    # 预测 7 天 (28 步 × 6小时)
    predictions = inferencer.predict_from_file(
        nc_path='./data/era5/val/2019_01.nc',
        num_steps=28
    )
    
    print(f"预测完成！形状: {predictions.shape}")
    # predictions: (28, 6, 720, 1440)
    
    # 可视化
    inferencer.visualize(predictions, variable_idx=2, variable_name='Temperature')


if __name__ == "__main__":
    main()
```

---

## ⚙️ 配置文件

```yaml
# configs/fourcastnet.yaml

# 数据配置
data:
  train_path: ./data/processed/era5_train.h5
  val_path: ./data/processed/era5_val.h5
  stats_path: ./data/era5/stats
  variables:
    - u10          # 10m 风速 u 分量
    - v10          # 10m 风速 v 分量
    - t2m          # 2m 温度
    - sp           # 地面气压
    - tp           # 总降水
    - msl          # 海平面气压
  resolution:
    lat: 720       # 0.25°
    lon: 1440      # 0.25°
  time_step: 6     # 小时

# 模型配置
model:
  name: FourCastNet
  in_channels: 6
  out_channels: 6
  modes: 12        # 傅里叶模态数
  width: 256       # 隐藏层宽度
  num_layers: 12   # FNO 层数

# 训练配置
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1.0e-3
  weight_decay: 1.0e-5
  grad_clip: 1.0
  scheduler: cosine
  warmup_epochs: 5

# 硬件配置
hardware:
  gpu_ids: [0, 1, 2, 3]  # 多 GPU
  num_workers: 8
  pin_memory: true

# 保存配置
save:
  save_dir: ./models
  save_interval: 10
  log_dir: ./logs
```

---

## 🏃 运行命令

```bash
# 1. 下载数据 (耗时较长)
python download_era5.py

# 2. 预处理数据
python preprocess.py

# 3. 训练模型
python train.py

# 4. 推理预测
python infer.py
```

---

## 📈 监控训练

```bash
# 启动 wandb
wandb login

# 在代码中已集成 wandb
# 训练时自动记录到 wandb 仪表盘
```

---

## ⚠️ 常见问题

| 问题 | 解决方案 |
|------|----------|
| **OOM (显存不足)** | 减小 batch_size 或 width |
| **训练不收敛** | 检查数据标准化，调小学习率 |
| **NaN 损失** | 检查数据是否含 NaN，增加 grad_clip |
| **推理慢** | 使用 TensorRT 加速 |
| **精度不够** | 增加 modes、width 或 num_layers |

---

## 🚀 进阶优化

### TensorRT 加速

```python
# export_onnx.py
import torch
from model import FourCastNet

model = FourCastNet(...)
model.load_state_dict(torch.load('best_model.ckpt')['model_state_dict'])
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 6, 720, 1440)
torch.onnx.export(
    model,
    dummy_input,
    'fourcastnet.onnx',
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# 转换 TensorRT
# trtexec --onnx=fourcastnet.onnx --saveEngine=fourcastnet.trt --fp16
```

---

## 📊 技术规格总结

| 参数 | 值 |
|------|-----|
| 空间分辨率 | 0.25° (720×1440 网格) |
| 时间步长 | 6 小时 |
| 输入变量 | 6 个 (可扩展) |
| 傅里叶模态数 | 12 |
| 隐藏层宽度 | 256 |
| FNO 层数 | 12 |
| 模型参数 | ~350M |
| 训练数据 | ERA5 (1979-2018) |
| 训练时长 | ~2 周 (64× A100) |
| 单步推理 | ~0.05s (A100) |
| 7天预测 | ~1.5s (A100) |

---

## 📚 参考资料

| 资源 | 链接 |
|------|------|
| FourCastNet 论文 | arxiv.org/abs/2202.11214 |
| NVIDIA Modulus | github.com/NVIDIA/modulus |
| ERA5 数据 | cds.climate.copernicus.eu |
| NVIDIA Earth-2 | nvidia.com/en-us/data-center/earth-2 |
| Modulus 文档 | docs.nvidia.com/deeplearning/modulus |

---

*文档创建时间: 2026-03-17*