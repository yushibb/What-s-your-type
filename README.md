# 人物评分深度学习系统

基于 FaceNet (InceptionResnetV1) + 概率分布回归头的人物评分系统，采用成对排序损失 + 回放缓存 + 软标签策略，支持增量训练和实时预测。

## 模型架构

```
人脸图片 → MTCNN 裁剪(160×160) → InceptionResnetV1 (frozen, vggface2) → 512维向量
         → MLP 回归头 (trainable) → 11类 Softmax → 概率分布期望 → 0.0-10.0 评分
```

| 组件 | 说明 |
|------|------|
| **MTCNN** | 人脸检测与裁剪，定位人脸区域并去除背景干扰 |
| **FaceNet (InceptionResnetV1)** | 预训练人脸特征提取器（vggface2 权重），完全冻结，输出 512 维嵌入 |
| **MLP 回归头** | `512 → 128 → 11 (Softmax)`，唯一可训练部分，输出 0-10 共 11 类概率分布 |
| **概率期望评分** | `Score = Σ(i × P(class_i))`，将分布期望作为最终评分 |

## 核心算法

### 1. 成对排序损失 (Pairwise Ranking Loss)

人类对绝对数字的感知很差，但对相对好坏的感知精准。训练时将数据转换为"图片对"：

- 图片 A 评分 8 分，图片 B 评分 6 分 → 隐含偏好 A ≻ B
- 引入 Margin Ranking Loss：`L_rank = max(0, margin - (f(A) - f(B)))`
- 迫使模型优先学习相对偏好顺序，而非死磕绝对数字
- 即使整体尺度漂移，只要相对顺序不变，模型依然稳固

### 2. 回放缓存 (Replay Buffer)

解决单样本在线更新的"学了新的，忘了旧的"问题：

- 维护容量 50 的动态回放缓存，持久化到 `data/replay_buffer.pt`
- 新样本录入时，从 Buffer 随机抽取 3-5 条历史样本，拼成 mini-batch 一起更新
- 老样本充当"锚点"，确保模型只微调偏好，不发生剧烈振荡

### 3. 概率分布输出 (11-class Softmax → 期望值)

用户的审美是模糊、主观的，标量回归无法表达这种不确定性：

- 回归头输出 11 个神经元（代表 0-10 分的概率），经 Softmax 得到概率分布
- 最终评分 = 概率分布期望值 `Σ(i × P(class_i))`
- 额外返回**置信度**（最高概率）和**犹豫度**（分布标准差）
- 天然对噪声具有强容忍度，模糊性转化为分布方差

### 4. 软标签训练

使用高斯软标签替代硬 one-hot 标签：

- 评分 7.5 → 以 7.5 为中心的高斯分布分散到 0-10 各类
- 缓解绝对回归的生硬和偏激，让模型输出更平滑

### 5. 极端感知加权

- 极端分数（< 3 或 > 7）样本权重 4x
- 极端样本过采样（被更多次抽取）
- 分布多样性正则（惩罚预测聚集在窄区间）

### 综合损失函数

```
L_total = L_mse + α·L_soft_ce + β·L_pairwise_ranking
```

| 损失 | 作用 |
|------|------|
| 极端感知加权 MSE | 约束绝对分值精度 |
| 软标签交叉熵 | 概率分布对齐，容忍主观模糊性 |
| 成对排序损失 | 保证相对偏好顺序正确 |

## 项目结构

```
├── app.py                 # Flask 主应用
├── model.py               # 深度学习模型定义（ScoreModel, ReplayBuffer）
├── train.py               # 训练模块（在线更新、全量微调、损失函数）
├── requirements.txt       # Python 依赖
├── deploy.sh              # 一键部署脚本
├── data/
│   ├── labels.csv         # 训练标签
│   ├── images/            # 训练图片
│   └── replay_buffer.pt   # 回放缓存持久化
├── models/
│   └── score_model.pth    # 模型权重
├── static/
│   ├── css/style.css      # 前端样式
│   └── js/main.js         # 前端逻辑
├── templates/
│   └── index.html         # 主页面模板
└── logs/                  # 日志目录
```

## 本地开发

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖（CPU 版）
pip install -r requirements.txt

# 或 GPU 版
pip install torch torchvision
pip install flask facenet-pytorch Pillow gunicorn

# 启动开发服务器
python app.py
```

访问 http://localhost:5000

## 服务器部署

```bash
# 运行部署脚本
bash deploy.sh
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload-training` | POST | 上传训练数据（图片+评分），自动在线更新模型 |
| `/api/training-samples` | GET | 获取已录入的训练样本列表 |
| `/api/delete-sample/<filename>` | DELETE | 删除指定训练样本 |
| `/api/predict` | POST | 上传图片预测评分（含概率分布、置信度、犹豫度） |
| `/api/predict-correct` | POST | 提交评分修正（保存为训练样本+回放缓存更新） |
| `/api/finetune` | POST | 启动全量微调（MSE + 软标签CE + 成对排序损失） |
| `/api/train-status` | GET | 查询训练状态 |
| `/api/model-info` | GET | 获取模型信息 |
| `/api/export-model` | GET | 导出模型权重 |
| `/api/import-model` | POST | 导入模型权重 |

## 功能说明

### 训练数据录入
- 上传人物图片并输入 0.0-10.0 评分
- 每次录入自动通过回放缓存在线更新模型（新样本 + 历史样本混合训练）
- 支持拖拽上传

### 分数预测
- 上传人物图片，MTCNN 自动裁剪人脸
- 输出概率分布期望评分 + 置信度 + 犹豫度
- 显示 0-10 分概率分布柱状图
- 支持评分修正：修正后自动保存为训练样本并更新模型

### 全量微调
- 用所有已有数据重新训练 MLP 回归头
- 损失函数：极端感知加权 MSE + 软标签交叉熵 + 成对排序损失
- 预计算嵌入向量缓存，训练秒级完成
- 建议数据量 ≥ 3 条时使用

### 模型导入导出
- 导出 `.pth` 模型权重文件
- 导入已有模型，自动兼容旧版（1输出→11输出自动迁移）
