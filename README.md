# 人物评分深度学习系统

基于 FaceNet (InceptionResnetV1) + MLP 回归头的人物评分深度学习系统，支持增量训练和实时预测。

## 模型架构

```
人脸图片 → MTCNN 裁剪 → InceptionResnetV1 (frozen, vggface2) → 512维向量
         → MLP 回归头 (trainable) → Sigmoid × 10 → 0.0-10.0 评分
```

- **Backbone**: FaceNet (InceptionResnetV1)，预训练权重 vggface2，完全冻结
- **回归头**: 512 → 128 → 1 (Sigmoid × 10)，可训练
- **在线更新**: 每录入一条数据，自动做 3 步 SGD 更新
- **全量微调**: 预计算嵌入向量缓存，MLP 回归头在缓存上训练

## 项目结构

```
├── app.py                 # Flask 主应用
├── model.py               # 深度学习模型定义
├── train.py               # 增量训练模块
├── requirements.txt       # Python 依赖
├── deploy.sh              # 一键部署脚本（远程服务器）
├── DEPLOY_NOTES.md        # 部署备忘录
├── data/
│   ├── labels.csv         # 训练标签
│   └── images/            # 训练图片
├── models/
│   └── score_model.pth    # 模型权重（训练后生成）
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

详见 [DEPLOY_NOTES.md](DEPLOY_NOTES.md)，核心步骤：

```bash
# 1. 将项目文件上传到服务器
scp -P 28702 -r . root@10.251.171.6:/root/

# 2. SSH 连接服务器
ssh -p 28702 root@10.251.171.6

# 3. 运行部署脚本
bash /root/deploy.sh
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload-training` | POST | 上传训练数据（图片+评分），自动在线更新模型 |
| `/api/training-samples` | GET | 获取已录入的训练样本列表 |
| `/api/delete-sample/<filename>` | DELETE | 删除指定训练样本 |
| `/api/predict` | POST | 上传图片预测评分 |
| `/api/predict-correct` | POST | 提交评分修正（保存为训练样本+更新模型） |
| `/api/finetune` | POST | 启动全量微调 |
| `/api/train-status` | GET | 查询训练状态 |
| `/api/model-info` | GET | 获取模型信息 |
| `/api/export-model` | GET | 导出模型权重 |
| `/api/import-model` | POST | 导入模型权重 |

## 功能说明

### 训练数据录入
- 上传人物图片并输入 0.0-10.0 评分
- 每次录入自动做 3 步 SGD 在线更新模型
- 支持拖拽上传

### 分数预测
- 上传人物图片，MTCNN 自动裁剪人脸
- 模型输出 0.0-10.0 评分
- 支持评分修正：修正后自动保存为训练样本并更新模型

### 全量微调
- 用所有已有数据重新训练 MLP 回归头
- 预计算嵌入向量缓存，训练秒级完成
- 建议数据量 ≥ 3 条时使用

### 模型导入导出
- 导出 `.pth` 模型权重文件
- 导入已有模型，可在其基础上继续训练
