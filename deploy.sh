#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 人物评分深度学习系统 — 一键部署脚本
# 在服务器上执行: bash deploy.sh
# ═══════════════════════════════════════════════════════════════
set -e

# ── 配置区（根据实际情况修改）──
APP_DIR="/opt/score-app"
APP_USER="root"                    # 当前以 root 运行
APP_PORT=5000                      # 外部访问端口
USE_GPU=true                      # 有 NVIDIA GPU 改为 true

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   人物评分深度学习系统 — 一键部署           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# 第 1 步：系统更新 & 安装基础依赖
# ═══════════════════════════════════════════════════════════════
echo "===== [1/8] 系统更新 & 安装基础依赖 ====="
apt update -y && apt upgrade -y
apt install -y python3 python3-pip python3-venv git nginx curl

# ═══════════════════════════════════════════════════════════════
# 第 2 步：创建项目目录
# ═══════════════════════════════════════════════════════════════
echo "===== [2/8] 创建项目目录 ====="
mkdir -p $APP_DIR/{data/images,models,static/{js,css},templates,logs}

# ═══════════════════════════════════════════════════════════════
# 第 3 步：写入项目源码
# ═══════════════════════════════════════════════════════════════
echo "===== [3/8] 写入项目源码 ====="

# ── model.py ──
cat > $APP_DIR/model.py << 'PYEOF'
"""
深度学习模型定义：基于 FaceNet (InceptionResnetV1) 的评分回归模型
架构：
  [MTCNN 人脸裁剪] -> [InceptionResnetV1 (frozen, vggface2)] -> 512维向量
  -> [MLP 回归头 (trainable)] -> Sigmoid * 10 -> 输出 0.0-10.0 评分
"""

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN


# ── MTCNN 人脸检测器（全局单例）──────────────────────────────

_mtcnn_instance = None


def get_mtcnn(device: torch.device) -> MTCNN:
    """获取 MTCNN 单例（用于人脸检测与裁剪）"""
    global _mtcnn_instance
    if _mtcnn_instance is None:
        _mtcnn_instance = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
        )
    return _mtcnn_instance


def detect_and_crop_face(image_pil, device: torch.device):
    """使用 MTCNN 检测并裁剪人脸区域

    Args:
        image_pil: PIL.Image 输入图片
        device: 计算设备

    Returns:
        裁剪后的人脸 PIL.Image，如果检测不到人脸则返回原图（resize 到 160x160）
    """
    mtcnn = get_mtcnn(device)
    # MTCNN 返回裁剪后的人脸张量或 None
    face_tensor = mtcnn(image_pil)

    if face_tensor is not None:
        # 将张量转回 PIL Image
        # face_tensor shape: [3, 160, 160]，值在 [-1, 1] 范围
        face_tensor = (face_tensor + 1.0) / 2.0  # 归一化到 [0, 1]
        face_tensor = face_tensor.clamp(0, 1)
        from torchvision.transforms.functional import to_pil_image
        return to_pil_image(face_tensor)
    else:
        # 未检测到人脸，返回 resize 后的原图
        return image_pil.resize((160, 160))


# ── 评分模型 ──────────────────────────────────────────────

class ScoreModel(nn.Module):
    """人物评分深度学习模型

    使用预训练 InceptionResnetV1 (vggface2) 提取 512 维人脸特征，
    冻结 Backbone，仅训练 MLP 回归头输出 0.0-10.0 的评分。
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # FaceNet Backbone：完全冻结，只做特征提取
        self.backbone = InceptionResnetV1(
            classify=False,
            pretrained="vggface2" if pretrained else None,
        )
        # 冻结 Backbone 所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # MLP 回归头：512 -> 128 -> 1
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 输出 [0, 1]，后续乘以 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 0.0-10.0 的评分"""
        with torch.no_grad():
            feat = self.backbone(x)  # [B, 512]
        score = self.regressor(feat)  # [B, 1]
        return score * 10.0

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """提取 512 维人脸嵌入向量（不经过回归头）"""
        with torch.no_grad():
            return self.backbone(x)

    def extract_embedding_from_image(
        self,
        image_path: str,
        device: torch.device,
        use_mtcnn: bool = True,
    ) -> torch.Tensor:
        """从图片路径直接提取 512 维嵌入向量

        Args:
            image_path: 图片文件路径
            device: 计算设备
            use_mtcnn: 是否使用 MTCNN 裁剪人脸

        Returns:
            512 维嵌入向量 (CPU tensor)
        """
        from torchvision import transforms
        from PIL import Image

        _transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        image = Image.open(image_path).convert("RGB")
        if use_mtcnn:
            image = detect_and_crop_face(image, device)
        input_tensor = _transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = self.backbone(input_tensor)  # [1, 512]

        return embedding.squeeze(0).cpu()  # [512]

    @staticmethod
    def load_model(model_path: str, device: torch.device) -> "ScoreModel":
        """加载已保存的模型权重（仅加载回归头，Backbone 保持 vggface2 预训练）"""
        model = ScoreModel(pretrained=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model
PYEOF

# ── train.py ──
cat > $APP_DIR/train.py << 'PYEOF'
"""
增量训练模块：支持单样本在线更新与全量微调
核心策略：
  - 预计算 512 维嵌入向量缓存，MLP 回归头在缓存上快速训练
  - 每次新增一条数据时，立即用该单样本做一步 SGD 更新模型（在线学习）
  - 也可选择全量数据微调（批量训练）
  - 始终在已有模型基础上更新，不从头训练
"""

import os
import gc
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import ScoreModel, detect_and_crop_face


# ── 全局嵌入缓存 ──────────────────────────────────────────────

_embedding_cache: dict[str, torch.Tensor] = {}


def get_cached_embedding(
    image_path: str,
    model: ScoreModel,
    device: torch.device,
    use_mtcnn: bool = True,
) -> torch.Tensor:
    """获取图片的 512 维嵌入向量，优先从缓存读取

    Args:
        image_path: 图片文件路径
        model: ScoreModel 实例（用于未缓存时提取嵌入）
        device: 计算设备
        use_mtcnn: 是否使用 MTCNN 裁剪人脸

    Returns:
        512 维嵌入向量 (CPU tensor)
    """
    if image_path in _embedding_cache:
        return _embedding_cache[image_path]

    embedding = model.extract_embedding_from_image(image_path, device, use_mtcnn)
    _embedding_cache[image_path] = embedding
    return embedding


def clear_embedding_cache():
    """清空嵌入缓存"""
    _embedding_cache.clear()


# ── 嵌入向量数据集 ──────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """基于预计算 512 维嵌入向量的轻量数据集"""

    def __init__(self, embeddings: list[torch.Tensor], scores: list[float]):
        self.embeddings = embeddings
        self.scores = scores

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.scores[idx] / 10.0, dtype=torch.float32)


# ── 单样本在线更新 ──────────────────────────────────────────────

def online_update_single(
    image_path: str,
    score: float,
    model_save_path: str,
    lr: float = 1e-4,
    steps: int = 3,
    use_mtcnn: bool = True,
) -> dict:
    """单样本在线更新：用一条新数据即时更新模型

    使用缓存的嵌入向量加速，MTCNN+FaceNet 只对新图跑一次。

    Args:
        image_path: 图片文件路径
        score: 评分 (0.0-10.0)
        model_save_path: 模型保存路径
        lr: 学习率
        steps: 对该样本重复训练的步数（强化记忆）
        use_mtcnn: 是否使用 MTCNN 裁剪人脸

    Returns:
        更新结果字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载或初始化模型
    model = _load_or_init_model(model_save_path, device)
    model.train()
    model.backbone.eval()

    # 获取缓存的嵌入向量（MTCNN+FaceNet 只跑一次）
    embedding = get_cached_embedding(image_path, model, device, use_mtcnn)
    embedding = embedding.to(device).unsqueeze(0)  # [1, 512]
    target = torch.tensor([[score / 10.0]], dtype=torch.float32, device=device)

    # 优化器只训练回归头参数
    optimizer = optim.Adam(model.regressor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        output = model.regressor(embedding)  # 直接走回归头
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 6))

    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # 清理内存
    del model, optimizer, criterion, embedding, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "online_update",
        "steps": steps,
        "losses": losses,
        "final_loss": losses[-1],
    }


# ── 全量微调 ──────────────────────────────────────────────────

def finetune_model(
    csv_path: str,
    image_dir: str,
    model_save_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-5,
    use_mtcnn: bool = True,
) -> dict:
    """全量数据微调：预计算嵌入后只训练 MLP 回归头

    优化流程：
      1. 一次性预计算所有图片的 512 维嵌入向量（MTCNN+FaceNet 只跑一遍）
      2. 在缓存的嵌入向量上训练 MLP 回归头（极快，秒级完成）

    Args:
        csv_path: 标签 CSV 文件路径
        image_dir: 图片目录
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率（增量微调建议较小值）
        use_mtcnn: 是否使用 MTCNN 裁剪人脸

    Returns:
        训练结果字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 第 1 步：读取 CSV，收集有效样本 ──
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(image_dir, row["filename"])
            if os.path.exists(img_path):
                samples.append((img_path, float(row["score"])))

    if len(samples) == 0:
        return {"error": "没有可用的训练数据"}

    # ── 第 2 步：预计算所有嵌入向量（MTCNN+FaceNet 只跑一次）──
    model = _load_or_init_model(model_save_path, device)
    model.eval()

    embeddings = []
    scores = []
    for img_path, score in samples:
        emb = get_cached_embedding(img_path, model, device, use_mtcnn)
        embeddings.append(emb)
        scores.append(score)

    # ── 第 3 步：在缓存嵌入上训练 MLP 回归头 ──
    model.train()
    model.backbone.eval()

    dataset = EmbeddingDataset(embeddings, scores)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.regressor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for emb_batch, score_batch in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model.regressor(emb_batch)  # 直接走回归头
            loss = criterion(outputs, score_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)
        loss_history.append(round(avg_loss, 6))

    # 保存
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # 清理
    del model, optimizer, criterion, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "finetune",
        "epochs": epochs,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "dataset_size": len(samples),
    }


# ── 辅助函数 ──────────────────────────────────────────────

def _load_or_init_model(model_save_path: str, device: torch.device) -> ScoreModel:
    """加载已有模型或初始化新模型"""
    if os.path.exists(model_save_path):
        return ScoreModel.load_model(model_save_path, device)
    else:
        model = ScoreModel(pretrained=True)
        model.to(device)
        return model
PYEOF

# ── app.py ──
cat > $APP_DIR/app.py << 'PYEOF'
"""
Flask 主应用：提供训练数据录入和分数预测两个独立入口
核心改进：
  - 每次新增数据自动在线更新模型（单样本 SGD）
  - 预测修正功能始终可用
  - MTCNN 人脸裁剪
  - 预测图片不保存到磁盘（除非用户修正）
"""

import os
import gc
import csv
import uuid
import shutil
import tempfile
import threading

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image

from model import ScoreModel, detect_and_crop_face, get_mtcnn
from train import online_update_single, finetune_model

# ── 应用配置 ──────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "data", "images")
app.config["LABELS_CSV"] = os.path.join(os.path.dirname(__file__), "data", "labels.csv")
app.config["MODEL_PATH"] = os.path.join(os.path.dirname(__file__), "models", "score_model.pth")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# 训练状态
_train_lock = threading.Lock()
_training_status = {"is_training": False, "progress": "", "result": None}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _init_labels_csv():
    csv_path = app.config["LABELS_CSV"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "score"])


def _cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 页面路由 ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── 模块1：训练数据录入 API ──────────────────────────────

@app.route("/api/upload-training", methods=["POST"])
def upload_training():
    """上传带评分的人物图片 → 保存 → 自动在线更新模型"""
    _init_labels_csv()

    if "image" not in request.files:
        return jsonify({"error": "请选择要上传的图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件格式"}), 400

    try:
        score = float(request.form.get("score", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "评分必须为 0.0-10.0 之间的数字"}), 400
    if not (0.0 <= score <= 10.0):
        return jsonify({"error": "评分必须在 0.0-10.0 之间"}), 400

    # 保存图片
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    # 写入 CSV
    csv_path = app.config["LABELS_CSV"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([unique_name, score])

    with open(csv_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1

    # 自动在线更新模型（后台执行，不阻塞响应）
    update_result = None
    try:
        update_result = online_update_single(
            image_path=save_path,
            score=score,
            model_save_path=app.config["MODEL_PATH"],
            lr=1e-4,
            steps=3,
        )
    except Exception as e:
        update_result = {"error": str(e)}

    return jsonify({
        "message": "训练数据录入成功，模型已自动更新",
        "filename": unique_name,
        "score": score,
        "total_samples": total,
        "model_updated": "error" not in (update_result or {}),
        "update_detail": update_result,
    })


@app.route("/api/training-samples", methods=["GET"])
def get_training_samples():
    _init_labels_csv()
    csv_path = app.config["LABELS_CSV"]
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return jsonify({"total": len(samples), "samples": samples})


@app.route("/api/delete-sample/<filename>", methods=["DELETE"])
def delete_sample(filename):
    csv_path = app.config["LABELS_CSV"]
    _init_labels_csv()

    rows = []
    found = False
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == filename:
                found = True
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                if os.path.exists(img_path):
                    os.remove(img_path)
            else:
                rows.append(row)

    if not found:
        return jsonify({"error": "未找到该样本"}), 404

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return jsonify({"message": "样本已删除"})


# ── 全量微调接口（可选，用于批量重训）──────────────────────

@app.route("/api/finetune", methods=["POST"])
def start_finetune():
    """全量微调：用所有已有数据做几轮训练"""
    global _training_status

    if _training_status["is_training"]:
        return jsonify({"error": "模型正在训练中，请稍后再试"}), 409

    _init_labels_csv()
    csv_path = app.config["LABELS_CSV"]
    with open(csv_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1

    if total < 3:
        return jsonify({"error": f"训练数据不足，当前仅有 {total} 条，至少需要 3 条"}), 400

    params = request.get_json() or {}
    epochs = params.get("epochs", 20)
    lr = params.get("lr", 5e-5)

    def _job():
        global _training_status
        try:
            _training_status["is_training"] = True
            _training_status["progress"] = "全量微调中..."
            _training_status["result"] = None
            result = finetune_model(
                csv_path=csv_path,
                image_dir=app.config["UPLOAD_FOLDER"],
                model_save_path=app.config["MODEL_PATH"],
                epochs=epochs,
                lr=lr,
            )
            _training_status["result"] = result
            _training_status["progress"] = "微调完成"
        except Exception as e:
            _training_status["result"] = {"error": str(e)}
            _training_status["progress"] = "微调失败"
        finally:
            _training_status["is_training"] = False

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"message": "全量微调已启动", "total_samples": total})


@app.route("/api/train-status", methods=["GET"])
def train_status():
    return jsonify(_training_status)


# ── 模块2：分数预测 API ──────────────────────────────────

_predict_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@app.route("/api/predict", methods=["POST"])
def predict():
    """上传图片 → MTCNN 裁剪人脸 → 预测评分（不保存图片到磁盘）"""
    if not os.path.exists(app.config["MODEL_PATH"]):
        return jsonify({"error": "模型尚未训练，请先录入训练数据"}), 400

    if "image" not in request.files:
        return jsonify({"error": "请选择要预测的图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件格式"}), 400

    model = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ScoreModel.load_model(app.config["MODEL_PATH"], device)

        # 从内存读取图片
        image = Image.open(file.stream).convert("RGB")

        # MTCNN 人脸裁剪
        face_image = detect_and_crop_face(image, device)

        input_tensor = _predict_transform(face_image).unsqueeze(0).to(device)

        with torch.no_grad():
            score = model(input_tensor).item()

        score = round(max(0.0, min(10.0, score)), 2)

        return jsonify({"score": score, "model": "FaceNet-Score"})

    except Exception as e:
        return jsonify({"error": f"预测失败：{str(e)}"}), 500
    finally:
        del model
        _cleanup_torch()


@app.route("/api/predict-correct", methods=["POST"])
def predict_correct():
    """预测结果修正：保存图片+修正评分 → 自动在线更新模型"""
    _init_labels_csv()

    if "image" not in request.files:
        return jsonify({"error": "请提供图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件格式"}), 400

    try:
        score = float(request.form.get("score", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "修正评分必须为 0.0-10.0 之间的数字"}), 400
    if not (0.0 <= score <= 10.0):
        return jsonify({"error": "修正评分必须在 0.0-10.0 之间"}), 400

    # 保存图片为训练样本
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    # 写入 CSV
    csv_path = app.config["LABELS_CSV"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([unique_name, score])

    with open(csv_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1

    # 自动在线更新模型
    update_result = None
    try:
        update_result = online_update_single(
            image_path=save_path,
            score=score,
            model_save_path=app.config["MODEL_PATH"],
            lr=1e-4,
            steps=3,
        )
    except Exception as e:
        update_result = {"error": str(e)}

    return jsonify({
        "message": "修正评分已保存，模型已自动更新",
        "filename": unique_name,
        "score": score,
        "total_samples": total,
        "model_updated": "error" not in (update_result or {}),
        "update_detail": update_result,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    model_exists = os.path.exists(app.config["MODEL_PATH"])
    info = {"model_exists": model_exists}
    if model_exists:
        stat = os.stat(app.config["MODEL_PATH"])
        info["model_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
    _init_labels_csv()
    with open(app.config["LABELS_CSV"], "r", encoding="utf-8") as f:
        info["total_samples"] = sum(1 for _ in f) - 1
    info["architecture"] = "FaceNet (InceptionResnetV1 + MLP)"
    return jsonify(info)


# ── 模型导入导出 ──────────────────────────────────────────

@app.route("/api/export-model", methods=["GET"])
def export_model():
    model_path = app.config["MODEL_PATH"]
    if not os.path.exists(model_path):
        return jsonify({"error": "当前没有已训练的模型可导出"}), 400
    try:
        return send_file(
            model_path,
            as_attachment=True,
            download_name="score_model_facenet.pth",
            mimetype="application/octet-stream",
        )
    except Exception as e:
        return jsonify({"error": f"导出失败：{str(e)}"}), 500


@app.route("/api/import-model", methods=["POST"])
def import_model():
    if "model" not in request.files:
        return jsonify({"error": "请选择要导入的模型文件"}), 400

    file = request.files["model"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        device = torch.device("cpu")
        test_model = ScoreModel(pretrained=True)
        state_dict = torch.load(tmp_path, map_location=device, weights_only=True)
        test_model.load_state_dict(state_dict, strict=True)
        del test_model
        _cleanup_torch()
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": f"模型文件无效：{str(e)}"}), 400

    model_dir = os.path.dirname(app.config["MODEL_PATH"])
    os.makedirs(model_dir, exist_ok=True)
    shutil.move(tmp_path, app.config["MODEL_PATH"])

    return jsonify({"message": "模型导入成功，可在其基础上继续训练"})


# ── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    _init_labels_csv()
    app.run(debug=True, host="0.0.0.0", port=5000)
PYEOF

# ── requirements.txt ──
cat > $APP_DIR/requirements.txt << 'PYEOF'
flask==3.1.0
facenet-pytorch==2.5.3
torch==2.11.0
torchvision==0.26.0
Pillow==12.1.1
gunicorn==23.0.0
PYEOF

echo "  Python 源码写入完成"

# ═══════════════════════════════════════════════════════════════
# 第 3.5 步：写入前端文件
# ═══════════════════════════════════════════════════════════════

# ── static/css/style.css ──
cat > $APP_DIR/static/css/style.css << 'CSSEOF'
/* ── 全局变量与重置 ──────────────────────────────────── */
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-dark: #4f46e5;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --bg-input: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
}

/* ── 头部 ─────────────────────────────────────────── */
.app-header {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.logo { display: flex; align-items: center; gap: 12px; }
.logo svg { width: 32px; height: 32px; color: var(--primary); }
.logo h1 {
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, var(--primary-light), var(--primary));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.header-actions { display: flex; align-items: center; gap: 16px; }
.model-actions { display: flex; gap: 8px; }

.action-btn {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 6px 12px; background: var(--bg-card-hover);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--text-muted); font-size: 13px; cursor: pointer;
    transition: all 0.2s; font-family: inherit;
}
.action-btn svg { width: 14px; height: 14px; }
.action-btn:hover:not(:disabled) { border-color: var(--primary); color: var(--primary-light); }
.action-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.model-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 20px;
    background: rgba(239, 68, 68, 0.15); color: var(--danger);
    font-size: 13px; font-weight: 500;
}
.model-badge.ready { background: rgba(34, 197, 94, 0.15); color: var(--success); }
.model-badge .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: currentColor; animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── 模型架构说明横幅 ─────────────────────────────── */
.arch-banner {
    max-width: 1200px; margin: 24px auto 0; padding: 0 32px;
}
.arch-flow {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 24px; flex-wrap: wrap;
}
.arch-step {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
    padding: 8px 12px; border-radius: var(--radius-sm);
    background: rgba(255,255,255,0.03); min-width: 80px; text-align: center;
}
.arch-step .arch-icon { font-size: 22px; }
.arch-step span { font-size: 12px; color: var(--text-muted); line-height: 1.4; }
.arch-step small { font-size: 10px; color: var(--text-dim); }
.arch-step.highlight { border: 1px solid var(--primary); background: rgba(99,102,241,0.1); }
.arch-step.highlight span { color: var(--primary-light); }
.arch-step.result { border: 1px solid var(--warning); background: rgba(245,158,11,0.1); }
.arch-step.result span { color: var(--warning); }
.arch-arrow { color: var(--text-dim); font-size: 18px; font-weight: 700; }

/* ── 模块选择 ─────────────────────────────────────── */
.module-selector {
    max-width: 1200px; margin: 24px auto 0; padding: 0 32px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
}
.module-tab {
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    padding: 24px; background: var(--bg-card); border: 2px solid var(--border);
    border-radius: var(--radius); cursor: pointer; transition: all 0.3s;
    color: var(--text-muted); font-family: inherit; font-size: 16px;
}
.module-tab svg { width: 28px; height: 28px; }
.module-tab small { font-size: 12px; color: var(--text-dim); }
.module-tab:hover { background: var(--bg-card-hover); border-color: var(--primary); color: var(--text); }
.module-tab.active {
    background: rgba(99, 102, 241, 0.1); border-color: var(--primary);
    color: var(--primary-light);
}
.module-tab.active small { color: var(--primary-light); opacity: 0.7; }

/* ── 面板切换 ─────────────────────────────────────── */
.module-panel { max-width: 1200px; margin: 24px auto; padding: 0 32px; display: none; }
.module-panel.active { display: block; }

/* ── 卡片 ─────────────────────────────────────────── */
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 28px; box-shadow: var(--shadow);
}
.card h2 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
.card-desc { color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }
.card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.card-header h2 { margin-bottom: 0; }
.badge {
    background: rgba(99, 102, 241, 0.15); color: var(--primary-light);
    padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 500;
}

/* ── 训练面板布局 ─────────────────────────────────── */
.panel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

/* ── 拖拽上传区 ──────────────────────────────────── */
.drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 32px; text-align: center; cursor: pointer; transition: all 0.3s;
    min-height: 180px; display: flex; align-items: center; justify-content: center;
}
.drop-zone:hover, .drop-zone.dragover {
    border-color: var(--primary); background: rgba(99, 102, 241, 0.05);
}
.drop-zone-content {
    display: flex; flex-direction: column; align-items: center; gap: 8px; color: var(--text-muted);
}
.drop-zone-content svg { width: 48px; height: 48px; opacity: 0.5; }
.drop-zone-content span { font-size: 14px; }
.drop-zone-content small { font-size: 12px; color: var(--text-dim); }
.drop-zone-content img { max-width: 100%; max-height: 200px; border-radius: var(--radius-sm); object-fit: contain; }

/* ── 表单 ─────────────────────────────────────────── */
.form-group { margin-bottom: 20px; }
.form-group label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 500; color: var(--text-muted); }
.range-hint { color: var(--text-dim); font-weight: 400; }
.score-input-wrapper { display: flex; align-items: center; gap: 12px; }
.score-input-wrapper input[type="range"] {
    flex: 1; height: 6px; -webkit-appearance: none; appearance: none;
    background: var(--border); border-radius: 3px; outline: none;
}
.score-input-wrapper input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%;
    background: var(--primary); cursor: pointer; box-shadow: 0 2px 6px rgba(99, 102, 241, 0.4);
}
.score-input-wrapper input[type="number"] {
    width: 80px; padding: 8px 12px; background: var(--bg-input);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--text); font-size: 16px; font-weight: 600; text-align: center; outline: none;
}
.score-input-wrapper input[type="number"]:focus { border-color: var(--primary); }
.score-display { margin-top: 8px; display: flex; align-items: center; gap: 12px; }
.score-value { font-size: 28px; font-weight: 700; color: var(--primary-light); min-width: 50px; }
.score-bar { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.score-bar-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--primary-light)); border-radius: 4px; transition: width 0.15s; }

/* ── 按钮 ─────────────────────────────────────────── */
.btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    padding: 12px 24px; border: none; border-radius: var(--radius-sm);
    font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.2s;
    font-family: inherit; width: 100%;
}
.btn svg { width: 18px; height: 18px; }
.btn-primary { background: linear-gradient(135deg, var(--primary), var(--primary-dark)); color: white; }
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4); }
.btn-train { background: linear-gradient(135deg, var(--success), #16a34a); color: white; }
.btn-train:hover { box-shadow: 0 4px 16px rgba(34, 197, 94, 0.4); }
.btn-predict { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; margin-top: 20px; }
.btn-predict:hover { box-shadow: 0 4px 16px rgba(245, 158, 11, 0.4); }
.btn-secondary { background: var(--bg-card-hover); color: var(--text); border: 1px solid var(--border); width: auto; }
.btn-secondary:hover { border-color: var(--primary); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; box-shadow: none !important; }

/* ── 自动更新状态 ─────────────────────────────────── */
.auto-update-status {
    display: flex; align-items: center; gap: 10px;
    margin-top: 16px; padding: 12px 16px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: var(--radius-sm);
    color: var(--primary-light); font-size: 13px;
}

/* ── 样本列表 ─────────────────────────────────────── */
.sample-list { max-height: 260px; overflow-y: auto; margin-bottom: 20px; }
.sample-list::-webkit-scrollbar { width: 4px; }
.sample-list::-webkit-scrollbar-track { background: transparent; }
.sample-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.sample-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 12px; border-radius: var(--radius-sm); transition: background 0.2s;
}
.sample-item:hover { background: rgba(255, 255, 255, 0.03); }
.sample-info { display: flex; align-items: center; gap: 10px; }
.sample-name { font-size: 13px; color: var(--text-muted); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sample-score { font-weight: 600; color: var(--primary-light); font-size: 15px; }
.sample-delete {
    background: none; border: none; color: var(--text-dim); cursor: pointer;
    padding: 4px; border-radius: 4px; transition: all 0.2s; display: flex; align-items: center;
}
.sample-delete:hover { color: var(--danger); background: rgba(239, 68, 68, 0.1); }
.empty-state { text-align: center; padding: 32px; color: var(--text-dim); }
.empty-state svg { width: 48px; height: 48px; margin-bottom: 12px; opacity: 0.3; }
.empty-state p { font-size: 14px; margin-bottom: 4px; }
.empty-state small { font-size: 12px; }

/* ── 训练区域 ─────────────────────────────────────── */
.train-section { border-top: 1px solid var(--border); padding-top: 20px; }
.train-section h3 { font-size: 15px; margin-bottom: 8px; color: var(--text-muted); }
.train-desc { font-size: 12px; color: var(--text-dim); margin-bottom: 12px; line-height: 1.6; }
.train-params { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.train-params label { font-size: 13px; color: var(--text-dim); display: flex; flex-direction: column; gap: 4px; }
.train-params input {
    padding: 8px 12px; background: var(--bg-input); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text); font-size: 14px; outline: none;
}
.train-params input:focus { border-color: var(--primary); }

/* ── 参数推荐说明 ─────────────────────────────────── */
.param-tips {
    background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: var(--radius-sm); padding: 12px 14px; margin-bottom: 16px;
    font-size: 12px; line-height: 1.7;
}
.tip-item { color: var(--text-muted); margin-bottom: 4px; }
.tip-item:last-child { margin-bottom: 0; }
.tip-item strong { color: var(--primary-light); font-weight: 600; }

.train-progress {
    display: flex; align-items: center; gap: 10px; margin-top: 12px;
    color: var(--primary-light); font-size: 14px;
}
.spinner {
    width: 18px; height: 18px; border: 2px solid var(--border);
    border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.train-result {
    margin-top: 12px; padding: 12px;
    background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: var(--radius-sm); font-size: 13px; color: var(--success);
}

/* ── 预测面板 ─────────────────────────────────────── */
.predict-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
.predict-drop { min-height: 300px; }

/* ── 预测结果 ─────────────────────────────────────── */
.result-score-wrapper { text-align: center; padding: 32px 0; }
.result-score {
    font-size: 72px; font-weight: 800;
    background: linear-gradient(135deg, var(--primary-light), var(--warning));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin-bottom: 20px;
}
.result-bar-bg { height: 12px; background: var(--border); border-radius: 6px; overflow: hidden; margin: 0 20px; }
.result-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
    border-radius: 6px; transition: width 1s ease-out;
}
.result-label { text-align: center; font-size: 18px; font-weight: 600; margin-top: 16px; color: var(--text); }
.result-meta { margin-top: 16px; font-size: 13px; color: var(--text-dim); text-align: center; }

/* ── 评分修正区域 ─────────────────────────────────── */
.correct-section { margin-top: 20px; border-top: 1px solid var(--border); padding-top: 16px; }
.correct-divider { text-align: center; margin-bottom: 12px; }
.correct-divider span {
    font-size: 13px; color: var(--warning); font-weight: 500;
    background: var(--bg-card); padding: 0 12px; position: relative;
}
.correct-label { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; display: block; }
.correct-input-row { display: flex; gap: 10px; margin-bottom: 8px; }
.correct-input {
    flex: 1; padding: 8px 14px; background: var(--bg-input);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--text); font-size: 16px; font-weight: 600; outline: none;
}
.correct-input:focus { border-color: var(--primary); }
.btn-correct {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white; width: auto; white-space: nowrap; padding: 8px 20px; font-size: 14px;
}
.btn-correct:hover { box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4); }
.correct-hint { font-size: 11px; color: var(--text-dim); margin-bottom: 12px; }
.btn-discard {
    background: transparent; color: var(--text-dim); border: 1px dashed var(--border);
    font-size: 13px; padding: 8px 16px; width: auto; margin: 0 auto; display: block;
}
.btn-discard:hover { color: var(--text-muted); border-color: var(--text-dim); }

/* ── 无模型提示 ──────────────────────────────────── */
.no-model-card { text-align: center; padding: 48px 28px; }
.no-model-icon { margin-bottom: 16px; }
.no-model-icon svg { width: 56px; height: 56px; color: var(--warning); }
.no-model-card h3 { font-size: 18px; margin-bottom: 8px; color: var(--warning); }
.no-model-card p {
    color: var(--text-muted); font-size: 14px; margin-bottom: 20px;
    max-width: 320px; margin-left: auto; margin-right: auto;
}

/* ── Toast 通知 ──────────────────────────────────── */
.toast-container {
    position: fixed; top: 20px; right: 20px; z-index: 1000;
    display: flex; flex-direction: column; gap: 10px;
}
.toast {
    padding: 14px 20px; border-radius: var(--radius-sm);
    font-size: 14px; font-weight: 500; box-shadow: var(--shadow);
    animation: slideIn 0.3s ease-out; max-width: 400px;
}
.toast.success { background: rgba(34, 197, 94, 0.9); color: white; }
.toast.error { background: rgba(239, 68, 68, 0.9); color: white; }
.toast.info { background: rgba(99, 102, 241, 0.9); color: white; }
.toast.warning { background: rgba(245, 158, 11, 0.9); color: white; }

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* ── 响应式 ───────────────────────────────────────── */
@media (max-width: 768px) {
    .panel-grid, .predict-layout, .module-selector { grid-template-columns: 1fr; }
    .app-header { padding: 12px 16px; }
    .module-panel, .module-selector { padding: 0 16px; }
    .result-score { font-size: 56px; }
    .header-actions { gap: 8px; }
    .model-actions { gap: 4px; }
    .action-btn span { display: none; }
    .arch-flow { gap: 6px; padding: 12px 16px; }
    .arch-step { padding: 6px 8px; min-width: 60px; }
    .arch-arrow { font-size: 14px; }
    .arch-banner { padding: 0 16px; }
}
CSSEOF

echo "  CSS 写入完成"

# ── static/js/main.js ──
cat > $APP_DIR/static/js/main.js << 'JSEOF'
/* ── 全局工具函数 ──────────────────────────────────── */

function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = "slideIn 0.3s ease-out reverse";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function switchModule(module) {
    document.querySelectorAll(".module-tab").forEach(tab => {
        tab.classList.toggle("active", tab.dataset.module === module);
    });
    document.getElementById("trainingPanel").classList.toggle("active", module === "training");
    document.getElementById("predictPanel").classList.toggle("active", module === "predict");

    if (module === "predict") {
        checkModelForPredict();
    }
    if (module === "training") {
        loadSamples();
    }
}

function previewImage(input, previewId) {
    const preview = document.getElementById(previewId);
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="预览">`;
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function syncScore(source) {
    const slider = document.getElementById("scoreSlider");
    const number = document.getElementById("scoreNumber");
    const display = document.getElementById("scoreDisplay");
    const barFill = document.getElementById("scoreBarFill");

    if (source === "slider") {
        number.value = slider.value;
    } else {
        let val = parseFloat(number.value) || 0;
        val = Math.max(0, Math.min(10, val));
        number.value = val;
        slider.value = val;
    }

    const val = parseFloat(slider.value);
    display.textContent = val.toFixed(1);
    barFill.style.width = (val / 10 * 100) + "%";
}

// ── 拖拽上传 ────────────────────────────────────────

function setupDropZone(zoneId, fileInputId) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;

    ["dragenter", "dragover"].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            zone.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            zone.classList.remove("dragover");
        });
    });
    zone.addEventListener("drop", e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const input = document.getElementById(fileInputId);
            input.files = files;
            input.dispatchEvent(new Event("change"));
        }
    });
}

// ── 训练数据录入 ────────────────────────────────────

async function submitTraining(event) {
    event.preventDefault();

    const fileInput = document.getElementById("trainingImage");
    const score = document.getElementById("scoreNumber").value;

    if (!fileInput.files || !fileInput.files[0]) {
        showToast("请先选择一张图片", "error");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("score", score);

    const btn = document.getElementById("uploadBtn");
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner"></div> 录入并更新中...`;

    // 显示自动更新状态
    const autoStatus = document.getElementById("autoUpdateStatus");
    const autoText = document.getElementById("autoUpdateText");
    autoStatus.style.display = "flex";
    autoText.textContent = "录入数据并更新模型中...";

    try {
        const res = await fetch("/api/upload-training", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "上传失败", "error");
            return;
        }

        // 显示模型更新结果
        if (data.model_updated) {
            const detail = data.update_detail;
            autoText.textContent = `模型已更新 (loss: ${detail.final_loss.toFixed(6)})`;
            showToast(`录入成功！评分 ${data.score}，模型已自动更新`, "success");
        } else {
            autoText.textContent = "模型更新失败，数据已保存";
            showToast(`录入成功！评分 ${data.score}，但模型更新失败`, "warning");
        }

        // 重置表单
        fileInput.value = "";
        document.getElementById("trainingPreview").innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <path d="M21 15l-5-5L5 21"/>
            </svg>
            <span>点击或拖拽上传图片</span>
            <small>支持 PNG / JPG / BMP / WebP</small>
        `;

        loadSamples();
        updateModelBadge();

        // 3秒后隐藏自动更新状态
        setTimeout(() => { autoStatus.style.display = "none"; }, 3000);
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
        autoStatus.style.display = "none";
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            录入数据并更新模型
        `;
    }
}

async function loadSamples() {
    try {
        const res = await fetch("/api/training-samples");
        const data = await res.json();

        const list = document.getElementById("sampleList");
        const count = document.getElementById("sampleCount");
        count.textContent = `${data.total} 条`;

        if (data.total === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                        <circle cx="12" cy="7" r="4"/>
                    </svg>
                    <p>暂无训练数据</p>
                    <small>请先上传带评分的图片</small>
                </div>
            `;
            return;
        }

        list.innerHTML = data.samples.map((s) => `
            <div class="sample-item">
                <div class="sample-info">
                    <span class="sample-name" title="${s.filename}">${s.filename}</span>
                    <span class="sample-score">${parseFloat(s.score).toFixed(1)}</span>
                </div>
                <button class="sample-delete" onclick="deleteSample('${s.filename}')" title="删除">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        `).join("");
    } catch (err) {
        console.error("加载样本失败:", err);
    }
}

async function deleteSample(filename) {
    if (!confirm(`确定删除 ${filename}？`)) return;
    try {
        const res = await fetch(`/api/delete-sample/${filename}`, { method: "DELETE" });
        const data = await res.json();
        if (res.ok) {
            showToast("样本已删除", "success");
            loadSamples();
        } else {
            showToast(data.error || "删除失败", "error");
        }
    } catch (err) {
        showToast("删除失败", "error");
    }
}

// ── 全量微调 ────────────────────────────────────────

async function startFinetune() {
    const btn = document.getElementById("trainBtn");
    const progress = document.getElementById("trainProgress");
    const result = document.getElementById("trainResult");

    btn.disabled = true;
    progress.style.display = "flex";
    result.style.display = "none";

    const epochs = parseInt(document.getElementById("trainEpochs").value) || 20;
    const lr = parseFloat(document.getElementById("trainLr").value) || 5e-5;

    try {
        const res = await fetch("/api/finetune", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs, lr }),
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "启动微调失败", "error");
            btn.disabled = false;
            progress.style.display = "none";
            return;
        }

        showToast("全量微调已启动", "info");
        pollTrainStatus();
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
        btn.disabled = false;
        progress.style.display = "none";
    }
}

function pollTrainStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch("/api/train-status");
            const data = await res.json();

            document.getElementById("trainProgressText").textContent = data.progress;

            if (!data.is_training) {
                clearInterval(interval);
                document.getElementById("trainBtn").disabled = false;
                document.getElementById("trainProgress").style.display = "none";

                if (data.result && !data.result.error) {
                    const r = data.result;
                    document.getElementById("trainResult").style.display = "block";
                    document.getElementById("trainResult").innerHTML = `
                        全量微调完成！数据量: ${r.dataset_size} | 轮数: ${r.epochs} |
                        最终 Loss: ${r.final_loss.toFixed(6)}
                    `;
                    showToast("全量微调完成！", "success");
                    updateModelBadge();
                } else if (data.result && data.result.error) {
                    showToast("微调失败：" + data.result.error, "error");
                }
            }
        } catch (err) {
            clearInterval(interval);
        }
    }, 2000);
}

// ── 分数预测 ────────────────────────────────────────

let _predictFileRef = null;

async function checkModelForPredict() {
    try {
        const res = await fetch("/api/model-info");
        const data = await res.json();

        const noModel = document.getElementById("noModelCard");
        const resultCard = document.getElementById("resultCard");

        if (!data.model_exists) {
            noModel.style.display = "block";
            resultCard.style.display = "none";
        } else {
            noModel.style.display = "none";
        }
    } catch (err) {
        console.error("检查模型状态失败:", err);
    }
}

async function predictScore() {
    const fileInput = document.getElementById("predictImage");

    if (!fileInput.files || !fileInput.files[0]) {
        showToast("请先选择一张图片", "error");
        return;
    }

    // 保留文件引用供后续修正使用
    _predictFileRef = fileInput.files[0];

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner"></div> 预测中...`;

    try {
        const res = await fetch("/api/predict", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "预测失败", "error");
            if (data.error && data.error.includes("尚未训练")) {
                document.getElementById("noModelCard").style.display = "block";
            }
            return;
        }

        // 显示结果
        const resultCard = document.getElementById("resultCard");
        resultCard.style.display = "block";
        document.getElementById("noModelCard").style.display = "none";

        const score = data.score;
        document.getElementById("resultScore").textContent = score.toFixed(1);

        // 修正输入默认值 = AI 评分
        document.getElementById("correctScoreInput").value = score.toFixed(1);

        // 确保修正区域可见（不隐藏）
        document.getElementById("correctSection").style.display = "";

        // 动画填充评分条
        setTimeout(() => {
            document.getElementById("resultBarFill").style.width = (score / 10 * 100) + "%";
        }, 100);

        // 评分等级
        let label = "";
        if (score >= 9.0) label = "出类拔萃";
        else if (score >= 7.5) label = "非常优秀";
        else if (score >= 6.0) label = "中上水平";
        else if (score >= 4.0) label = "中等水平";
        else if (score >= 2.0) label = "有待提升";
        else label = "需要关注";

        document.getElementById("resultLabel").textContent = label;
        document.getElementById("resultMeta").textContent = `模型: ${data.model} | MTCNN人脸裁剪 | ${new Date().toLocaleTimeString()}`;

        showToast(`预测评分：${score.toFixed(1)} 分`, "success");
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"/>
                <path d="M21 21l-4.35-4.35"/>
            </svg>
            预测评分
        `;
    }
}

/**
 * 提交评分修正 → 保存为训练样本 → 自动更新模型
 * 修正区域始终保留，不清除，下次预测时自动刷新默认值
 */
async function submitCorrection() {
    if (!_predictFileRef) {
        showToast("没有可修正的预测图片，请先预测一张图片", "error");
        return;
    }

    const scoreInput = document.getElementById("correctScoreInput");
    const score = parseFloat(scoreInput.value);

    if (isNaN(score) || score < 0 || score > 10) {
        showToast("请输入 0.0-10.0 之间的修正评分", "error");
        return;
    }

    const formData = new FormData();
    formData.append("image", _predictFileRef);
    formData.append("score", score.toString());

    try {
        const res = await fetch("/api/predict-correct", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "提交修正失败", "error");
            return;
        }

        const updateMsg = data.model_updated ? "，模型已自动更新" : "，但模型更新失败";
        showToast(`修正评分 ${score.toFixed(1)} 已保存为训练样本${updateMsg}`, "success");

        // 清除文件引用，但不清除修正区域 UI
        _predictFileRef = null;

        updateModelBadge();
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
    }
}

/**
 * 丢弃预测图片（不保存为训练样本）
 * 只清除文件引用，修正区域 UI 保持不变
 */
function discardCorrection() {
    _predictFileRef = null;
    showToast("图片已丢弃，不会保存为训练样本", "info");
}

// ── 模型状态徽章 ────────────────────────────────────

async function updateModelBadge() {
    try {
        const res = await fetch("/api/model-info");
        const data = await res.json();
        const badge = document.getElementById("modelBadge");
        const text = document.getElementById("modelStatusText");
        const exportBtn = document.getElementById("exportBtn");

        if (data.model_exists) {
            badge.classList.add("ready");
            text.textContent = `模型已就绪 (${data.model_size_mb}MB | ${data.total_samples}条数据)`;
            if (exportBtn) exportBtn.disabled = false;
        } else {
            badge.classList.remove("ready");
            text.textContent = "模型未训练";
            if (exportBtn) exportBtn.disabled = true;
        }
    } catch (err) {
        console.error("更新模型状态失败:", err);
    }
}

// ── 模型导入导出 ────────────────────────────────────

function exportModel() {
    const link = document.createElement("a");
    link.href = "/api/export-model";
    link.download = "score_model_facenet.pth";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("模型导出中...", "info");
}

async function importModel(input) {
    if (!input.files || !input.files[0]) return;

    const file = input.files[0];
    if (!file.name.endsWith(".pth")) {
        showToast("请选择 .pth 格式的模型文件", "error");
        input.value = "";
        return;
    }

    if (!confirm(`确定导入模型「${file.name}」？将替换当前模型，可在此基础上继续训练。`)) {
        input.value = "";
        return;
    }

    const formData = new FormData();
    formData.append("model", file);

    try {
        const res = await fetch("/api/import-model", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "导入失败", "error");
            return;
        }

        showToast("模型导入成功，可在其基础上继续训练", "success");
        updateModelBadge();
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
    } finally {
        input.value = "";
    }
}

// ── 页面初始化 ──────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    setupDropZone("trainingDropZone", "trainingImage");
    setupDropZone("predictDropZone", "predictImage");
    loadSamples();
    updateModelBadge();
});
JSEOF

echo "  JS 写入完成"

# ── templates/index.html ──
cat > $APP_DIR/templates/index.html << 'HTMLEOF'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>人物评分深度学习系统</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <header class="app-header">
        <div class="header-content">
            <div class="logo">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8Z"/>
                    <path d="M12 6v6l4 2"/>
                </svg>
                <h1>人物评分深度学习系统</h1>
            </div>
            <div class="header-actions">
                <div class="model-actions">
                    <button class="action-btn" onclick="exportModel()" title="导出模型" id="exportBtn" disabled>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        <span>导出</span>
                    </button>
                    <button class="action-btn" onclick="document.getElementById('importModelFile').click()" title="导入模型">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                        <span>导入</span>
                    </button>
                    <input type="file" id="importModelFile" accept=".pth" hidden onchange="importModel(this)">
                </div>
                <div class="model-badge" id="modelBadge">
                    <span class="dot"></span>
                    <span id="modelStatusText">模型未加载</span>
                </div>
            </div>
        </div>
    </header>

    <main class="app-main">
        <!-- 模型架构说明 -->
        <div class="arch-banner">
            <div class="arch-flow">
                <div class="arch-step">
                    <div class="arch-icon">🖼️</div>
                    <span>人脸图片</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <div class="arch-icon">✂️</div>
                    <span>MTCNN 裁剪</span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">
                    <div class="arch-icon">🧠</div>
                    <span>FaceNet (frozen)<br><small>512维向量</small></span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step highlight">
                    <div class="arch-icon">📐</div>
                    <span>MLP 回归头<br><small>可训练</small></span>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-step result">
                    <div class="arch-icon">⭐</div>
                    <span>0.0-10.0 评分</span>
                </div>
            </div>
        </div>

        <!-- 模块选择卡片 -->
        <div class="module-selector">
            <button class="module-tab active" data-module="training" onclick="switchModule('training')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 5v14M5 12h14"/>
                </svg>
                <span>训练数据录入</span>
                <small>上传图片 + 评分，自动更新模型</small>
            </button>
            <button class="module-tab" data-module="predict" onclick="switchModule('predict')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
                    <rect x="9" y="1" width="6" height="4" rx="1"/>
                    <path d="M9 14l2 2 4-4"/>
                </svg>
                <span>分数预测</span>
                <small>上传图片，AI 自动评分</small>
            </button>
        </div>

        <!-- 模块1：训练数据录入 -->
        <section class="module-panel active" id="trainingPanel">
            <div class="panel-grid">
                <!-- 左侧：上传表单 -->
                <div class="card upload-card">
                    <h2>录入训练数据</h2>
                    <p class="card-desc">上传人物图片并输入对应评分，系统将自动更新模型</p>

                    <form id="trainingForm" onsubmit="submitTraining(event)">
                        <div class="form-group">
                            <label>选择人物图片</label>
                            <div class="drop-zone" id="trainingDropZone" onclick="document.getElementById('trainingImage').click()">
                                <div class="drop-zone-content" id="trainingPreview">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                                        <circle cx="8.5" cy="8.5" r="1.5"/>
                                        <path d="M21 15l-5-5L5 21"/>
                                    </svg>
                                    <span>点击或拖拽上传图片</span>
                                    <small>支持 PNG / JPG / BMP / WebP</small>
                                </div>
                            </div>
                            <input type="file" id="trainingImage" accept="image/*" hidden onchange="previewImage(this, 'trainingPreview')">
                        </div>

                        <div class="form-group">
                            <label for="scoreInput">评分 <span class="range-hint">(0.0 - 10.0)</span></label>
                            <div class="score-input-wrapper">
                                <input type="range" id="scoreSlider" min="0" max="10" step="0.1" value="5.0" oninput="syncScore('slider')">
                                <input type="number" id="scoreNumber" min="0" max="10" step="0.1" value="5.0" oninput="syncScore('number')">
                            </div>
                            <div class="score-display">
                                <span class="score-value" id="scoreDisplay">5.0</span>
                                <div class="score-bar">
                                    <div class="score-bar-fill" id="scoreBarFill" style="width: 50%"></div>
                                </div>
                            </div>
                        </div>

                        <button type="submit" class="btn btn-primary" id="uploadBtn">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                <polyline points="17 8 12 3 7 8"/>
                                <line x1="12" y1="3" x2="12" y2="15"/>
                            </svg>
                            录入数据并更新模型
                        </button>
                    </form>

                    <!-- 自动更新状态提示 -->
                    <div class="auto-update-status" id="autoUpdateStatus" style="display:none">
                        <div class="spinner"></div>
                        <span id="autoUpdateText">模型更新中...</span>
                    </div>
                </div>

                <!-- 右侧：数据列表 + 全量微调 -->
                <div class="card data-card">
                    <div class="card-header">
                        <h2>已录入数据</h2>
                        <span class="badge" id="sampleCount">0 条</span>
                    </div>

                    <div class="sample-list" id="sampleList">
                        <div class="empty-state">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                                <circle cx="12" cy="7" r="4"/>
                            </svg>
                            <p>暂无训练数据</p>
                            <small>请先上传带评分的图片</small>
                        </div>
                    </div>

                    <div class="train-section">
                        <h3>全量微调（可选）</h3>
                        <p class="train-desc">每次录入数据时模型会自动在线更新。如需用全部数据重新微调，可在此操作：</p>
                        <div class="train-params">
                            <label>
                                训练轮数
                                <input type="number" id="trainEpochs" value="20" min="5" max="200">
                            </label>
                            <label>
                                学习率
                                <input type="number" id="trainLr" value="0.00005" step="0.00001" min="0.00001" max="0.01">
                            </label>
                        </div>
                        <div class="param-tips">
                            <div class="tip-item">
                                <strong>训练轮数推荐：</strong>
                                <span>少量数据(&lt;20条)建议 10-20 轮；中等数据(20-100条)建议 20-50 轮；大量数据(100+条)建议 30-100 轮</span>
                            </div>
                            <div class="tip-item">
                                <strong>学习率推荐：</strong>
                                <span>全量微调建议 5e-5（0.00005）以保护已有知识；首次训练可尝试 1e-4；微调阶段 1e-5~5e-5</span>
                            </div>
                            <div class="tip-item">
                                <strong>自动更新机制：</strong>
                                <span>每录入一条新数据，模型会立即做 3 步 SGD 在线更新，无需手动触发。全量微调适合数据量较大时统一优化。</span>
                            </div>
                        </div>
                        <button class="btn btn-train" id="trainBtn" onclick="startFinetune()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="5 3 19 12 5 21 5 3"/>
                            </svg>
                            全量微调
                        </button>
                        <div class="train-progress" id="trainProgress" style="display:none">
                            <div class="spinner"></div>
                            <span id="trainProgressText">全量微调中...</span>
                        </div>
                        <div class="train-result" id="trainResult" style="display:none"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 模块2：分数预测 -->
        <section class="module-panel" id="predictPanel">
            <div class="predict-layout">
                <div class="card predict-card">
                    <h2>上传图片预测评分</h2>
                    <p class="card-desc">上传一张人物图片，AI 模型将自动给出评分（MTCNN 自动裁剪人脸）</p>

                    <div class="drop-zone predict-drop" id="predictDropZone" onclick="document.getElementById('predictImage').click()">
                        <div class="drop-zone-content" id="predictPreview">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                <rect x="3" y="3" width="18" height="18" rx="2"/>
                                <circle cx="8.5" cy="8.5" r="1.5"/>
                                <path d="M21 15l-5-5L5 21"/>
                            </svg>
                            <span>点击或拖拽上传图片</span>
                            <small>支持 PNG / JPG / BMP / WebP</small>
                        </div>
                    </div>
                    <input type="file" id="predictImage" accept="image/*" hidden onchange="previewImage(this, 'predictPreview')">

                    <button class="btn btn-predict" onclick="predictScore()" id="predictBtn">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                        预测评分
                    </button>
                </div>

                <div class="card result-card" id="resultCard" style="display:none">
                    <h2>预测结果</h2>
                    <div class="result-score-wrapper">
                        <div class="result-score" id="resultScore">0.0</div>
                        <div class="result-bar-bg">
                            <div class="result-bar-fill" id="resultBarFill"></div>
                        </div>
                    </div>
                    <p class="result-label" id="resultLabel">--</p>
                    <div class="result-meta" id="resultMeta"></div>

                    <!-- 评分修正区域：始终可用，不会消失 -->
                    <div class="correct-section" id="correctSection">
                        <div class="correct-divider">
                            <span>认为 AI 评分有误？</span>
                        </div>
                        <div class="correct-form">
                            <label class="correct-label">输入正确评分：</label>
                            <div class="correct-input-row">
                                <input type="number" id="correctScoreInput" min="0" max="10" step="0.1" placeholder="0.0-10.0" class="correct-input">
                                <button class="btn btn-correct" onclick="submitCorrection()">提交修正</button>
                            </div>
                            <p class="correct-hint">提交后将作为训练样本保存，模型会立即自动更新</p>
                        </div>
                        <button class="btn btn-discard" onclick="discardCorrection()">不需要修正，丢弃图片</button>
                    </div>
                </div>

                <!-- 无模型提示 -->
                <div class="card no-model-card" id="noModelCard" style="display:none">
                    <div class="no-model-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="8" x2="12" y2="12"/>
                            <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                    </div>
                    <h3>模型尚未训练</h3>
                    <p>请先在「训练数据录入」模块中录入训练数据，录入第一条后模型即自动创建。</p>
                    <button class="btn btn-secondary" onclick="switchModule('training')">前往录入训练数据</button>
                </div>
            </div>
        </section>
    </main>

    <!-- Toast 通知 -->
    <div class="toast-container" id="toastContainer"></div>

    <script src="/static/js/main.js"></script>
</body>
</html>
HTMLEOF

echo "  HTML 写入完成"

# ── data/labels.csv ──
if [ ! -f $APP_DIR/data/labels.csv ]; then
    echo "filename,score" > $APP_DIR/data/labels.csv
fi

echo "  所有项目文件写入完成"

# ═══════════════════════════════════════════════════════════════
# 第 4 步：Python 虚拟环境 & 安装依赖
# ═══════════════════════════════════════════════════════════════
echo "===== [4/8] 安装 Python 依赖 ====="

cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

if [ "$USE_GPU" = true ]; then
    echo "  安装 GPU 版 PyTorch..."
    pip install torch torchvision
else
    echo "  安装 CPU 版 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install flask facenet-pytorch Pillow gunicorn

echo "  Python 依赖安装完成"

# ═══════════════════════════════════════════════════════════════
# 第 5 步：Gunicorn 配置
# ═══════════════════════════════════════════════════════════════
echo "===== [5/8] 配置 Gunicorn ====="

cat > $APP_DIR/gunicorn.conf.py << 'PYEOF'
bind = "127.0.0.1:8000"
workers = 2
threads = 2
timeout = 120
pidfile = '/tmp/score-app-gunicorn.pid'
accesslog = '/opt/score-app/logs/access.log'
errorlog = '/opt/score-app/logs/error.log'
loglevel = 'info'
PYEOF

echo "  Gunicorn 配置完成"

# ═══════════════════════════════════════════════════════════════
# 第 6 步：Systemd 服务
# ═══════════════════════════════════════════════════════════════
echo "===== [6/8] 配置 Systemd 自启服务 ====="

cat > /etc/systemd/system/score-app.service << PYEOF
[Unit]
Description=Score App - FaceNet Scoring System
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn -c gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=30
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
PYEOF

systemctl daemon-reload
systemctl enable score-app

echo "  Systemd 服务配置完成"

# ═══════════════════════════════════════════════════════════════
# 第 7 步：Nginx 反向代理
# ═══════════════════════════════════════════════════════════════
echo "===== [7/8] 配置 Nginx 反向代理 ====="

cat > /etc/nginx/sites-available/score-app << 'NGINXEOF'
server {
    listen 5000;
    server_name _;

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 120s;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    location /static/ {
        alias /opt/score-app/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}
NGINXEOF

ln -sf /etc/nginx/sites-available/score-app /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t

echo "  Nginx 配置完成"

# ═══════════════════════════════════════════════════════════════
# 第 8 步：启动所有服务
# ═══════════════════════════════════════════════════════════════
echo "===== [8/8] 启动服务 ====="

systemctl restart nginx
systemctl restart score-app

sleep 3

# ── 验证 ──
echo ""
echo "===== 验证部署 ====="

# 检查 Gunicorn
if systemctl is-active --quiet score-app; then
    echo "  ✅ score-app 服务运行中"
else
    echo "  ❌ score-app 服务未启动，查看日志: journalctl -u score-app -n 20"
fi

# 检查 Nginx
if systemctl is-active --quiet nginx; then
    echo "  ✅ Nginx 运行中"
else
    echo "  ❌ Nginx 未启动，查看日志: journalctl -u nginx -n 20"
fi

# 检查 Python 依赖
source $APP_DIR/venv/bin/activate
python -c "from model import ScoreModel; print('  ✅ 模型加载OK')" 2>&1 || echo "  ❌ 模型加载失败"
python -c "import torch; print(f'  ✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>&1 || echo "  ❌ PyTorch 加载失败"

# HTTP 测试
sleep 2
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/ 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  ✅ HTTP 访问正常 (200)"
else
    echo "  ❌ HTTP 访问异常 (HTTP $HTTP_CODE)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  部署完成！                                              ║"
echo "║                                                          ║"
echo "║  本地访问:  http://127.0.0.1:5000                        ║"
echo "║  外网访问:  http://<服务器公网IP>:5000                     ║"
echo "║                                                          ║"
echo "║  管理命令:                                               ║"
echo "║    查看状态:  systemctl status score-app                   ║"
echo "║    查看日志:  tail -f /opt/score-app/logs/error.log       ║"
echo "║    重启服务:  systemctl restart score-app                 ║"
echo "║    停止服务:  systemctl stop score-app                    ║"
echo "║                                                          ║"
echo "║  注意事项:                                               ║"
echo "║    1. 若外网无法访问，请检查防火墙/安全组是否放行 5000 端口  ║"
echo "║    2. 生产环境建议配置 HTTPS (参考 DEPLOY.md)              ║"
echo "║    3. 首次加载 FaceNet 模型需下载约 100MB 预训练权重        ║"
echo "╚════════════════════════════════════════════════════════════╝"
