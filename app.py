"""
Flask 主应用 v3：用户登录 + 数据隔离 + 成对排序 + 回放缓存 + 概率分布输出

新增：
  - 用户注册/登录/登出（Flask-Login + SQLite）
  - 每用户独立数据空间：data/users/<username>/images, labels.csv, replay_buffer.pt
  - 每用户独立模型：data/users/<username>/score_model.pth
  - 新用户初始化：空数据集 + 预训练权重模型
  - 登出仅清除会话，数据与模型持久保留
"""

import os
import gc
import csv
import uuid
import shutil
import tempfile
import threading
import hashlib

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import torch
from torchvision import transforms
from PIL import Image

from model import ScoreModel, detect_and_crop_face, UserPreferenceProfile
from train import online_update_single, finetune_model, rlhf_preference_update

# ── 应用配置 ──────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["USER_DATA_ROOT"] = os.path.join(os.path.dirname(__file__), "data", "users")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# ── 认证系统 ──────────────────────────────────────────────

login_manager = LoginManager(app)
login_manager.login_view = "login"
bcrypt = Bcrypt(app)

_train_lock = threading.Lock()
_training_status = {"is_training": False, "progress": "", "result": None}


class User(UserMixin):
    """Flask-Login 用户模型"""
    def __init__(self, username):
        self.id = username

    @staticmethod
    def _users_file():
        return os.path.join(os.path.dirname(__file__), "data", "users.json")

    @staticmethod
    def _load_users():
        path = User._users_file()
        if not os.path.exists(path):
            return {}
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_users(users):
        path = User._users_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get(user_id):
        users = User._load_users()
        if user_id in users:
            return User(user_id)
        return None

    @staticmethod
    def verify(username, password):
        users = User._load_users()
        if username not in users:
            return None
        if bcrypt.check_password_hash(users[username], password):
            return User(username)
        return None

    @staticmethod
    def create(username, password):
        users = User._load_users()
        if username in users:
            return False  # 已存在
        users[username] = bcrypt.generate_password_hash(password).decode("utf-8")
        User._save_users(users)
        # 初始化用户数据空间
        user_dir = get_user_data_dir(username)
        os.makedirs(os.path.join(user_dir, "images"), exist_ok=True)
        _init_labels_csv(username)
        # 初始化模型（复制预训练权重）
        _init_user_model(username)
        return True


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# ── 用户数据目录管理 ──────────────────────────────────────

def get_user_data_dir(username):
    """获取用户数据根目录"""
    return os.path.join(app.config["USER_DATA_ROOT"], username)


def get_user_paths(username):
    """获取用户所有数据路径"""
    d = get_user_data_dir(username)
    return {
        "root": d,
        "images": os.path.join(d, "images"),
        "labels_csv": os.path.join(d, "labels.csv"),
        "model": os.path.join(d, "score_model.pth"),
        "profile": os.path.join(d, "user_profile.json"),
        "replay_buffer": os.path.join(d, "replay_buffer.pt"),
    }


def _init_labels_csv(username):
    paths = get_user_paths(username)
    if not os.path.exists(paths["labels_csv"]):
        with open(paths["labels_csv"], "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "score"])


def _init_user_model(username):
    """为新用户初始化模型（使用预训练 FaceNet + 随机 MLP 回归头）"""
    paths = get_user_paths(username)
    if os.path.exists(paths["model"]):
        return
    device = torch.device("cpu")
    model = ScoreModel(pretrained=True)
    model.to(device)
    os.makedirs(os.path.dirname(paths["model"]), exist_ok=True)
    torch.save(model.state_dict(), paths["model"])
    del model
    gc.collect()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_user_profile(username):
    paths = get_user_paths(username)
    return UserPreferenceProfile.load(paths["profile"])


def _save_user_profile(username, profile):
    paths = get_user_paths(username)
    profile.save(paths["profile"])


# ── 认证页面路由 ──────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("index"))
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        return render_template("login.html", error="请输入用户名和密码")

    user = User.verify(username, password)
    if user is None:
        return render_template("login.html", error="用户名或密码错误")

    login_user(user, remember=True)
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("index"))
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")

    if not username or not password:
        return render_template("register.html", error="请输入用户名和密码")

    if len(username) < 2 or len(username) > 20:
        return render_template("register.html", error="用户名需 2-20 个字符")

    if not username.isalnum() and "_" not in username:
        return render_template("register.html", error="用户名只能包含字母、数字和下划线")

    if len(password) < 4:
        return render_template("register.html", error="密码至少 4 位")

    if password != confirm:
        return render_template("register.html", error="两次密码不一致")

    if not User.create(username, password):
        return render_template("register.html", error="用户名已被注册")

    user = User.verify(username, password)
    login_user(user, remember=True)
    return redirect(url_for("index"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── 主页面路由 ──────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html", username=current_user.id)


# ── 模块1：训练数据录入 API ──────────────────────────────

@app.route("/api/upload-training", methods=["POST"])
@login_required
def upload_training():
    username = current_user.id
    _init_labels_csv(username)
    paths = get_user_paths(username)

    if "image" not in request.files:
        return jsonify({"error": "请选择要上传的图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件格式"}), 400

    try:
        score = float(request.form.get("score", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "评分必须为 0.0-10.0 之间的数字"}), 400
    if not (0.0 <= score <= 10.0):
        return jsonify({"error": "评分必须在 0.0-10.0 之间"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(paths["images"], unique_name)
    os.makedirs(paths["images"], exist_ok=True)
    file.save(save_path)

    csv_path = paths["labels_csv"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([unique_name, score])

    with open(csv_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1

    update_result = None
    try:
        update_result = online_update_single(
            image_path=save_path,
            score=score,
            model_save_path=paths["model"],
            replay_buffer_path=paths["replay_buffer"],
            lr=1e-4,
            steps=3,
            use_extreme_loss=True,
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
@login_required
def get_training_samples():
    username = current_user.id
    _init_labels_csv(username)
    paths = get_user_paths(username)

    samples = []
    with open(paths["labels_csv"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return jsonify({"total": len(samples), "samples": samples})


@app.route("/api/delete-sample/<filename>", methods=["DELETE"])
@login_required
def delete_sample(filename):
    username = current_user.id
    _init_labels_csv(username)
    paths = get_user_paths(username)

    rows = []
    found = False
    with open(paths["labels_csv"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == filename:
                found = True
                img_path = os.path.join(paths["images"], filename)
                if os.path.exists(img_path):
                    os.remove(img_path)
            else:
                rows.append(row)

    if not found:
        return jsonify({"error": "未找到该样本"}), 404

    with open(paths["labels_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return jsonify({"message": "样本已删除"})


# ── 全量微调接口 ──────────────────────────────────────────

@app.route("/api/finetune", methods=["POST"])
@login_required
def start_finetune():
    global _training_status
    username = current_user.id

    if _training_status["is_training"]:
        return jsonify({"error": "模型正在训练中，请稍后再试"}), 409

    _init_labels_csv(username)
    paths = get_user_paths(username)

    with open(paths["labels_csv"], "r", encoding="utf-8") as f:
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
            _training_status["progress"] = "全量微调中（成对排序+软标签+极端感知）..."
            _training_status["result"] = None
            result = finetune_model(
                csv_path=paths["labels_csv"],
                image_dir=paths["images"],
                model_save_path=paths["model"],
                replay_buffer_path=paths["replay_buffer"],
                epochs=epochs,
                lr=lr,
                use_extreme_loss=True,
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
@login_required
def train_status():
    return jsonify(_training_status)


# ── 模块2：分数预测 API ──────────────────────────────────

MIN_TRAINING_SAMPLES_FOR_PREDICT = 10

_predict_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    username = current_user.id
    paths = get_user_paths(username)

    _init_labels_csv(username)
    with open(paths["labels_csv"], "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in f) - 1

    if total_samples < MIN_TRAINING_SAMPLES_FOR_PREDICT:
        return jsonify({
            "error": f"训练数据不足，当前仅有 {total_samples} 条，至少需要 {MIN_TRAINING_SAMPLES_FOR_PREDICT} 条才能使用预测功能",
            "total_samples": total_samples,
            "required": MIN_TRAINING_SAMPLES_FOR_PREDICT,
        }), 400

    if not os.path.exists(paths["model"]):
        return jsonify({"error": "模型尚未训练，请先录入训练数据"}), 400

    if "image" not in request.files:
        return jsonify({"error": "请选择要预测的图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件格式"}), 400

    model = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ScoreModel.load_model(paths["model"], device)

        image = Image.open(file.stream).convert("RGB")
        face_image = detect_and_crop_face(image, device)
        input_tensor = _predict_transform(face_image).unsqueeze(0).to(device)

        with torch.no_grad():
            score, probs = model.forward_with_distribution(input_tensor)

        score_val = round(max(0.0, min(10.0, score.item())), 2)
        dist = probs[0].cpu().tolist()

        peak_class = dist.index(max(dist))
        confidence = round(max(dist), 4)

        import math
        variance = sum((i - score_val) ** 2 * p for i, p in enumerate(dist))
        std_dev = round(math.sqrt(variance), 2)

        return jsonify({
            "score": score_val,
            "model": "FaceNet-Score-v2",
            "distribution": {str(i): round(p, 4) for i, p in enumerate(dist)},
            "confidence": confidence,
            "peak": peak_class,
            "std_dev": std_dev,
        })

    except Exception as e:
        return jsonify({"error": f"预测失败：{str(e)}"}), 500
    finally:
        del model
        _cleanup_torch()


@app.route("/api/predict-correct", methods=["POST"])
@login_required
def predict_correct():
    username = current_user.id
    _init_labels_csv(username)
    paths = get_user_paths(username)

    if "image" not in request.files:
        return jsonify({"error": "请提供图片"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件格式"}), 400

    try:
        score = float(request.form.get("score", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "修正评分必须为 0.0-10.0 之间的数字"}), 400
    if not (0.0 <= score <= 10.0):
        return jsonify({"error": "修正评分必须在 0.0-10.0 之间"}), 400

    raw_score = float(request.form.get("raw_score", "0"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(paths["images"], unique_name)
    os.makedirs(paths["images"], exist_ok=True)
    file.save(save_path)

    csv_path = paths["labels_csv"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([unique_name, score])

    with open(csv_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1

    profile = _get_user_profile(username)
    profile.add_correction(raw_score, score, filename=unique_name)
    _save_user_profile(username, profile)

    update_result = None
    try:
        update_result = online_update_single(
            image_path=save_path,
            score=score,
            model_save_path=paths["model"],
            replay_buffer_path=paths["replay_buffer"],
            lr=1e-4,
            steps=3,
            use_extreme_loss=True,
        )
    except Exception as e:
        update_result = {"error": str(e)}

    return jsonify({
        "message": "修正评分已保存，模型已通过回放缓存更新",
        "filename": unique_name,
        "score": score,
        "raw_score": raw_score,
        "total_samples": total,
        "model_updated": "error" not in (update_result or {}),
        "update_detail": update_result,
    })


# ── 用户偏好档案 API ──────────────────────────────────

@app.route("/api/user-profile", methods=["GET"])
@login_required
def get_user_profile():
    username = current_user.id
    profile = _get_user_profile(username)
    return jsonify({
        "corrections_count": len(profile.corrections),
        "preference_mean": round(profile.preference_mean, 2),
        "preference_std": round(profile.preference_std, 2),
    })


# ── RLHF 偏好对齐 API ──────────────────────────────────

@app.route("/api/rlhf-update", methods=["POST"])
@login_required
def rlhf_update():
    global _training_status
    username = current_user.id

    if _training_status["is_training"]:
        return jsonify({"error": "模型正在训练中，请稍后再试"}), 409

    paths = get_user_paths(username)
    profile = _get_user_profile(username)
    if len(profile.corrections) < 2:
        return jsonify({"error": "修正记录不足，至少需要 2 条"}), 400

    params = request.get_json() or {}

    def _job():
        global _training_status
        try:
            _training_status["is_training"] = True
            _training_status["progress"] = "RLHF 偏好对齐中..."
            _training_status["result"] = None
            current_profile = _get_user_profile(username)
            result = rlhf_preference_update(
                model_save_path=paths["model"],
                image_dir=paths["images"],
                user_profile=current_profile,
            )
            _training_status["result"] = result
            _training_status["progress"] = "RLHF 偏好对齐完成"
        except Exception as e:
            _training_status["result"] = {"error": str(e)}
            _training_status["progress"] = "RLHF 偏好对齐失败"
        finally:
            _training_status["is_training"] = False

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"message": "RLHF 偏好对齐已启动", "corrections_count": len(profile.corrections)})


# ── 模型信息 API ──────────────────────────────────────────

@app.route("/api/model-info", methods=["GET"])
@login_required
def model_info():
    username = current_user.id
    paths = get_user_paths(username)

    model_exists = os.path.exists(paths["model"])
    info = {"model_exists": model_exists}
    if model_exists:
        stat = os.stat(paths["model"])
        info["model_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
    _init_labels_csv(username)
    with open(paths["labels_csv"], "r", encoding="utf-8") as f:
        info["total_samples"] = sum(1 for _ in f) - 1
    info["architecture"] = "FaceNet-v2 (InceptionResnetV1 + 11-class Softmax)"

    info["gpu"] = {
        "available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    info["replay_buffer"] = os.path.exists(paths["replay_buffer"])
    info["predict_ready"] = info["total_samples"] >= MIN_TRAINING_SAMPLES_FOR_PREDICT
    info["predict_required"] = MIN_TRAINING_SAMPLES_FOR_PREDICT

    return jsonify(info)


# ── 模型导入导出 ──────────────────────────────────────────

@app.route("/api/export-model", methods=["GET"])
@login_required
def export_model():
    username = current_user.id
    paths = get_user_paths(username)

    model_path = paths["model"]
    if not os.path.exists(model_path):
        return jsonify({"error": "当前没有已训练的模型可导出"}), 400
    try:
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"score_model_{username}.pth",
            mimetype="application/octet-stream",
        )
    except Exception as e:
        return jsonify({"error": f"导出失败：{str(e)}"}), 500


@app.route("/api/import-model", methods=["POST"])
@login_required
def import_model():
    username = current_user.id
    paths = get_user_paths(username)

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
        test_model.load_state_dict(state_dict, strict=False)
        del test_model
        _cleanup_torch()
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": f"模型文件无效：{str(e)}"}), 400

    model_dir = os.path.dirname(paths["model"])
    os.makedirs(model_dir, exist_ok=True)
    shutil.move(tmp_path, paths["model"])

    return jsonify({"message": "模型导入成功，可在其基础上继续训练"})


# ── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(app.config["USER_DATA_ROOT"], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
