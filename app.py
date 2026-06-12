"""
Flask 主应用 v2：成对排序 + 回放缓存 + 概率分布输出

核心改进：
  - 概率分布输出：11类 Softmax → 期望值评分，容忍主观噪声
  - 回放缓存：单样本更新时混合历史样本，防抖动
  - 成对排序损失：优先学习相对偏好顺序
  - 高斯软标签：主观模糊性 → 分布方差
"""

import os
import gc
import csv
import uuid
import shutil
import tempfile
import threading

from flask import Flask, render_template, request, jsonify, send_file
import torch
from torchvision import transforms
from PIL import Image

from model import ScoreModel, detect_and_crop_face, UserPreferenceProfile
from train import online_update_single, finetune_model, rlhf_preference_update

# ── 应用配置 ──────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "data", "images")
app.config["LABELS_CSV"] = os.path.join(os.path.dirname(__file__), "data", "labels.csv")
app.config["MODEL_PATH"] = os.path.join(os.path.dirname(__file__), "models", "score_model.pth")
app.config["USER_PROFILE_PATH"] = os.path.join(os.path.dirname(__file__), "data", "user_profile.json")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

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


def _get_user_profile() -> UserPreferenceProfile:
    return UserPreferenceProfile.load(app.config["USER_PROFILE_PATH"])


def _save_user_profile(profile: UserPreferenceProfile):
    profile.save(app.config["USER_PROFILE_PATH"])


# ── 页面路由 ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── 模块1：训练数据录入 API ──────────────────────────────

@app.route("/api/upload-training", methods=["POST"])
def upload_training():
    _init_labels_csv()

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
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    csv_path = app.config["LABELS_CSV"]
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
            model_save_path=app.config["MODEL_PATH"],
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


# ── 全量微调接口 ──────────────────────────────────────────

@app.route("/api/finetune", methods=["POST"])
def start_finetune():
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
            _training_status["progress"] = "全量微调中（成对排序+软标签+极端感知）..."
            _training_status["result"] = None
            result = finetune_model(
                csv_path=csv_path,
                image_dir=app.config["UPLOAD_FOLDER"],
                model_save_path=app.config["MODEL_PATH"],
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
    """上传图片 → MTCNN 裁剪 → 概率分布预测 → 期望值评分"""
    if not os.path.exists(app.config["MODEL_PATH"]):
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
        model = ScoreModel.load_model(app.config["MODEL_PATH"], device)

        image = Image.open(file.stream).convert("RGB")
        face_image = detect_and_crop_face(image, device)
        input_tensor = _predict_transform(face_image).unsqueeze(0).to(device)

        with torch.no_grad():
            score, probs = model.forward_with_distribution(input_tensor)

        score_val = round(max(0.0, min(10.0, score.item())), 2)
        dist = probs[0].cpu().tolist()

        # 模型置信度（最高概率）和峰值类别
        peak_class = dist.index(max(dist))
        confidence = round(max(dist), 4)

        # 分布方差（衡量模型对评分的模糊程度）
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
def predict_correct():
    """预测修正 → 保存样本 → 回放缓存更新模型"""
    _init_labels_csv()

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
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    csv_path = app.config["LABELS_CSV"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([unique_name, score])

    with open(csv_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1

    # 记录修正历史
    profile = _get_user_profile()
    profile.add_correction(raw_score, score, filename=unique_name)
    _save_user_profile(profile)

    # 回放缓存 + 在线更新
    update_result = None
    try:
        update_result = online_update_single(
            image_path=save_path,
            score=score,
            model_save_path=app.config["MODEL_PATH"],
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
def get_user_profile():
    profile = _get_user_profile()
    return jsonify({
        "corrections_count": len(profile.corrections),
        "preference_mean": round(profile.preference_mean, 2),
        "preference_std": round(profile.preference_std, 2),
    })


# ── RLHF 偏好对齐 API ──────────────────────────────────

@app.route("/api/rlhf-update", methods=["POST"])
def rlhf_update():
    global _training_status

    if _training_status["is_training"]:
        return jsonify({"error": "模型正在训练中，请稍后再试"}), 409

    profile = _get_user_profile()
    if len(profile.corrections) < 2:
        return jsonify({"error": f"修正记录不足，至少需要 2 条"}), 400

    params = request.get_json() or {}

    def _job():
        global _training_status
        try:
            _training_status["is_training"] = True
            _training_status["progress"] = "RLHF 偏好对齐中..."
            _training_status["result"] = None
            current_profile = _get_user_profile()
            result = rlhf_preference_update(
                model_save_path=app.config["MODEL_PATH"],
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
def model_info():
    model_exists = os.path.exists(app.config["MODEL_PATH"])
    info = {"model_exists": model_exists}
    if model_exists:
        stat = os.stat(app.config["MODEL_PATH"])
        info["model_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
    _init_labels_csv()
    with open(app.config["LABELS_CSV"], "r", encoding="utf-8") as f:
        info["total_samples"] = sum(1 for _ in f) - 1
    info["architecture"] = "FaceNet-v2 (InceptionResnetV1 + 11-class Softmax)"

    info["gpu"] = {
        "available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # 回放缓存信息
    replay_path = os.path.join(os.path.dirname(__file__), "data", "replay_buffer.pt")
    info["replay_buffer"] = os.path.exists(replay_path)

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
            download_name="score_model_facenet_v2.pth",
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
        # 兼容新旧模型
        test_model.load_state_dict(state_dict, strict=False)
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
