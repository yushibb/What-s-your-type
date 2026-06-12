"""
深度学习模型定义 v2：基于 FaceNet 的评分模型（概率分布输出）

架构：
  [MTCNN 人脸裁剪] -> [InceptionResnetV1 (frozen, vggface2)] -> 512维向量
  -> [MLP 回归头 (trainable)] -> 11类 logits -> Softmax -> 概率分布
  -> 期望值 = Σ(i × P(class_i)) -> 输出 0.0-10.0 评分

改进（v2）：
  - 11类概率分布输出，替代单标量 Sigmoid，天然容忍主观噪声
  - 期望值作为最终评分，更平滑、更稳健
  - 可获取模型置信度信息（分布方差 = 模型对主观模糊性的感知）
  - 兼容旧版 (Sigmoid×10) 模型权重自动迁移
"""

import json
import os

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN

NUM_CLASSES = 11  # 0, 1, 2, ..., 10


# ── MTCNN 人脸检测器（全局单例）──────────────────────────────

_mtcnn_instance = None


def get_mtcnn(device: torch.device) -> MTCNN:
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
    """使用 MTCNN 检测并裁剪人脸区域"""
    mtcnn = get_mtcnn(device)
    face_tensor = mtcnn(image_pil)

    if face_tensor is not None:
        face_tensor = (face_tensor + 1.0) / 2.0
        face_tensor = face_tensor.clamp(0, 1)
        from torchvision.transforms.functional import to_pil_image
        return to_pil_image(face_tensor)
    else:
        return image_pil.resize((160, 160))


# ── 评分模型 v2 ──────────────────────────────────────────────

class ScoreModel(nn.Module):
    """人物评分深度学习模型 (v2 - 概率分布输出)

    MLP 回归头输出 11 类 logits，经 Softmax 转为概率分布，
    通过计算期望值得到 0.0-10.0 的评分。

    优势：
    - 概率分布天然对主观噪声有容忍度（用户模糊性 → 分布方差）
    - 期望值更平滑，缓解绝对回归的生硬和偏激
    - 可获取模型置信度信息
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = InceptionResnetV1(
            classify=False,
            pretrained="vggface2" if pretrained else None,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),  # 11 classes: 0-10
        )

        # 类别值 [0,1,...,10] 作为 buffer，随模型移动到设备
        self.register_buffer('class_values',
                             torch.arange(NUM_CLASSES, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 0.0-10.0 的评分（期望值）"""
        with torch.no_grad():
            feat = self.backbone(x)
        logits = self.regressor(feat)
        probs = torch.softmax(logits, dim=-1)
        score = (probs * self.class_values).sum(dim=-1, keepdim=True)
        return score

    def forward_with_distribution(self, x: torch.Tensor):
        """前向传播，返回 (评分, 概率分布)"""
        with torch.no_grad():
            feat = self.backbone(x)
        logits = self.regressor(feat)
        probs = torch.softmax(logits, dim=-1)
        score = (probs * self.class_values).sum(dim=-1, keepdim=True)
        return score, probs

    def logits_to_score(self, logits: torch.Tensor) -> torch.Tensor:
        """将 logits 转为期望评分 (0-10)"""
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.class_values).sum(dim=-1, keepdim=True)

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """将 logits 转为概率分布"""
        return torch.softmax(logits, dim=-1)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """提取 512 维人脸嵌入向量"""
        with torch.no_grad():
            return self.backbone(x)

    def extract_embedding_from_image(
        self,
        image_path: str,
        device: torch.device,
        use_mtcnn: bool = True,
    ) -> torch.Tensor:
        """从图片路径直接提取 512 维嵌入向量"""
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
            embedding = self.backbone(input_tensor)

        return embedding.squeeze(0).cpu()

    @staticmethod
    def load_model(model_path: str, device: torch.device) -> "ScoreModel":
        """加载模型权重，自动兼容旧版 (Sigmoid×10) 并迁移"""
        model = ScoreModel(pretrained=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        # 检测旧版模型 (regressor.3 输出 1 维) 并迁移到 11 维
        if 'regressor.3.weight' in state_dict:
            old_weight = state_dict['regressor.3.weight']
            if old_weight.shape[0] == 1:
                # 迁移策略：W_new[i] = W_old * (2*i/10 - 1)
                #            b_new[i] = b_old * (2*i/10 - 1)
                # 使得旧模型预测高分 → 新模型分布峰在高分段
                old_bias = state_dict.pop('regressor.3.bias')
                scale_factors = torch.tensor(
                    [2.0 * i / 10.0 - 1.0 for i in range(NUM_CLASSES)],
                    dtype=torch.float32, device=old_weight.device,
                )
                state_dict['regressor.3.weight'] = (
                    old_weight.repeat(NUM_CLASSES, 1) * scale_factors.unsqueeze(1)
                )
                state_dict['regressor.3.bias'] = (
                    old_bias.repeat(NUM_CLASSES) * scale_factors
                )

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model


# ── 用户偏好档案（保留用于修正历史追踪）──────────────────────────

class UserPreferenceProfile:
    """用户偏好档案：跟踪修正历史"""

    def __init__(self):
        self.corrections: list[dict] = []
        self.scale = 1.0
        self.bias = 0.0
        self.preference_mean = 5.0
        self.preference_std = 2.5

    def add_correction(self, raw_score: float, corrected_score: float,
                       filename: str = "") -> None:
        self.corrections.append({
            "raw": float(raw_score),
            "corrected": float(corrected_score),
            "filename": filename,
        })
        self._update_preference_stats()

    def _update_preference_stats(self) -> None:
        if len(self.corrections) == 0:
            return
        scores = [c["corrected"] for c in self.corrections]
        self.preference_mean = sum(scores) / len(scores)
        if len(scores) > 1:
            variance = sum((s - self.preference_mean) ** 2 for s in scores) / (len(scores) - 1)
            self.preference_std = max(0.5, variance ** 0.5)

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "bias": self.bias,
            "corrections": self.corrections,
            "preference_mean": self.preference_mean,
            "preference_std": self.preference_std,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserPreferenceProfile":
        profile = cls()
        profile.scale = data.get("scale", 1.0)
        profile.bias = data.get("bias", 0.0)
        profile.corrections = data.get("corrections", [])
        profile.preference_mean = data.get("preference_mean", 5.0)
        profile.preference_std = data.get("preference_std", 2.5)
        return profile

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserPreferenceProfile":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
