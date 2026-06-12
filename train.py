"""
增量训练模块 v2：成对排序 + 回放缓存 + 概率分布输出

核心策略（v2 升级）：
  - 回归头输出 11 类概率分布，期望值作为评分（软化噪声）
  - 成对排序损失 (Pairwise Ranking Loss)：优先学习相对偏好顺序
  - 回放缓存 (Replay Buffer)：单样本更新时混合历史样本防抖动
  - 高斯软标签 (Soft Label)：将主观模糊性转化为概率分布方差
  - 极端感知加权 + 过采样：缓解预测趋中
  - 预计算 512 维嵌入缓存，MLP 在缓存上快速训练
"""

import os
import gc
import csv
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from model import ScoreModel, detect_and_crop_face, UserPreferenceProfile, NUM_CLASSES


# ── 全局嵌入缓存 ──────────────────────────────────────────────

_embedding_cache: dict[str, torch.Tensor] = {}


def get_cached_embedding(
    image_path: str,
    model: ScoreModel,
    device: torch.device,
    use_mtcnn: bool = True,
) -> torch.Tensor:
    if image_path in _embedding_cache:
        return _embedding_cache[image_path]
    embedding = model.extract_embedding_from_image(image_path, device, use_mtcnn)
    _embedding_cache[image_path] = embedding
    return embedding


def clear_embedding_cache():
    _embedding_cache.clear()


# ── 高斯软标签 ──────────────────────────────────────────────

def make_soft_labels(scores: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """将 0-10 评分转为 11 类高斯软标签分布

    scores=7.5, sigma=1.0 → P(7)≈0.35, P(8)≈0.35, P(6)≈0.13, P(9)≈0.13
    主观模糊性（觉得 7 也行 8 也行）自然编码为分布宽度。

    Args:
        scores: [B] 评分张量 (0-10)
        sigma: 高斯宽度，越大越平滑
    Returns:
        [B, 11] 软标签概率分布
    """
    classes = torch.arange(NUM_CLASSES, dtype=torch.float32, device=scores.device)
    diff = classes.unsqueeze(0) - scores.unsqueeze(1)  # [B, 11]
    dist = torch.exp(-0.5 * (diff / sigma) ** 2)
    return dist / dist.sum(dim=-1, keepdim=True)


# ── 嵌入向量数据集 ──────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """基于预计算 512 维嵌入向量的轻量数据集（返回原始 0-10 评分）"""

    def __init__(self, embeddings: list[torch.Tensor], scores: list[float]):
        self.embeddings = embeddings
        self.scores = scores

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.scores[idx], dtype=torch.float32)


# ════════════════════════════════════════════════════════════════
# 损失函数
# ════════════════════════════════════════════════════════════════

class ExtremeAwareMSELoss(nn.Module):
    """极端感知加权 MSE 损失

    weight(score) = 1 + α × (|score - center| / half_range)^power
    score=0 或 10 → 权重 4x，score=5 → 权重 1x
    """

    def __init__(self, center=5.0, half_range=5.0, alpha=3.0, power=2.0):
        super().__init__()
        self.center = center
        self.half_range = half_range
        self.alpha = alpha
        self.power = power

    def forward(self, pred, target):
        """pred, target: 均为 0-10 范围"""
        distance = (target - self.center).abs() / self.half_range
        weight = 1.0 + self.alpha * distance.pow(self.power)
        return (weight * (pred - target).pow(2)).mean()


class PairwiseRankingLoss(nn.Module):
    """成对排序损失 (Margin Ranking Loss)

    核心思想：不要求模型精确输出绝对分数，只要求相对顺序正确。
    如果 A 的评分 > B 的评分，则 f(A) 应大于 f(B) + margin。

    L_rank = max(0, margin - (f(A) - f(B)))  当 score(A) > score(B)
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor):
        """向量化成对排序损失

        Args:
            pred_scores: [B] 预测评分 (0-10)
            target_scores: [B] 真实评分 (0-10)
        """
        n = pred_scores.size(0)
        if n < 2:
            return pred_scores.sum() * 0.0  # 保持计算图，返回 0

        # 成对差分矩阵
        pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)   # [B, B]
        target_diff = target_scores.unsqueeze(1) - target_scores.unsqueeze(0)

        # 只考虑有意义的配对（评分差 > 0.5）
        threshold = 0.5
        valid = target_diff > threshold  # target[i] > target[j] + 0.5

        # 排序违规：当 target[i] > target[j] 但 pred[i] - pred[j] < margin
        violations = torch.relu(self.margin - pred_diff)

        num_pairs = valid.sum()
        if num_pairs > 0:
            return (violations * valid.float()).sum() / num_pairs
        return pred_scores.sum() * 0.0


# ════════════════════════════════════════════════════════════════
# 回放缓存 (Replay Buffer)
# ════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """经验回放缓存：单样本更新时混合历史样本防抖动

    核心问题：只用 1 张新图做 3 步 SGD，模型"学了新图，忘了旧图"。
    解决方案：维护最近 50 张图片的 (embedding, score)，
    更新时从 Buffer 抽取 3~5 张老样本 + 新样本组成 mini-batch。
    老样本充当"锚点"，确保模型只微调偏好，不发生剧烈振荡。
    """

    def __init__(self, capacity: int = 50, persist_path: str = None):
        self.capacity = capacity
        self.persist_path = persist_path
        self.embeddings: list[torch.Tensor] = []
        self.scores: list[float] = []
        self._load()

    def add(self, embedding: torch.Tensor, score: float):
        self.embeddings.append(embedding.cpu())
        self.scores.append(score)
        if len(self.scores) > self.capacity:
            self.embeddings.pop(0)
            self.scores.pop(0)
        self._save()

    def sample(self, n: int):
        """随机采样 n 条，返回 (embeddings_list, scores_list)"""
        if len(self.scores) == 0:
            return [], []
        indices = random.sample(range(len(self.scores)), min(n, len(self.scores)))
        return [self.embeddings[i] for i in indices], [self.scores[i] for i in indices]

    def __len__(self):
        return len(self.scores)

    def _save(self):
        if self.persist_path and len(self.embeddings) > 0:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            data = {
                'embeddings': torch.stack(self.embeddings),
                'scores': self.scores,
            }
            torch.save(data, self.persist_path)

    def _load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                data = torch.load(self.persist_path, map_location='cpu', weights_only=True)
                self.embeddings = [data['embeddings'][i] for i in range(data['embeddings'].size(0))]
                self.scores = data['scores']
            except Exception:
                self.embeddings = []
                self.scores = []


# ════════════════════════════════════════════════════════════════
# 过采样权重计算
# ════════════════════════════════════════════════════════════════

def compute_oversample_weights(scores: list[float], center: float = 5.0,
                                half_range: float = 5.0, alpha: float = 3.0,
                                power: float = 2.0) -> list[float]:
    weights = []
    for score in scores:
        distance = abs(score - center) / half_range
        w = 1.0 + alpha * (distance ** power)
        weights.append(w)
    return weights


# ════════════════════════════════════════════════════════════════
# 单样本在线更新（v2: 回放缓存 + 软标签）
# ════════════════════════════════════════════════════════════════

def online_update_single(
    image_path: str,
    score: float,
    model_save_path: str,
    lr: float = 1e-4,
    steps: int = 3,
    use_mtcnn: bool = True,
    use_extreme_loss: bool = True,
) -> dict:
    """单样本在线更新（v2: 回放缓存 + 软标签 + 概率分布）

    改进：
    1. 新样本 + 回放缓存历史样本组成 mini-batch，防抖动
    2. 高斯软标签交叉熵 + 极端感知 MSE 双重损失
    3. 概率分布输出，期望值作为评分
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_or_init_model(model_save_path, device)
    model.train()
    model.backbone.eval()

    # 提取新样本嵌入
    new_embedding = get_cached_embedding(image_path, model, device, use_mtcnn)

    # 加入回放缓存
    replay_path = os.path.join(os.path.dirname(model_save_path), "..", "data", "replay_buffer.pt")
    replay_buffer = ReplayBuffer(capacity=50, persist_path=os.path.abspath(replay_path))
    replay_buffer.add(new_embedding.cpu(), score)

    # 从缓存采样历史样本
    old_embs, old_scores = replay_buffer.sample(4)

    # 组装 batch
    batch_embs = [new_embedding.to(device)]
    batch_scores = [score]
    for emb, s in zip(old_embs, old_scores):
        batch_embs.append(emb.to(device))
        batch_scores.append(s)

    batch_emb_tensor = torch.stack(batch_embs)
    batch_score_tensor = torch.tensor(batch_scores, dtype=torch.float32, device=device)

    # 损失函数
    extreme_mse = ExtremeAwareMSELoss() if use_extreme_loss else None

    optimizer = optim.Adam(model.regressor.parameters(), lr=lr)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()

        # 前向：logits → 概率 → 期望评分
        logits = model.regressor(batch_emb_tensor)
        probs = model.logits_to_probs(logits)
        pred_scores = model.logits_to_score(logits).squeeze(-1)  # [B], 0-10

        # 1. 极端感知 MSE（在 0-10 空间）
        if extreme_mse is not None:
            mse_loss = extreme_mse(pred_scores, batch_score_tensor)
        else:
            mse_loss = F.mse_loss(pred_scores, batch_score_tensor)

        # 2. 高斯软标签 KL 散度
        soft_targets = make_soft_labels(batch_score_tensor, sigma=1.0)
        log_probs = torch.log_softmax(logits, dim=-1)
        ce_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')

        # 总损失
        loss = mse_loss + 0.5 * ce_loss
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 6))

    # 保存
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "online_update",
        "steps": steps,
        "losses": losses,
        "final_loss": losses[-1],
        "loss_type": "mse+soft_ce",
        "replay_buffer_size": len(replay_buffer),
    }


# ════════════════════════════════════════════════════════════════
# 全量微调（v2: 成对排序 + 软标签 + 极端感知 + 过采样）
# ════════════════════════════════════════════════════════════════

def finetune_model(
    csv_path: str,
    image_dir: str,
    model_save_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-5,
    use_mtcnn: bool = True,
    use_extreme_loss: bool = True,
) -> dict:
    """全量微调（v2: 成对排序 + 软标签 + 概率分布）

    三重损失约束：
    1. 极端感知 MSE：绝对分数拟合（0-10 空间）
    2. 高斯软标签 KL 散度：容忍主观噪声
    3. 成对排序损失：确保相对偏好顺序正确
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 读取 CSV ──
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(image_dir, row["filename"])
            if os.path.exists(img_path):
                samples.append((img_path, float(row["score"])))

    if len(samples) == 0:
        return {"error": "没有可用的训练数据"}

    # ── 预计算嵌入 ──
    model = _load_or_init_model(model_save_path, device)
    model.eval()

    embeddings = []
    scores = []
    for img_path, score in samples:
        emb = get_cached_embedding(img_path, model, device, use_mtcnn)
        embeddings.append(emb)
        scores.append(score)

    # ── 训练 ──
    model.train()
    model.backbone.eval()

    dataset = EmbeddingDataset(embeddings, scores)

    # 极端分数过采样
    sample_weights = compute_oversample_weights(scores)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights) * 2,
        replacement=True,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    # 损失函数
    extreme_mse = ExtremeAwareMSELoss() if use_extreme_loss else None
    ranking_loss_fn = PairwiseRankingLoss(margin=1.0)

    optimizer = optim.Adam(model.regressor.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for emb_batch, score_batch in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)  # 0-10

            optimizer.zero_grad()

            # 前向
            logits = model.regressor(emb_batch)
            probs = model.logits_to_probs(logits)
            pred_scores = model.logits_to_score(logits).squeeze(-1)  # [B], 0-10

            # 1. 极端感知 MSE
            if extreme_mse is not None:
                mse_loss = extreme_mse(pred_scores, score_batch)
            else:
                mse_loss = F.mse_loss(pred_scores, score_batch)

            # 2. 高斯软标签 KL 散度
            soft_targets = make_soft_labels(score_batch, sigma=1.0)
            log_probs = torch.log_softmax(logits, dim=-1)
            ce_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')

            # 3. 成对排序损失
            rank_loss = ranking_loss_fn(pred_scores, score_batch)

            # 4. 分布多样性正则
            diversity_penalty = torch.tensor(0.0, device=device)
            if pred_scores.numel() > 1:
                diversity_penalty = 0.05 / (pred_scores.std() + 0.05)

            # 总损失
            loss = mse_loss + 0.5 * ce_loss + 0.3 * rank_loss + diversity_penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        loss_history.append(round(avg_loss, 6))

    # 保存
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    del model, optimizer, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "finetune",
        "epochs": epochs,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "dataset_size": len(samples),
        "loss_type": "mse+soft_ce+pairwise_ranking",
        "oversampling": "extreme_weighted_2x",
    }


# ════════════════════════════════════════════════════════════════
# RLHF 风格偏好对齐（保留，适配 v2 架构）
# ════════════════════════════════════════════════════════════════

def rlhf_preference_update(
    model_save_path: str,
    user_profile: UserPreferenceProfile,
    epochs: int = 10,
    lr: float = 1e-4,
    kl_weight: float = 0.1,
    preference_strength: float = 2.0,
) -> dict:
    """RLHF 风格偏好对齐更新（v2: 适配概率分布输出）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(user_profile.corrections) == 0:
        return {"error": "没有修正记录"}

    train_model = _load_or_init_model(model_save_path, device)
    train_model.train()
    train_model.backbone.eval()

    ref_model = copy.deepcopy(train_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    image_dir = os.path.join(os.path.dirname(model_save_path), "..", "data", "images")
    image_dir = os.path.abspath(image_dir)

    valid_embeddings = []
    valid_targets = []

    with torch.no_grad():
        for corr in user_profile.corrections:
            filename = corr.get("filename", "")
            if not filename:
                continue
            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                continue
            emb = train_model.extract_embedding_from_image(img_path, device).to(device)
            valid_embeddings.append(emb)
            valid_targets.append(corr["corrected"])

    if len(valid_embeddings) == 0:
        return {"error": "修正记录中没有可用的图片文件"}

    optimizer = optim.Adam(train_model.regressor.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0

        for emb, target_val in zip(valid_embeddings, valid_targets):
            optimizer.zero_grad()

            logits = train_model.regressor(emb.unsqueeze(0))
            pred_score = train_model.logits_to_score(logits).squeeze()

            with torch.no_grad():
                ref_logits = ref_model.regressor(emb.unsqueeze(0))
                ref_score = ref_model.logits_to_score(ref_logits).squeeze()

            target_t = torch.tensor(target_val, device=device)

            # 偏好损失
            pref_loss = F.mse_loss(pred_score, target_t) * preference_strength

            # KL 惩罚
            kl_loss = ((pred_score - ref_score) ** 2) * kl_weight

            # 软标签
            soft_target = make_soft_labels(target_t.unsqueeze(0), sigma=1.0)
            log_probs = torch.log_softmax(logits, dim=-1)
            ce_loss = F.kl_div(log_probs, soft_target, reduction='batchmean')

            loss = pref_loss + kl_loss + 0.3 * ce_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        n = len(valid_embeddings)
        loss_history.append(round(total_loss / n, 6))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(train_model.state_dict(), model_save_path)

    del train_model, ref_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "rlhf_preference_update",
        "epochs": epochs,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "corrections_used": len(valid_embeddings),
        "kl_weight": kl_weight,
        "preference_strength": preference_strength,
    }


def optimize_calibration(user_profile, lr=0.01, steps=100):
    """校准层优化（已禁用，保留接口兼容性）"""
    return {"mode": "calibration_optimization", "status": "disabled"}


# ── 辅助函数 ──────────────────────────────────────────────

def _load_or_init_model(model_save_path: str, device: torch.device) -> ScoreModel:
    if os.path.exists(model_save_path):
        return ScoreModel.load_model(model_save_path, device)
    else:
        model = ScoreModel(pretrained=True)
        model.to(device)
        return model
