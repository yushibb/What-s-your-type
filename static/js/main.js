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

        if (data.model_updated) {
            const detail = data.update_detail;
            const lossType = detail.loss_type === "extreme_aware_mse" ? "极端感知" : "标准";
            autoText.textContent = `模型已更新 (${lossType} Loss: ${detail.final_loss.toFixed(6)})`;
            showToast(`录入成功！评分 ${data.score}，模型已自动更新（${lossType}损失）`, "success");
        } else {
            autoText.textContent = "模型更新失败，数据已保存";
            showToast(`录入成功！评分 ${data.score}，但模型更新失败`, "warning");
        }

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
                    <span class="sample-score ${parseFloat(s.score) <= 2 || parseFloat(s.score) >= 8 ? 'extreme' : ''}">${parseFloat(s.score).toFixed(1)}</span>
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

        showToast("全量微调已启动（极端感知+过采样+多样性正则）", "info");
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
                    const lossType = r.loss_type || "unknown";
                    const oversampling = r.oversampling ? ` | 过采样: ${r.oversampling}` : "";
                    document.getElementById("trainResult").style.display = "block";
                    document.getElementById("trainResult").innerHTML = `
                        微调完成！数据量: ${r.dataset_size} | 轮数: ${r.epochs} |
                        损失类型: ${lossType}${oversampling} |
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

// ── 概率分布可视化 ────────────────────────────────────

function renderDistribution(distribution, peak, score) {
    const container = document.getElementById("distChart");
    if (!container || !distribution) return;

    const probs = Object.values(distribution);
    const maxProb = Math.max(...probs);
    const scoreRound = Math.round(score);

    container.innerHTML = probs.map((p, i) => {
        const height = maxProb > 0 ? (p / maxProb * 100) : 0;
        const isPeak = i === peak;
        const isScore = i === scoreRound;
        const cls = isPeak ? 'dist-bar peak' : isScore ? 'dist-bar score' : 'dist-bar';
        return `<div class="dist-col">
            <div class="${cls}" style="height:${height}%" title="${i}分: ${(p*100).toFixed(1)}%"></div>
            <span class="dist-label">${i}</span>
        </div>`;
    }).join('');
}


// ── 分数预测 ────────────────────────────────────────

let _predictFileRef = null;
let _lastRawScore = null;  // 记录最近一次原始预测分数

async function checkModelForPredict() {
    try {
        const res = await fetch("/api/model-info");
        const data = await res.json();

        const noModel = document.getElementById("noModelCard");
        const resultCard = document.getElementById("resultCard");

        if (!data.model_exists || !data.predict_ready) {
            noModel.style.display = "block";
            resultCard.style.display = "none";

            // 更新提示信息
            const noModelText = noModel.querySelector("p");
            if (noModelText) {
                if (!data.model_exists) {
                    noModelText.textContent = "模型尚未训练，请先录入训练数据";
                } else if (!data.predict_ready) {
                    noModelText.textContent = `训练数据不足，当前 ${data.total_samples}/${data.predict_required} 条，至少需要 ${data.predict_required} 条训练数据才能使用预测功能`;
                }
            }
        } else {
            noModel.style.display = "none";
        }

        // 更新校准信息展示
        updateCalibrationDisplay(data.calibration);
    } catch (err) {
        console.error("检查模型状态失败:", err);
    }
}

function updateCalibrationDisplay(calibration) {
    const calInfo = document.getElementById("calibrationInfo");
    if (!calInfo || !calibration) return;

    const { scale, bias, corrections_count } = calibration;
    const isDefault = Math.abs(scale - 1.0) < 0.01 && Math.abs(bias) < 0.001;

    if (isDefault && corrections_count < 2) {
        const hint = corrections_count === 0
            ? '校准: 未启用（修正预测≥2次后自动校准）'
            : `校准: 未启用（已有1条修正，还需≥1条才启用）`;
        calInfo.innerHTML = `<span class="cal-default">${hint}</span>`;
    } else {
        const biasDir = bias > 0.001 ? "偏宽" : bias < -0.001 ? "偏严" : "中性";
        const scaleDir = scale > 1.05 ? "放大" : scale < 0.95 ? "压缩" : "标准";
        calInfo.innerHTML = `<span class="cal-active">校准: ${scaleDir}(${scale.toFixed(2)}) / ${biasDir}(${(bias * 10).toFixed(2)}) | 修正${corrections_count}次</span>`;
    }
}

async function predictScore() {
    const fileInput = document.getElementById("predictImage");

    if (!fileInput.files || !fileInput.files[0]) {
        showToast("请先选择一张图片", "error");
        return;
    }

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
            if (data.error && (data.error.includes("尚未训练") || data.error.includes("训练数据不足"))) {
                document.getElementById("noModelCard").style.display = "block";
                const noModelText = document.getElementById("noModelCard").querySelector("p");
                if (noModelText) noModelText.textContent = data.error;
            }
            return;
        }

        // 记录原始预测分数（供修正时使用）
        _lastRawScore = data.score;

        const resultCard = document.getElementById("resultCard");
        resultCard.style.display = "block";
        document.getElementById("noModelCard").style.display = "none";

        const score = data.score;
        document.getElementById("resultScore").textContent = score.toFixed(1);

        // 修正输入默认值 = AI 评分
        document.getElementById("correctScoreInput").value = score.toFixed(1);

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

        // 元信息（含概率分布信息）
        let meta = `${data.model} | MTCNN人脸裁剪`;
        if (data.confidence) meta += ` | 置信度: ${(data.confidence * 100).toFixed(0)}%`;
        if (data.std_dev) meta += ` | 犹豫度: ${data.std_dev.toFixed(1)}`;
        meta += ` | ${new Date().toLocaleTimeString()}`;
        document.getElementById("resultMeta").textContent = meta;

        // 概率分布条形图
        renderDistribution(data.distribution, data.peak, score);

        showToast(`预测评分：${score.toFixed(1)} 分（置信度${data.confidence ? (data.confidence * 100).toFixed(0) + '%' : 'N/A'}）`, "success");
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
 * 提交评分修正 → 保存为训练样本 → 更新模型 + 更新校准层
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
    // 传递原始预测分数（用于校准层训练）
    formData.append("raw_score", (_lastRawScore || 0).toString());

    try {
        const res = await fetch("/api/predict-correct", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "提交修正失败", "error");
            return;
        }

        const updateMsg = data.model_updated ? "，模型已自动更新" : "，但模型更新失败";
        showToast(`修正评分 ${score.toFixed(1)} 已保存为训练样本${updateMsg}`, "success");

        _predictFileRef = null;
        _lastRawScore = null;

        updateModelBadge();
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
    }
}

function discardCorrection() {
    _predictFileRef = null;
    _lastRawScore = null;
    showToast("图片已丢弃，不会保存为训练样本", "info");
}

// ── RLHF 偏好对齐 ──────────────────────────────────

async function startRlhfUpdate() {
    const profile = await fetch("/api/user-profile").then(r => r.json()).catch(() => null);

    if (!profile || profile.corrections_count < 2) {
        showToast(`修正记录不足（${profile?.corrections_count || 0} 条），至少需要 2 条修正才能进行 RLHF 对齐`, "warning");
        return;
    }

    if (!confirm(`将使用 ${profile.corrections_count} 条修正记录进行 RLHF 偏好对齐。\n模型将在 KL 惩罚约束下向您的偏好靠拢，继续？`)) {
        return;
    }

    const btn = document.getElementById("rlhfBtn");
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner"></div> RLHF 对齐中...`;

    try {
        const res = await fetch("/api/rlhf-update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs: 10, lr: 1e-4, kl_weight: 0.1, preference_strength: 2.0 }),
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "RLHF 启动失败", "error");
            btn.disabled = false;
            btn.innerHTML = `RLHF 偏好对齐`;
            return;
        }

        showToast(`RLHF 偏好对齐已启动（${data.corrections_count} 条修正）`, "info");
        pollRlhfStatus();
    } catch (err) {
        showToast("网络错误：" + err.message, "error");
        btn.disabled = false;
        btn.innerHTML = `RLHF 偏好对齐`;
    }
}

function pollRlhfStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch("/api/train-status");
            const data = await res.json();

            const btn = document.getElementById("rlhfBtn");
            if (btn) {
                btn.innerHTML = `<div class="spinner"></div> ${data.progress}`;
            }

            if (!data.is_training) {
                clearInterval(interval);
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = `RLHF 偏好对齐`;
                }

                if (data.result && !data.result.error) {
                    const r = data.result;
                    showToast(`RLHF 对齐完成！修正${r.corrections_used}条 | KL权重${r.kl_weight} | 偏好强度${r.preference_strength}`, "success");
                    updateModelBadge();
                } else if (data.result && data.result.error) {
                    showToast("RLHF 对齐失败：" + data.result.error, "error");
                }
            }
        } catch (err) {
            clearInterval(interval);
        }
    }, 2000);
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
            let statusText = `模型就绪 (${data.model_size_mb}MB | ${data.total_samples}条)`;
            if (data.gpu && data.gpu.available) {
                statusText += ` | GPU: ${data.gpu.device_name}`;
            }
            text.textContent = statusText;
            if (exportBtn) exportBtn.disabled = false;
        } else {
            badge.classList.remove("ready");
            text.textContent = "模型未训练";
            if (exportBtn) exportBtn.disabled = true;
        }

        // 更新校准信息
        updateCalibrationDisplay(data.calibration);
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
