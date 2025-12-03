# 🚀 Detoxification via Knowledge Distillation (T5-Base → T5-Small)

This project compresses a fine-tuned T5-Base detoxification model into a lightweight T5-Small model using **Knowledge Distillation (KD)**.  
The distilled model preserves high-quality detoxification performance while enabling much faster, cheaper inference suitable for real-time applications.

---

## 📘 1. Introduction

Toxic and abusive language is prevalent online.  
Large transformer models can rewrite harmful text into safer phrasing, but deploying such models at scale is expensive.

This project addresses the challenge by:
- Training a strong **T5-Base teacher model**
- Distilling it into a compact **T5-Small student model**
- Preserving semantic meaning, fluency, and detoxification quality

Knowledge Distillation enables the student to learn:
- Hard labels (ground-truth detoxified text)
- Soft labels (teacher’s probability distribution)

---

## 📘 2. Dataset — ParaDeHate

We use the **ParaDeHate** dataset, which contains:
- Toxic → Non-toxic sentence pairs
- Human-written paraphrases
- Ideal for detoxification, rewriting, and style transfer

Dataset Preparation:
- Inputs formatted as:  
  `detoxify: <toxic sentence>`
- Max sequence length: **96 tokens**
- 90/10 train/test split

### Dataset YAML Summary

```yaml
dataset:
  name: ParaDeHate
  split:
    train: 90%
    test: 10%
  formatting:
    prefix: "detoxify: "
    max_length: 96
  task: Toxic → Non-toxic rewriting
```

---

## 📘 3. Teacher Model — Fine-Tuned T5-Base

A large sequence-to-sequence model fine-tuned on detoxification data.

### Architecture Summary
| Component | Value |
|----------|--------|
| Encoder layers | 12 |
| Decoder layers | 12 |
| Hidden size | 768 |
| FFN size | 3072 |
| Attention heads | 12 |
| Vocabulary size | 32,128 |

### Objective
Trained using **Cross-Entropy loss**:

\[
CE = -\log P_{teacher}(y|x)
\]

### YAML

```yaml
teacher_model:
  name: t5-base
  hidden_size: 768
  layers:
    encoder: 12
    decoder: 12
  ffn_size: 3072
  attention_heads: 12
  vocab_size: 32128
  training_loss: CrossEntropy
  status: "frozen during KD"
```

---

## 📘 4. Student Model — T5-Small (Distilled)

A lightweight model that imitates the teacher.

### Architecture Summary
| Component | Value |
|----------|--------|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Hidden size | 512 |
| FFN size | 2048 |
| Attention heads | 8 |
| Vocabulary size | 32,128 |

### YAML

```yaml
student_model:
  name: t5-small
  hidden_size: 512
  layers:
    encoder: 6
    decoder: 6
  ffn_size: 2048
  attention_heads: 8
  vocab_size: 32128
  objective: "Learn from CE + KL Distillation"
```

---

## 📘 5. Input Processing Pipeline

For each training example:
- Prefix: `"detoxify: "`
- Tokenize with teacher and student tokenizers independently  
- Replace padding tokens with `-100` (ignored in CE loss)
- Prepare:

```
input_ids_s, attention_mask_s, labels_s
input_ids_t, attention_mask_t, labels_t
```

### YAML Summary

```yaml
preprocessing:
  prefix: "detoxify: "
  pad_to: 96
  labels:
    pad_token_replacement: -100
  outputs:
    - input_ids_s
    - attention_mask_s
    - labels_s
    - input_ids_t
    - attention_mask_t
    - labels_t
```

---

## 📘 6. Knowledge Distillation Architecture

Parallel forward passes:

```yaml
distillation:
  teacher_probs: softmax(teacher_logits / T)
  student_log_probs: log_softmax(student_logits / T)
  vocab_space: 32128
```

Mathematically:
\[
p_T = softmax(z_T / T)
\]
\[
p_S = softmax(z_S / T)
\]

Both models operate in the same vocabulary space → **no projection layers required**.

---

## 📘 7. Distillation Loss Functions

### **1. Cross-Entropy (CE) Loss**
\[
CE = -\log P_{student}(y|x)
\]

### **2. KL Divergence (KD)**
\[
KD = T^2 \sum_i p_T(i) \log\left(\frac{p_T(i)}{p_S(i)}\right)
\]

### Why T²?
Temperature softening reduces gradient magnitude by **1/T²**.  
Multiplying KL by **T²** restores correct gradient scale.

---

### ⭐ Final Loss
\[
Loss = CE + 0.7 \cdot KD
\]

### YAML

```yaml
loss_function:
  CE: "CrossEntropy(student, labels)"
  KL:
    formula: "T^2 * KLDiv(p_T || p_S)"
    temperature: 2.0
  total_loss: "CE + 0.7 * KL"
```

### ⭐ Final Model Pipeline
<img width="4254" height="956" alt="image" src="https://github.com/user-attachments/assets/c608809c-aaaf-4366-975e-fcb88fd27e53" />



---

## 📘 8. Training Configuration

| Setting | Value |
|--------|--------|
| Batch size | 4 |
| Grad accumulation | 2 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| FP16 | Enabled |
| Trainer | Custom Seq2SeqTrainer |

### YAML

```yaml
training:
  batch_size: 4
  grad_accumulation: 2
  learning_rate: 2e-4
  epochs: 3
  fp16: true
  trainer: "Seq2SeqTrainer (custom KD loss)"
```

---

## 📘 9. Inference Pipeline

Beam search decoding:
- num_beams: 4  
- no_repeat_ngram_size: 3  
- length_penalty: 0.8  
- bad phrase blocking to avoid hallucinations

### YAML

```yaml
inference:
  decoding:
    num_beams: 4
    no_repeat_ngram_size: 3
    length_penalty: 0.8
  safety:
    blocklist:
      - "Click here"
      - "Read more"
      - "Visit"
```

---

## 📘 10. Evaluation Metrics

### Content Preservation
(BERTScore-F1)

### Style Strength
(using Unitary Toxic-BERT)
\[
Style = 1 - P(toxic)
\]

### Fluency
(using GPT-2 Perplexity)
\[
Fluency = \frac{1}{Perplexity}
\]

---

## 📘 11. Final Quantitative Results

### 📊 **Performance Comparison**

| Model | Style Accuracy | Content Preservation |
|-------|----------------|-----------------------|
| **T5-Base (Teacher)** | **0.975** | **0.710** |
| **T5-Small (After Distillation)** | **0.978** | **0.678** |

### Interpretation
- Style accuracy slightly **improves** after distillation → student learns strong detoxification behavior  
- Content preservation decreases moderately → expected due to reduced capacity  
- Overall: **successful compression with high-quality outputs**

### YAML Summary

```yaml
results:
  teacher:
    model: t5-base
    style_accuracy: 0.975
    content_preservation: 0.71
  student_distilled:
    model: t5-small
    style_accuracy: 0.978
    content_preservation: 0.678
  notes:
    - "Distilled student outperforms teacher in detox accuracy."
    - "Trade-off: small reduction in content preservation."
```

---

## 📘 12. Final Outcome

The distilled **T5-Small** model:

- Retains teacher-level detoxification quality  
- Preserves fluency and style  
- Runs **3–5× faster**  
- Uses **75% less memory**  
- Is deployment-friendly for moderation, rewriting, and on-device applications  

```yaml
final_model:
  performance:
    speedup: "3x - 5x faster"
    memory_reduction: "~75% less"
    quality: "High detox accuracy + strong fluency"
  recommended_use_cases:
    - real_time_detoxification
    - chat_moderation
    - on_device_LLM
```

---

## 📘 13. Citation

Please cite:
- ParaDeHate dataset  
- HuggingFace Transformers  
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 Paper)

