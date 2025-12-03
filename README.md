# Deep-Learning
1. Problem Statement

Online platforms encounter large volumes of toxic, abusive, and hateful language.
While large language models (LLMs) can rewrite such inputs into neutral or polite phrasing, they are slow, expensive, and difficult to deploy at scale.

Goal of this project:
Build a small, fast, low-cost model that can:

Remove toxic language

Preserve the original meaning

Maintain fluent, natural writing style

We achieve this through knowledge distillation:
training a compact T5-Small model to mimic a fine-tuned T5-Base teacher.

2. Dataset: ParaDeHate

We train and evaluate our model using the ParaDeHate dataset:

Contains paired toxic → non-toxic rewrites

Covers many forms of abusive, hateful, and offensive language

Includes rewritten text that maintains meaning while removing toxicity

Allows supervised detoxification training

This dataset ensures the student learns both correctness (gold rewrites) and style safety.

3. Model Architecture
Teacher Model: T5-Base (Fine-tuned)-

Produces high-quality detoxified outputs

Provides soft probability distributions for distillation

Frozen during training (no gradient updates)

Student Model: T5-Small-

Lightweight and efficient

Learns from both gold targets and teacher predictions

Final deployment model (fast inference)

Loss Function-

The training objective combines:

Cross-Entropy Loss
Teaches the model to output the correct detoxified sentence.

KL Divergence Loss
Teaches the student to imitate the teacher’s probability distribution (its “style” and preferences).

Final loss:

Loss = CE(student, target) + λ * KL(student || teacher)

This gives the student teacher-level style accuracy without heavy computation.

4. Pipeline Overview
Step 1 — Load Dataset

Load ParaDeHate

Split into train/test

Extract toxic inputs and gold detoxified targets

Step 2 — Load Teacher and Student

Teacher: Fine-tuned T5-Base

Student: Pretrained T5-Small

Teacher is frozen

Step 3 — Dual Tokenization

Teacher and student tokenizers encode each sample separately

Padding tokens replaced with -100 (ignored during loss)

Step 4 — Collator

Produces batches containing both:

Student inputs + labels

Teacher inputs + labels

Decoder input IDs (shifted targets)

Step 5 — Distillation Training

For each sample:

Teacher generates soft logits

Student generates predicted logits

Compute CE + KL loss

Update only the student model

Step 6 — Generation

Use beam search, n-gram blocking, and banned-phrase filtering to ensure:

No hallucinations

No spammy outputs

Clean, fluent text

Step 7 — Evaluation

Metrics computed:

Style accuracy (toxicity reduction)

Content preservation (BERTScore)

Fluency (GPT-2 perplexity)

<img width="744" height="299" alt="image" src="https://github.com/user-attachments/assets/fbe11d8d-d0d4-4737-b1ea-0ee22309a8ff" />

6. Interpretation of Results
   
T5-Base (Teacher)-

Very strong detoxification performance

Good style safety

Medium-level content preservation

Large and expensive model

DistilBART-

Lighter than T5-Base

Loses a significant amount of content preservation (drops to 0.53)

Despite good style accuracy, it is not reliable enough for safe rewriting

Ultimately rejected

T5-Small (Student, After Distillation)-

Matches or slightly exceeds teacher style safety (0.978 vs 0.975)

Significantly improves over DistilBART

Achieves ~95% of teacher performance with a model that is:

Smaller

Cheaper

Faster

In real-world applications, this distilled student model is easier to deploy and maintain.

7. Why This Distillation Works

Teacher provides nuanced “soft” guidance → student learns fluent, safe rewriting

CE loss gives student the correct cleaned-up target

KL divergence teaches the teacher’s style and fluency

Joint training produces a balanced, robust detoxifier

8. Conclusion

This project demonstrates that a compact model can effectively perform textual detoxification when trained with a carefully designed knowledge distillation pipeline.

The final T5-Small student model is:

Fast

Lightweight

Highly accurate

Style-safe

Meaning-preserving

making it ideal for real-time toxicity removal systems, chat moderation, and safe content generation.
