# MediDefine-AI

A hands-on QLoRA fine-tuning of a Llama-2 model on Wikipedia medical-term definitions, using 4-bit NF4 quantisation, PEFT LoRA adapters and TRL's `SFTTrainer`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AryadipMridha/MediDefine-AI/blob/main/Medi_Define_AI.ipynb)

## About this project

This project works through the parameter-efficient fine-tuning workflow end to end: load a 7B-class model in 4-bit, attach LoRA adapters, run supervised fine-tuning on an instruction-formatted dataset, and query the result. It builds on a notebook from a Hugging Face LLM fine-tuning course. The training run is intentionally small (see [Limitations](#limitations)). The point is to exercise the full QLoRA pipeline on a single Colab GPU, not to produce a production medical model.

## Pipeline

1. **Load the base model in 4-bit.** `aboonaji/llama2finetune-v2`, a Llama-2 checkpoint on the Hugging Face Hub (two fp16 shards, about 13.5 GB in total), is loaded with `bitsandbytes` NF4 quantisation and fp16 compute. `use_cache` is disabled for training and `pretraining_tp` is set to 1.
2. **Prepare the tokenizer.** The pad token is set to the EOS token, with right-side padding.
3. **Load the data.** The `train` split of [`aboonaji/wiki_medical_terms_llam2_format`](https://huggingface.co/datasets/aboonaji/wiki_medical_terms_llam2_format) has 6,861 examples. They are already in Llama-2 chat format (`<s>[INST] ... [/INST] ...`) in a single `text` column.
4. **Fine-tune with LoRA.** `SFTTrainer` freezes the quantised base weights and trains only the low-rank adapter matrices. Gradient checkpointing is switched on automatically by PEFT's k-bit preparation, as the training logs show.
5. **Generate.** A `transformers` text-generation pipeline answers prompts wrapped in the Llama-2 template `<s>[INST] {prompt} [/INST]`.

## Configuration

| Component | Setting |
|---|---|
| Base model | `aboonaji/llama2finetune-v2` (Llama-2, 7B-class) |
| Quantisation | 4-bit NF4 (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`), fp16 compute dtype |
| LoRA rank `r` | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| LoRA target modules | Not set, so PEFT's default for Llama applies (`q_proj`, `v_proj`) |
| Task type | `CAUSAL_LM` |
| Dataset | `aboonaji/wiki_medical_terms_llam2_format`, `train` split (6,861 rows) |
| Max sequence length | 1024 (the `SFTTrainer` default; not set explicitly) |
| Per-device batch size | 4 |
| Max steps | 100 |
| Optimiser, LR, schedule | `transformers` 4.31 defaults, not set explicitly: AdamW, learning rate 5e-5, linear decay |
| Generation | `max_length=300` |

## Training run

These are the statistics reported by `sft_trainer.train()` on a Colab GPU runtime:

| Key                            | Value                           |
|--------------------------------|---------------------------------|
| global_step                    | 100                             |
| training_loss                  | 1.6548727416992188              |
| train_runtime                  | 1446.394                        |
| train_samples_per_second       | 0.277                           |
| train_steps_per_second         | 0.069                           |
| total_flos                     | 8228119310991360.0              |
| train_loss                     | 1.6548727416992188              |
| epoch                          | 0.06                            |

`train_runtime` is in seconds (about 24 minutes). Over 100 steps at batch size 4 the model saw 400 examples, about 6% of one pass over the 6,861-row dataset, which is where `epoch = 0.06` comes from.

### Sample output

Prompt: `Please tell me about sclera`

> The sclera is the white, tough outer layer of the eyeball. everybody has one. It is the largest part of the eyeball, making up about 80% of the eye's volume. The sclera is a tough, protective layer that helps protect the eye from injury and infection. ...

The answer is fluent and on topic, but it is not reliable. For example, the sclera covers most of the eye's outer surface; it does not make up 80% of the eye's volume. The response is also cut off by `max_length=300`.

## Limitations

- **This is a demo-scale run.** 100 steps and 0.06 epochs is a smoke test that the QLoRA pipeline works end to end. It is not a converged fine-tune.
- **There is no evaluation.** No validation split or metric is used. The only number recorded is the average training loss, and the default logging interval (500 steps) is longer than the run, so no per-step loss curve was logged.
- **There is no baseline comparison.** The base model was not queried before fine-tuning, so the sample answer cannot be attributed to the adapter rather than the base checkpoint.
- **The adapter is not saved.** The notebook never calls `save_model` or `push_to_hub`, and the default checkpoint interval (500 steps) is longer than the run, so the trained adapter existed only in the Colab session.
- **The model can produce confident factual errors.** Do not use it for medical advice.
- **The dependency pins are from mid-2023** (`transformers` 4.31, `bitsandbytes` 0.40.2). They match the original run, but current Colab runtimes may need a newer stack.

## How to run

### Google Colab (recommended)

1. Open `Medi_Define_AI.ipynb` with the **Open in Colab** badge above.
2. Choose **Runtime → Change runtime type → GPU**. 4-bit loading with `bitsandbytes` needs a CUDA GPU.
3. Run all cells. The first cell installs the pinned libraries, and the first model load downloads about 13.5 GB of weights.
4. To ask your own question, edit `user_prompt` in the last cell and run it again.

### Script (Linux with a CUDA GPU)

```bash
git clone https://github.com/AryadipMridha/MediDefine-AI.git
cd MediDefine-AI
pip install -r requirements.txt
python Medi_Define_AI.py
```

## Repository layout

| File | Contents |
|---|---|
| `Medi_Define_AI.ipynb` | Colab notebook with saved outputs (training stats, sample generation) |
| `Medi_Define_AI.py` | The same pipeline as a plain Python script |
| `requirements.txt` | Library versions pinned by the original notebook |
