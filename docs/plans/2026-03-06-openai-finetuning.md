# Plan: Fine-Tune OpenAI GPT-4o for Rakuten Product Classification

## Context

The project already has 3 modeling phases (TF-IDF+LightGBM, OpenCV+LightGBM, CLIP+MLP). We add an OpenAI GPT fine-tuning approach for multimodal product classification. **GPT-5 does not support fine-tuning yet**, so we use **`gpt-4o-2024-08-06`** which supports vision fine-tuning. Fine-tune on **1,000 stratified samples only** (cost-effective, sufficient to evaluate the approach).

### Cost Estimates
- Training: 500 samples × ~135 tokens/example × 3 epochs = **~$5 total**
- Test inference: 13,812 examples = **~$9.30**
- **Total estimated cost: ~$15**

## Files to Create/Modify

| File | Action |
|------|--------|
| `models/openai_finetune.py` | **Create** — Main script: prepare JSONL, launch fine-tuning, predict |
| `notebooks/modeling.ipynb` | **Modify** — Add Phase 4 section for OpenAI fine-tuning results |

## Implementation

### Step 1: Create `models/openai_finetune.py`

3 CLI subcommands: `prepare`, `train`, `predict`.

#### 1a. `prepare` — Build JSONL training file

- Load `data/train_cleaned.pkl` (columns: `designation`, `description`, `productid`, `imageid`, `prdtypecode`)
- Stratified sample of 500 examples (fixed, ensures all 27 classes represented)
- For each row:
  - Read image from `images/image_train/image_{imageid}_product_{productid}.jpg`
  - Resize to 512×512, base64-encode as JPEG
  - Build chat message with system prompt + user (text + image) + assistant (label)
- Write to `data/openai_train.jsonl`
- Use `detail: "low"` on images (85 tokens/image)

JSONL format per line:
```json
{
  "messages": [
    {"role": "system", "content": "You are a product classifier. Given the product title, description, and image, output only the product category code (integer)."},
    {"role": "user", "content": [
      {"type": "text", "text": "Title: {designation}\nDescription: {description}"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{b64}", "detail": "low"}}
    ]},
    {"role": "assistant", "content": "{prdtypecode}"}
  ]
}
```

#### 1b. `train` — Upload file and launch fine-tuning job

- Upload JSONL via `client.files.create(purpose="fine-tune")`
- Create job via `client.fine_tuning.jobs.create(model="gpt-4o-2024-08-06")`
- Support `--status JOB_ID` to check job progress

#### 1c. `predict` — Run inference on test set

- Load `data/test_cleaned.pkl`
- For each row, build same message format (image from `images/image_test/`)
- Call fine-tuned model with `max_tokens=10`
- Parse integer category code from response
- Save to `data/openai_test_predictions.csv`
- Add retry logic + rate limiting

### Step 2: Add Phase 4 to modeling notebook

- Document the approach and show sample JSONL entries
- Display results (confusion matrix, per-class F1, weighted F1)
- Compare with previous phases

## Key Design Decisions

- **`detail: "low"`** — reduces cost 10x, sufficient for product classification
- **500 stratified samples** — no full-dataset training, keeps cost under $15
- **Original `designation` + `description`** in prompts (not `text_clean`) — GPT handles raw text better
- **Simple integer label** as assistant response — minimal output tokens

## Usage

```bash
# 1. Build the 1,000-sample JSONL file (~2 min to encode images)
python models/openai_finetune.py prepare

# 2. Upload and launch fine-tuning (~$10, takes 30-60 min on OpenAI servers)
python models/openai_finetune.py train

# 3. Check job status (poll until status=succeeded) ftjob-JblMQR92SNTZNJDBHxPagFXy
python models/openai_finetune.py train --status ftjob-xxx

# 4. Run predictions on the 13,812 test examples (~$9)
python models/openai_finetune.py predict --model ft:gpt-4o-2024-08-06:your-org::id
```

> `OPENAI_API_KEY` is loaded automatically from `.env` at the project root.

### Fine-tuning job status meanings

| Status           | Meaning                    |
|------------------|----------------------------|
| `validating_files` | Checking your JSONL format |
| `queued`         | Waiting to start           |
| `running`        | Training in progress       |
| `succeeded`      | Done — model ready         |
| `failed`         | Something went wrong       |
| `cancelled`      | Manually cancelled         |

## Verification

1. `python models/openai_finetune.py prepare` → inspect the 1,000-sample JSONL
2. `python models/openai_finetune.py train` → launch fine-tuning (~$10)
3. `python models/openai_finetune.py train --status JOB_ID` → check progress
4. `python models/openai_finetune.py predict --model ft:gpt-4o-...` → predict test set (~$9)
5. Compare weighted F1 with CLIP+MLP baseline
