# MLOps Assignment 2 — DistilBERT for Goodreads Genre Classification

Fine-tuning `distilbert-base-cased` on UCSD Goodreads reviews for 8-class book genre classification. Training on Kaggle T4 GPU, metrics tracked on W&B, trained model published to Hugging Face Hub.

**Course:** MLOps · IIT Jodhpur
**Instructor:** Dr. Hardik Jain
**Author:** Rohit Patel

---

## Links

| | |
|---|---|
| Kaggle Notebook | https://www.kaggle.com/code/rohitpatel0008/mlops-vs-1-final-rohit |
| Hugging Face Model | https://huggingface.co/rpaut03l/distilbert-goodreads-genres |
| W&B Dashboard | https://wandb.ai/rohitspatel0008/mlops-assignment-2?nw=nwuserrohitpatel0008 |
| GitHub | https://github.com/rpaut03l/mlops-assignment2 |

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.5344 |
| F1 (weighted) | 0.5005 |
| Eval Loss | 2.88 |

8 genres, 320 reviews in held-out test set. Random baseline on 8 classes is 0.125 so 0.53 means the model is learning real signal.

---

## What this does

1. Downloads Goodreads reviews for 8 genres from UCSD (poetry, children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, romance, young_adult)
2. Samples 200 reviews per genre — 160 train, 40 test
3. Tokenizes with DistilBERT tokenizer (max_length 256)
4. Fine-tunes `distilbert-base-cased` for 3 epochs on Kaggle T4 GPU with fp16
5. Streams loss, accuracy, F1, GPU metrics live to W&B
6. Saves classification report as a versioned W&B Artifact
7. Pushes trained model + tokenizer to Hugging Face Hub
8. Writes HF model URL back into W&B run summary

---

## How to run

Need three free accounts: Kaggle, Hugging Face, W&B.

**Kaggle setup:**
1. Create > New Notebook
2. Settings > Accelerator > GPU T4 x2
3. Settings > Internet > On (default is Off, don't miss this)
4. Add-ons > Secrets:
   - `WANDB_API_KEY` from https://wandb.ai/authorize
   - `HF_TOKEN` from https://huggingface.co/settings/tokens (Write scope, not Read)
   - Tick "Attach to notebook" for both

Then File > Import Notebook > upload `mlops_assignment2.ipynb` from this repo. Run All. Around 10-15 minutes start to finish.

Change `HF_USERNAME` in the notebook to your own HF username before running.

---

## Hyperparameters

| Setting | Value |
|---|---|
| Model | distilbert-base-cased |
| Max length | 256 |
| Train batch | 16 |
| Eval batch | 32 |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Warmup steps | 50 |
| Weight decay | 0.01 |
| fp16 | Enabled |

---

## Using the trained model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("rpaut03l/distilbert-goodreads-genres")
model = AutoModelForSequenceClassification.from_pretrained("rpaut03l/distilbert-goodreads-genres")

inputs = tok("A gripping thriller with unexpected twists", return_tensors="pt")
preds = model(**inputs).logits.argmax(-1).item()
print(model.config.id2label[preds])
```

---

## Files

- `README.md` — this file
- `mlops_assignment2.ipynb` — the Kaggle notebook that ran the pipeline (downloaded from the public Kaggle link above)
- `requirements.txt` — Python dependencies
- `eval_report.json` — per-class classification report (also logged as W&B Artifact)
- `MLOps_Assignment2_Report_Rohit.pdf` — the report submitted on LMS

---

## Credits

- Dataset: UCSD Book Graph by Mengting Wan and Julian McAuley
- Starter notebook: Maria Antoniak, Melanie Walsh, AI for Humanists
- Base model: distilbert-base-cased by Hugging Face
- Course: Dr. Hardik Jain, IIT Jodhpur
