## Cross-Stage Popularity Guidance to Debias LLM-based Recommendation

## Abstract
To adapt large language models (LLMs) for recommendation tasks, Supervised Fine-Tuning (SFT) has become the dominant paradigm, often combined with Direct Preference Optimization (DPO) to model user preferences, with final recommendations generated during inference. However, each stage inevitably introduces popularity bias. Recent debiasing methods for LLM-based recommendation systems (LRSs) intervene at different stages, while some rely on external models or multi-round iterative procedures, increasing the complexity of the debiasing pipeline. Moreover, due to differences in learning paradigms across SFT, DPO, and inference, coordinating popularity debiasing across these stages remains insufficiently explored.
To address these issues, we propose CPGLRec, a cross-stage popularity guidance framework for debiasing LLM-based recommendation. CPGLRec constructs a unified popularity signal by moderately smoothing item popularity, providing consistent guidance throughout the recommendation pipeline.
(1) In the SFT stage, the model is guided to balance recommendation proportions across different popularity groups through regularization constraints.
(2) In the DPO stage, the over-amplification of popularity differences is mitigated through popularity-aware preference optimization.
(3) In the inference stage, the ranking of LLM-generated items is adjusted through popularity-calibrated reranking.
Experiments on three real-world datasets show that CPGLRec substantially improves recommendation fairness and diversity while maintaining competitive accuracy.

## How to Train Using CPGLRec Framework
```bash
python mydata/MovieLens1M/process.py
python get_item_embeddings.py
bash run_stage1.sh
bash run_dpo_data.sh
bash run_stage2.sh
bash run_inference.sh
bash run_evaluate.sh
```

## Prepare the pre-trained Huggingface model of Qwen2.5-1.5B-Instruct:
Qwen2.5-1.5B-Instruct
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

## Download the datasets
MovieLens 1M
https://grouplens.org/datasets/movielens/1m/

Beauty
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

CDs and Vinyl
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
