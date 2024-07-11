# cot-rec

CoT4Rec: Unveiling User Preferences through Chain of Thought for Recommender Systems

## Requirements

```
python==3.9
dashscope==1.14.1
openai==1.34.0
huggingface-hub>=0.4.0
numpy==1.23.2
pandas==1.4.3
rouge==1.0.1
sentence-transformers==2.2.2
transformers==4.30.0
nltk==3.6.6
evaluate==0.4.0
rouge_score==0.1.2
rich>=13.3.2
```

## Dataset Construction

The processed datasets are in [`./dataIntegration/`](https://github.com/815382636/cot-rec/blob/main/data-integration/README.md)

## Quick Start

### Training

```
# user preference generation (example for ml100k)
bash rec-ml-100k-rationale.sh

# recommendation
bash rec-ml-100k-answer.sh
```

### Inference

```
bash rec-ml-100k-test.sh
```
