# CoT4Rec

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

The processed datasets are in [`./dataIntegration/`](https://github.com/815382636/cot-rec/blob/main/dataIntegration/README.md)

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

## Citation

If the code and the paper are useful for you, it is appreciable to cite our paper:

```
@article{yue2025large,
  title={CoT4Rec: Unveiling User Preferences through Chain of Thought for Recommender Systems},
  author={Weiqi, Yue and Yuyu, Yin and Xin, Zhang and Binbin, Shi and Tingting, Liang and Jian, Wan},
  journal={proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Thanks

The code refers to the repo [Cbox4cr](https://github.com/liuqidong07/MOELoRA-peft](https://github.com/crazyfat/Cbox4cr), [auto-cot](https://github.com/amazon-science/auto-cot) and [mm-cot](https://github.com/amazon-science/mm-cot).
