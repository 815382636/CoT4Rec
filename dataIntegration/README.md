# Template Construction

## Datasets

**MovieLens Datasets**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/).

**Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/).

## File description

|        file        |            description             |
| :----------------: | :--------------------------------: |
|   \*/process1.py   |           Build template           |
|   \*/process2.py   |         Add candidate set          |
|     prompt.py      |          Invoke the GPT-4          |
|       run.py       | Call LLM to build user preferences |
| cluster_process.py | Build CoT prefix using clustering  |

## Example to run codes

```
#  ML-100k dataset
> cd data-integration
> python run.py --dataset ml100k --cot cluster
```
