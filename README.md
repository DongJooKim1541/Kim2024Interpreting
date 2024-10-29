# Interpreting Pretext Tasks for Active Learning: A Reinforcement Learning Approach - Official Pytorch Implementation
- This article was accepted into the [Scientific Reports](https://www.nature.com/srep/) journal on October 17, 2024.
- This article was published on October 28, 2024.

## Experiment Settings
- CIFAR10, CIFAR100 (saved in `./DATA`)
- [SVHN](http://ufldl.stanford.edu/housenumbers/), [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02), [ImageNet-64](https://www.image-net.org/download.php) is available on their respective links
- Pretext task: Rotation prediction task
- Please create `./checkpoint` and `./loss` dirs in the python project

## Prerequisites
```
Python >= 3.7
CUDA >= 11.0
PyTorch >= 1.7.1
numpy >= 1.16.0
```

## Running the Code
To generate train and test dataset for CIFAR10 and CIFAR100.
```
make_data.py
```
To train the rotation predition task on the unlabeled set.
```
rotation.py
```
To extract pretext task losses and create multi-armed-bandit groups(batches).
```
make_batches.py
```
To train and evaluate model for AL task.
```
main.py
```

## Citation
If you use this code in your research, or find our work helpful for your works, please citing us with the bibtex below
```
@article{kim2024interpreting,
  title = {Interpreting Pretext Tasks for Active Learning: A Reinforcement Learning Approach},
  author = {Kim, Dongjoo and Lee, Minsik},
  journal={Scientific Reports},
  volume={},
  number={},
  pages={},
  year = {2024},
  month={},
}
```