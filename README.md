# biaffineparser
Deep Biaffine Parser implementation as in https://arxiv.org/abs/1611.01734
Since I have no access to CoNLL09 data or the Penn Treebank data, I trained it on the publicly available CoNLL17 dataset. Here I uploaded a csv version converted using https://github.com/interrogator/conll-df

# Installing:
To correctly clone the project install git lfs first https://git-lfs.github.com/

# Run Training
Run the training launching
```python
python -m parser.train
```

At the moment Hyperparameters information is in the code.
