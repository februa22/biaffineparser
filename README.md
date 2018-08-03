# biaffineparser
Deep Biaffine Parser implementation as in https://arxiv.org/abs/1611.01734
Since I have no access to CoNLL09 data or the Penn Treebank data, I trained it on the publicly available CoNLL17 dataset. Here I uploaded a csv version converted using https://github.com/interrogator/conll-df

# Installing:
Here's how to properly set up the project:
* To correctly clone the project install git lfs first https://git-lfs.github.com/
* Install pytorch 0.4 https://pytorch.org/
* Install requirements.txt by running `pip install -r requirements.txt`

# Run Training
Run the training launching
```python
python -m parser.train
```

At the moment Hyperparameters information is in the code.
