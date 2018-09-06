# biaffineparser

Deep Biaffine Parser implementation as in https://arxiv.org/abs/1611.01734

## Installing

Here's how to properly set up the project:

- Install TensorFlow >= 1.10.0 https://www.tensorflow.org/install/
- Install requirements.txt by running `pip install -r requirements.txt`

## Run Training

Run the script for training:

```shell
./train.sh
```

## Results

세종 데이터셋에 대한 정확도 성능

No. | Systems | 어절 표현 | Embeddings | UAS | LAS
--- | --- | --- | --- | :---: | :---:
1 | Biaffine | 평균 | Pre-trained on train-set | 87.40 | 82.82
2 | Biaffine | 평균 | Random | 89.01 | 85.96
