create env

```bash
conda create -n mnist_mlops python=3.7 -y
```

activate env
```bash
conda activate mnist_mlops
```

install requirements
```bash
pip install -r requirements.txt
```

dvc init

dvc add data_given/mnist_test.csv