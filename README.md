# Unsupervised anomaly detection

A python project to find anomaly in an Xray image.

The aim is detect pneumonia in a thorax radiography and explain AI decision.

## Model

I use a `VGG16` model.

## M1 use

```bash
source ~/.zshrc
conda create -n tf_m1 python=3.11
conda activate tf_m1
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

## Training

In your conda env :

```bash
python train.py
```

## Results

![output](./media/output.png)
![output2](./media/output2.png)

## Tensorboard

**_Source :_**
[tensorboard-doc](https://www.tensorflow.org/tensorboard/get_started?hl=fr)

### For python notebook

```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```
