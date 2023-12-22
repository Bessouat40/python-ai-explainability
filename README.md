# Unsupervised anomaly detection

A python project to find anomaly in an Xray image.

The aim is detect pneumonia in a thorax radiography and explain AI decision.

## Model

I use a `VGG16` model.

## Results

![output](./media/output.png)

## Tensorboard

**_Source :_**
[tensorboard-doc](https://www.tensorflow.org/tensorboard/get_started?hl=fr)

### For python notebook

```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

## Improvement

I will try to use data augmentation and fine tuned model.
