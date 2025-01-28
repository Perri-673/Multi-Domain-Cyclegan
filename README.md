# 420 GANs : Automating Sketch-to-Image Transformation Using CycleGANs
A clean, simple and readable implementation of CycleGAN in PyTorch. We've tried to replicate the original paper as closely as possible, so if you read the paper the implementation should be pretty much identical. The results from this implementation I would say is on par with the paper, I have included some examples results below.

## Results
The model was trained on Sketch<->Photo dataset for 400 epochs.

| Input | Generated | Actual |
|:---:|:---:|:---:|
| ![](real_sketch_0.png) | ![](sketch_to_photo_0.png) | ![](original_photo.jpg) |
| ![](real_photo_0.png) | ![](photo_to_sketch_0.png) | ![](original_sketch.jpg) |



### Sketch and Photo Dataset
TRAIN: 88
VAL : 100

### Training
| Parameters |
|:---:|
| BATCH_SIZE = 1 |
| LEARNING_RATE = 1e-5 |
| LAMBDA_CYCLE = 10 |
| NUM_WORKERS = 4 |
| NUM_EPOCHS = 400 |
| OPTIMIZER = ADAM |
| LOSS FN = L1-Loss & MSE |

### Inference
Colab Notebook link: https://colab.research.google.com/drive/1Q59QB5-req2mpGIl1pB-v1Of0iZy5zbp?usp=sharing
## CycleGAN paper
### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

#### Abstract
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach. 
```
@misc{zhu2020unpaired,
      title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks}, 
      author={Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros},
      year={2020},
      eprint={1703.10593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
