### Knowledge Distillation (experimental feature)

#### Transferring knowledge
Knowledge Distillation is method of model compression in which small model (student) is trained to mimic a pretrained
 large model (teacher) through knowledge transferring. Modern method was proposed
  by [Hinton et al., 2015](https://arxiv.org/pdf/1503.02531.pdf).
  
Knowledge is transferred from the teacher model to the student by minimizing loss function in which the target is the
predictions of the teacher model. At the moment we present two types of loss functions and you must explicitly
 choose one of them in the config.
 
 MSE distillation loss:
 
 ![{L}_{MSE}(z^{s}, z^{t}) = || z^s - z^t ||_2^2](https://latex.codecogs.com/png.latex?{L}_{MSE}(z^{s},%20z^{t})%20=%20||%20z^s%20-%20z^t%20||_2^2)
 
 Cross-Entropy distillation loss:
 
 ![{p}_{i}=\frac{\exp({z}_{i})}{\sum_{j}(\exp({z}_{j}))}](https://latex.codecogs.com/png.latex?{p}_{i}=%20\frac{\exp({z}_{i})}{\sum_{j}(\exp({z}_{j}))})
 
 ![{L}_{CE}({p}^{s}, {p}^{t}) = -\sum_{i}{p}^{t}_{i}*\log({p}^{s}_{i})](https://latex.codecogs.com/png.latex?{L}_{CE}({p}^{s},%20{p}^{t})%20=%20-\sum_{i}{p}^{t}_{i}*\log({p}^{s}_{i}))
 
 Knowledge Distillation loss function in total is combined with regular loss function, so overall loss function will be
  computed as:
  
 ![L = {L}_{reg}({z}^{s}, y) + {L}_{distill}({z}^{s}, {z}^{t})](https://latex.codecogs.com/png.latex?L%20=%20{L}_{reg}({z}^{s},%20y)%20+%20{L}_{distill}({z}^{s},%20{z}^{t}))
  
 ![kd_pic](../pics/knowledge_distillation.png)
  
  Note: At [Hinton et al., 2015](https://arxiv.org/pdf/1503.02531.pdf) Cross-Entropy distillation loss was proposed with
  temperature parameter but we don't use it or assume that T=1
  
#### User guide

Knowledge distillation can be combined with any other compression algorithm (e.g. pruning or quantization) for
 compressed model accuracy improvement.

Tested with classification, object detection and semantic segmentation tasks presented as examples. Models that has
 been tested: ResNet34, GoogleNet, MobileNetv2, SSD300_mobilenet, ICNet.
 
##### Limitations

At the current moment algorithm working only of PyTorch NNCF.

Training the same configuration with Knowledge Distillation requires more time and GPU memory
 then without Knowledge Distillation. On average, memory (for all GPU execution modes) and time overhead is below 20% each.

Outputs of model that shouldn't be differentiated must have `requires_grad=False`.

##### How to turn on

Knowledge Distillation is a `CompressionAlgorithm` so you should add to config the one of the following schemes:
```
{
    "algorithm": "knowledge_distillation",
    "type": "softmax"
}
```
```
{
    "algorithm": "knowledge_distillation",
    "type": "mse"
}
```
Example of a config: [example](../../examples/torch/classification/configs/pruning/resnet34_pruning_geometric_median_kd.json)