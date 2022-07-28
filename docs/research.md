# End-to-End Instance Segmentation with polygon predictions

## Papers to read

### Instance Segmentation with Transformers
-  Instance Segmentation Transformer  [https://arxiv.org/abs/2105.00637](arxiv:2105:00637)
-  Feature Pyramid Networks for Object Detection [https://arxiv.org/abs/1612.03144](arxiv:1612.03144)
-  

### Polygon Instance Segmentation
- PolyTransform [arXiv:1912.02801](https://arxiv.org/abs/1912.02801)
  
### Polygon Annotation
-  Annotating Object Instances with a Polygon-RNN	[arXiv:1704.05548](https://arxiv.org/abs/1704.05548)
-  Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++ [arXiv:1803.09693](https://arxiv.org/abs/1803.09693)
-  Fast Interactive Object Annotation with Curve-GCN [arxiv:1903.06874](https://arxiv.org/abs/1903.06874)

## Architecture Ideas

### UNet with MaxPooling/MaxUnpooling to predict vertex locations
- does not provide an ordering

### Transformer
- SOTA approach is to use CNN and FPN for image features (values) and ROI (queries)
    - usually use ResNet or ResNeXt for the CNN
    - FPN provides features at different scales
- PolyTransform Paper [arXiv:1912.02801](https://arxiv.org/abs/1912.02801)
    - achieves SOTA on CityScapes, predicting polygon rather than mask
    - relies on an instance segmentation network to initialize the polygons, then refines them based on image features
      - utilizes RCNN for object boundaries to initialize polygons
      - network acts as a refinement process on the predictions given by another instance segmentation network
    - **Would be interesting to see if Instance Segmentation Transformer  [https://arxiv.org/abs/2105.00637](arxiv:2105:00637) could be combined with this to produce a single model for polygon instance segmentation prediction**
- using a transformer to predict each subsequent vertex might be ill-suited as vertices should communicate a movement as described in PolyTransform
  - moving one vertex effects the two edges it is connected to, effecting the vertices next to it 

### RNN to mimic polygon annotation networks
- Graph Convolutional Network to iteratively "move" vertices closer to object boundaries
  - seed with ellipse wider than detected bounding box

## Issues with Predicting Vertices

Unlike predicting a mask where we are penalizing the error in just the value at a poisition, evaluating a criterion on 
a set of vertices may pose more problems. 

### Which order do we predict?

It is important to predict the correct ordering to maintain the shape of the polygon and evaluate the loss.

### Where do we start?

- Predefined?
- Closest to some region?
- Does it matter?

### How do we take the loss of something that can have multiple valid orderings?

- Average out loss for all valid orderings?
  - Perhaps minimum / maximum instead of average?
  - Could be a "smoother" approach but may penalize good