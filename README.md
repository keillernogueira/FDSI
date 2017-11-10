# Flood Detection in Satellite Images [FDSI]

![alt text](divulg.png "Examples of obtained results")

Natural disaster monitoring is a fundamental task to create prevention strategies, as well as to help authorities to act in the control of damages, coordinate rescues, and help victims.
Among all natural disasters, flooding is possibly the most extensive and devastating one being considered as the world's most costly type of natural hazard in terms of both economic losses and human causalities.
Although extremely important, floods are difficult to monitor, because they are highly dependent on several local conditions, such as precipitation, drainage network, and land cover.
A first and essential step towards such monitoring is based on identifying areas most vulnerable to flooding, helping authorities to focus on such regions while monitoring inundations.
In this work, we tackled such task using distinct strategies all based on ConvNets.
Specifically, we proposed novel ConvNet architectures specialized in identifying flooding areas as well as a new strategy focuses on exploiting network diversity of these ConvNets for inundation identification.
This work was considered the winner of the Flood-Detection in Satellite Images, a subtask of [2017 Multimedia Satellite Task](http://www.multimediaeval.org/mediaeval2017/multimediasatellite/), which was part of the traditional MediaEval Benchmark.

It allows training of several ConvNets (using TensorFlow framework) for semantic segmentation of remote sensing images.
Among the networks, it includes:

  - Two architectures based uniquely on [Dilated Convolutions](https://arxiv.org/abs/1511.07122) (the best in our experiments)
  - Two architectures based on [SegNets](https://arxiv.org/abs/1505.07293)
  - Two architectures based on [Deconvolutonal Networks](https://arxiv.org/abs/1409.1556)

There is also a code of the meta-learning combination performed to combine all network outcomes using Linear SVM.
