# Deep-Learning-Augmentation-and-Learning-Rate
Code to study data augmentation and the learning rate hyperparameter for image classification

![](img/galaxy_orig.png)

![](img/galaxy_augmentation1.png)

### Data Augmentation in Computer Vision

Data augmentation increases the size of a training dataset by applying various transformations to the original images to generate synthetic images based on a reference dataset. Data augmentation helps to create computer vision deep learning models that are more robust and reduces the risk of overfitting, particularly when the original dataset is small.

Data augmentation helps computer vision models along several dimensions:

**Combat Overfitting**: Overfitting occurs when a model learns the training data too well, including its noise and outliers, and performs poorly on unseen data. By augmenting the data, the model gets exposed to various variations of the data, making it generalize better to new, unseen examples.

**Increase Dataset Size**: For deep learning models, having a larger dataset is beneficial. However, collecting new data can be expensive or impractical. Augmentation is a way to synthetically expand the dataset.

**Improve Model Robustness**: By exposing the model to different variations of the data during training, the model becomes more robust to these variations during inference.

In the notebook we demonstrate two approaches to image data augmentation. These are on-the-fly augmentation and pre-computed augmentation:

**On-the-fly Augmentation**: During each epoch of training, images are read from disk and augmented in real-time before being fed into the model. This ensures that the model sees slightly different variations of images in each epoch. Libraries like TensorFlow and Keras provide utilities to do on-the-fly augmentation.

**Pre-computed Augmentation**: Transformations are applied to the entire dataset in advance, and the augmented images are saved to disk. The model then trains on this expanded dataset.

### Learning Rate Hyperparameter in Deep Learning

**Learning rate** is one of the most critical hyperparameters in the training of deep learning models, including the example here of training deep convolutional neural networks (CNNs) for image classification. The learning rate determines how much the model should adjust its weights in response to the calculated gradient from the loss function.

Some important considerations for learning rate values in training deep learning models:

**Step Size in Optimization**: The learning rate determines the magnitude of the steps taken in the weight space towards the minimum of the loss function. A larger learning rate results in bigger jumps, while a smaller learning rate results in smaller jumps.

**Convergence Speed**: A high learning rate might speed up convergence initially since it covers more ground in fewer steps.
However, if the learning rate is too high, the model can overshoot the optimal point in the loss landscape, potentially causing divergence, where the loss becomes unstable and increases.
On the other hand, a small learning rate ensures more careful and precise steps, but it may lead to slower convergence.
Local Minima & Saddle Points: A slightly higher learning rate can sometimes help the model jump out of local minima or avoid getting stuck in saddle points, which are common in high-dimensional spaces like those of deep neural networks.

**Generalization and Overfitting**: Very high learning rates can prevent a model from converging, which could result in underfitting.Very low learning rates, especially if not reduced over time, can cause the model to overfit, as the model might start fitting to the noise in the training data after already capturing the main patterns. 

**Learning Rate Schedules**: In practice, using a fixed learning rate throughout training may not be optimal. Adaptive learning rate schedules, like learning rate annealing, step decay, or using algorithms like Adam which adaptively adjust the learning rate, can help in converging faster and reaching better local optima.
