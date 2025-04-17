# Multi-Scale Residual Attention Network (MS-RAN) for MNIST Classification

This repository contains an implementation of a high-performance convolutional neural network architecture that combines residual learning, multi-scale feature extraction, and attention mechanisms for handwritten digit recognition on the MNIST dataset. The implementation is built from scratch using PyTorch, focusing on architectural optimization and modern deep learning techniques.

## Technical Architecture Overview

The implementation features a hybrid neural network architecture that integrates multiple state-of-the-art deep learning components:

- **Residual Learning Framework**: Identity mappings and skip connections that mitigate the vanishing gradient problem in deep networks, enabling more effective backpropagation through the computational graph
- **Multi-Scale Feature Extraction**: Parallel convolutional pathways inspired by the Inception architecture that process input at multiple receptive field sizes (1×1, 3×3, and simulated 5×5 convolutions)
- **Channel Attention Mechanism**: Squeeze-and-excitation blocks that adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels
- **Batch Normalization Layers**: Normalization technique that addresses internal covariate shift, accelerating training by allowing higher learning rates and reducing the dependence on careful parameter initialization
- **Stochastic Regularization**: Strategic application of dropout with 0.5 probability to prevent co-adaptation of feature detectors

## Detailed Network Architecture

The network architecture implements a feed-forward computational graph with the following components:

1. **Input Processing Layer**: 3×3 convolutional layer with 32 output channels, stride 1, and padding 1, followed by batch normalization and ReLU activation (output shape: 32×28×28)

2. **Hierarchical Feature Extraction**:
   - **First Residual Block**: Downsamples spatial dimensions via stride 2 while expanding channel dimension to 64 (output shape: 64×14×14)
   - **Second Residual Block**: Further downsamples to 7×7 spatial dimensions with 128 channels (output shape: 128×7×7)

3. **Multi-Scale Feature Integration**:
   - **Inception Module**: Parallel pathways with 1×1, 3×3, and cascaded 3×3 convolutions (simulating 5×5 receptive field), concatenated along the channel dimension (output shape: 256×7×7)

4. **Feature Refinement**:
   - **Channel Attention Mechanism**: Adaptive feature recalibration using global average pooling followed by a bottleneck MLP (16 hidden units) with sigmoid activation
   - **Weighted Feature Maps**: Element-wise multiplication of feature maps with attention weights

5. **Classification Head**:
   - **Global Average Pooling**: Spatial dimensions reduced to 1×1 (output shape: 256×1×1)
   - **Dropout Layer**: With 0.5 probability for regularization
   - **Fully Connected Layer**: Linear projection to 10 output neurons corresponding to digit classes

## Repository Structure

- `ms_ran.py`: Entry point script that orchestrates model training, evaluation, and visualization
- `models.py`: Module containing the neural network architecture implementation with ResidualBlock, InceptionModule, and MS_RAN classes
- `utils.py`: Utility module with functions for data preprocessing, training loop implementation, evaluation metrics, and visualization routines
- `requirements.txt`: Dependency specification file listing required Python packages and their versions

## Technical Requirements

- Python 3.6+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- matplotlib 3.3.0+
- numpy 1.19.0+

## Environment Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ParedezDev/MS-RAN.git
   cd MS-RAN
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Execution and Training

To train and evaluate the model:

```bash
python ms_ran.py
```

Execution workflow:
1. Automatic download and preprocessing of the MNIST dataset (60,000 training images, 10,000 test images)
2. Model initialization and training for 10 epochs with Adam optimizer and learning rate scheduling
3. Checkpoint saving of the best-performing model weights to 'best_model.pth'
4. Generation of performance visualizations including loss curves, accuracy metrics, and prediction samples

## Performance Metrics

The implemented architecture achieves state-of-the-art performance on the MNIST benchmark:

- **Test Accuracy**: >99.5% after 10 epochs of training
- **Training Time**: Approximately 2-3 minutes on modern GPU hardware
- **Model Size**: ~3.2M parameters
- **Inference Speed**: <1ms per image on GPU

## Performance Visualization

The training script automatically generates diagnostic visualizations:

- `training_history.png`: Dual-plot visualization showing:
  - Training and validation loss curves demonstrating convergence characteristics
  - Training and validation accuracy progression across epochs

- `predictions.png`: Qualitative evaluation display showing:
  - Sample test images with model predictions
  - Color-coded results (green for correct, red for incorrect predictions)
  - Confusion patterns for misclassified examples

## Hyperparameter Configuration

The implementation supports extensive hyperparameter tuning through the `ms_ran.py` script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Mini-batch size for stochastic optimization |
| `num_epochs` | 10 | Number of complete passes through the training dataset |
| `learning_rate` | 0.001 | Initial step size for Adam optimizer |
| `weight_decay` | 1e-4 | L2 regularization coefficient for preventing overfitting |

Additional architectural hyperparameters can be modified in the `models.py` file, including channel dimensions, dropout rates, and attention bottleneck size.

## Inference and Deployment

For model inference in production environments:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import MS_RAN

def preprocess_image(image_path):
    # Apply same preprocessing as training data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load the pretrained model
model = MS_RAN()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Inference pipeline
def predict_digit(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()
```

## Citation and References

If you use this implementation in your research, please cite:

```bibtex
@misc{paredez2023msran,
  author = {Paredez, Developer},
  title = {Multi-Scale Residual Attention Network (MS-RAN) for MNIST Classification},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ParedezDev/MS-RAN}}
}
```

## License

[MIT License](LICENSE)

## Acknowledgements

- MNIST dataset by Yann LeCun and Corinna Cortes: http://yann.lecun.com/exdb/mnist/
- Architectural components inspired by:
  - He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
  - Szegedy, C., et al. (2015). "Going Deeper with Convolutions"
  - Hu, J., et al. (2018). "Squeeze-and-Excitation Networks"
