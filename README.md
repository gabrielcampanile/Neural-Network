# Artificial Neural Networks - UNIFESP

**Computer Science Course | 5th Semester**

## About the Repository

This repository contains the implementations of projects developed for the Artificial Neural Networks course in the Computer Science program at UNIFESP. The objective is to implement and study different neural network architectures learned in the course, from fundamental concepts to advanced applications.

Artificial neural networks are computational models inspired by the functioning of the nervous system, capable of learning complex patterns from data. They form the basis of various applications in artificial intelligence, including pattern recognition, natural language processing, and computer vision.

## Implemented Projects

### Project 0 - Perceptron

- **Description**: Implementation of the basic artificial neuron
- **Concepts**: Activation function, supervised learning, linear classification
- **Status**: Completed

### Project 1 - MLP (Multi-Layer Perceptron)

- **Description**: Neural network with multiple hidden layers
- **Concepts**: Backpropagation, gradient descent, non-linear activation functions
- **Status**: Completed

### Project 2 - SOM (Self-Organizing Maps)

- **Description**: Self-organizing maps for unsupervised learning
- **Concepts**: Vector quantization, clustering, data visualization
- **Status**: Completed

### Project 3 - VAE (Variational Autoencoders)

- **Description**: Variational autoencoders for data generation
- **Concepts**: Encoding-decoding, latent space, Bayesian inference
- **Status**: Completed

### Project 4 - RNN (Recurrent Neural Networks)

- **Description**: Recurrent neural networks for sequential data
- **Concepts**: Temporal memory, LSTM, sequence processing
- **Status**: Completed

### Project 5 - CNN (Convolutional Neural Networks)

- **Description**: Convolutional neural networks for image processing
- **Concepts**: Convolution, pooling, feature extraction
- **Status**: Completed

### Final Project - Neural Machine Translation

- **Description**: Implementation and comparison of neural translation models (English ↔ Portuguese)
- **Models Implemented**:
  - GRU-based Encoder-Decoder with Attention
  - Transformer Architecture (state-of-the-art)
- **Concepts**: Sequence-to-sequence learning, attention mechanisms, transformer architecture, BLEU evaluation
- **Dataset**: English-Portuguese parallel corpus
- **Objective**: Compare traditional RNN-based approaches with modern transformer models
- **Status**: GRU model completed, Transformer model in development

## Technologies Used

- **Python**: Main programming language
- **Jupyter Notebook**: Interactive development and documentation
- **PyTorch**: Deep learning framework for building and training neural networks
- **TensorFlow/Keras**: Alternative framework for neural network construction
- **Scikit-learn**: Machine learning tools and preprocessing
- **NumPy**: Numerical computing and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization and results plotting
- **NLTK/spaCy**: Natural language processing tools (Final Project)

## Repository Structure

```
├── Project 0 - Perceptron/
│   └── Projeto0.ipynb
├── Project 1 - MLP/
│   ├── Projeto_1.pdf
│   ├── Projeto1.ipynb
│   └── projeto1.py
├── Project 2 - SOM/
│   └── Projeto2.ipynb
├── Project 3 - VAE/
│   └── Projeto3.ipynb
├── Project 4 - RNN/
│   └── Projeto4.ipynb
├── Project 5 - CNN/
│   └── Projeto5.ipynb
└── Final Project - Translator/
    ├── GRU_translator_EN_PT.ipynb
    ├── data/
    │   └── por-eng.txt
    └── GRU - saved model/
        ├── gru_decoder_completo.pth
        ├── gru_decoder_pesos.pth
        ├── gru_encoder_completo.pth
        ├── gru_encoder_pesos.pth
        └── loss.png
```

## Learning Objectives

- Understand the mathematical foundations of neural networks
- Implement different neural network architectures
- Apply training and optimization techniques
- Analyze the performance of different models
- Explore practical applications in real-world problems
- Compare traditional and modern approaches in neural machine translation

## Author

Developed by Gabriel Belchior as part of studies in the Artificial Neural Networks course at UNIFESP.  
[LinkedIn](https://www.linkedin.com/in/gabriel-belchior-campanile/)
