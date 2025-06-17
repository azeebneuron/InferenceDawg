![ChatGPT Image Jun 17, 2025, 06_34_58 PM](https://github.com/user-attachments/assets/7e6f25d6-214a-4be6-a2e9-22feda1d3e97)

> Let's see how far I go

## Progress Dashboard

| Category | Status | Completion | Modules |
|----------|--------|------------|---------|
| **Revisiting the Basics** | Started | █░░░░░░░░░░░░░░░░░░░░░░░░ | 0 / 10 |
| **The Art of Model Miniaturization** | Not Started | ░░░░░░░░░░░░░░░░░░░░░░░░░ | 0 / 6 |
| **The Science of Serving** | Not Started | ░░░░░░░░░░░░░░░░░░░░░░░░░ | 0 / 2 |
| **Total Projects Completed** | Not Started | ░░░░░░░░░░░░░░░░░░░░░░░░░ | **0 / 8** |

## Revisiting the Basics

Build rock-solid foundations by implementing core ML/DL concepts from first principles before diving into advanced optimization techniques.

### Module 0.1: Core Machine Learning Fundamentals

**Mathematical foundations and classical algorithms implemented from scratch**

#### Core Concepts Checklist

- [ ] **Linear Algebra Essentials:**
  - [ ] Vector operations, dot products, norms, and orthogonality
  - [ ] Matrix multiplication, transpose, inverse, and eigendecomposition
  - [ ] Principal Component Analysis (PCA) for dimensionality reduction
- [ ] **Probability and Statistics:**
  - [ ] Bayes' theorem and conditional probability
  - [ ] Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)
  - [ ] Gaussian distributions and Central Limit Theorem
- [ ] **Optimization Fundamentals:**
  - [ ] Gradient descent variants: batch, mini-batch, stochastic
  - [ ] Learning rate scheduling and momentum
  - [ ] Convex vs. non-convex optimization landscapes

#### Knowledge Checks

- [ ] **Exercise 1:** Implement PCA from scratch using eigendecomposition, apply to 2D dataset visualization
- [ ] **Exercise 2:** Derive and implement gradient descent for linear regression with mathematical proof
- [ ] **Exercise 3:** Code Naive Bayes classifier from scratch, compare with sklearn implementation
- [ ] **Exercise 4:** Implement K-means clustering with different initialization strategies
- [ ] **Exercise 5:** Build logistic regression with different regularization techniques (L1, L2, Elastic Net)

#### Capstone Project: ML Algorithm Library

- [ ] **Objective:** Create comprehensive ML library with 5+ algorithms implemented from mathematical foundations
- [ ] **Step 1:** Implement Linear/Logistic Regression with gradient descent optimization
- [ ] **Step 2:** Build K-Means and Gaussian Mixture Models for clustering
- [ ] **Step 3:** Create Decision Tree with entropy/gini splitting criteria
- [ ] **Step 4:** Implement Support Vector Machine with different kernels
- [ ] **Step 5:** Add cross-validation and hyperparameter tuning utilities
- [ ] **Step 6:** Benchmark against scikit-learn on standard datasets (Iris, Wine, Breast Cancer)
- [ ] **Step 7:** Create comprehensive documentation with mathematical derivations

---

### Module 0.2: Deep Learning Foundations

**Neural network fundamentals with manual backpropagation implementation**

#### Core Concepts Checklist

- [ ] **Neural Network Architecture:**
  - [ ] Perceptron and Multi-Layer Perceptron (MLP) structure
  - [ ] Activation functions: ReLU, Sigmoid, Tanh, Leaky ReLU, GELU
  - [ ] Universal approximation theorem and representation power
- [ ] **Backpropagation Algorithm:**
  - [ ] Forward pass: input → hidden → output computation
  - [ ] Backward pass: error propagation and gradient computation
  - [ ] Chain rule application for multi-layer networks
  - [ ] Weight update rules and parameter optimization
- [ ] **Training Dynamics:**
  - [ ] Loss functions: MSE, Cross-Entropy, Hinge Loss
  - [ ] Batch normalization and layer normalization
  - [ ] Dropout and other regularization techniques
  - [ ] Gradient clipping and exploding/vanishing gradients

#### Knowledge Checks

- [ ] **Exercise 1:** Manually compute forward and backward pass for 2-layer network on XOR problem
- [ ] **Exercise 2:** Implement different activation functions and analyze their gradient properties
- [ ] **Exercise 3:** Code batch normalization from scratch and explain internal covariate shift
- [ ] **Exercise 4:** Build dropout implementation and demonstrate regularization effect
- [ ] **Exercise 5:** Compare different optimizers (SGD, Adam, RMSprop) on same network architecture

#### Capstone Project: Neural Network Framework

- [ ] **Objective:** Build mini deep learning framework supporting arbitrary network architectures
- [ ] **Step 1:** Create base `Layer` class with forward/backward abstract methods
- [ ] **Step 2:** Implement `LinearLayer`, `ActivationLayer`, `DropoutLayer`, `BatchNormLayer`
- [ ] **Step 3:** Build `Network` class for layer composition and training loops
- [ ] **Step 4:** Add different optimizers (SGD, Adam) and loss functions
- [ ] **Step 5:** Test on MNIST digit classification with 3+ layer architectures
- [ ] **Step 6:** Compare performance with PyTorch equivalent implementation
- [ ] **Step 7:** Add learning curves visualization and hyperparameter sensitivity analysis

---

### Module 0.3: Convolutional Neural Networks (CNNs)

**Computer vision architectures and spatial feature learning**

#### Core Concepts Checklist

- [ ] **Convolution Operation:**
  - [ ] 2D convolution mechanics: kernels, stride, padding, dilation
  - [ ] Feature map computation and spatial dimension calculations
  - [ ] Convolution vs. correlation mathematical distinction
- [ ] **CNN Architecture Components:**
  - [ ] Convolutional layers and learnable filter banks
  - [ ] Pooling layers: Max, Average, Global Average Pooling
  - [ ] Fully connected layers for classification heads
- [ ] **Classic CNN Architectures:**
  - [ ] **LeNet-5:** MNIST handwritten digit recognition pioneer
  - [ ] **AlexNet:** ImageNet breakthrough with ReLU and dropout
  - [ ] **VGG:** Deep networks with small 3x3 filters
  - [ ] **ResNet:** Skip connections solving vanishing gradient problem
- [ ] **Advanced CNN Concepts:**
  - [ ] Receptive field calculation and effective receptive field
  - [ ] Depthwise separable convolutions (MobileNet inspiration)
  - [ ] Dilated/Atrous convolutions for dense prediction tasks

#### Knowledge Checks

- [ ] **Exercise 1:** Implement 2D convolution from scratch with different padding/stride configurations
- [ ] **Exercise 2:** Calculate receptive field for 5-layer CNN with given kernel sizes and strides
- [ ] **Exercise 3:** Implement max pooling and average pooling operations manually
- [ ] **Exercise 4:** Code basic ResNet block with skip connection and analyze gradient flow
- [ ] **Exercise 5:** Build depthwise separable convolution and compare parameter count with standard conv

#### Capstone Project: CNN Architecture Zoo

- [ ] **Objective:** Implement and compare classic CNN architectures on CIFAR-10 dataset
- [ ] **Step 1:** Build LeNet-5 architecture and train on CIFAR-10 (baseline)
- [ ] **Step 2:** Implement AlexNet-style architecture with ReLU, dropout, and data augmentation
- [ ] **Step 3:** Create VGG-like network with deep small-filter design
- [ ] **Step 4:** Build ResNet-18 with proper skip connections and batch normalization
- [ ] **Step 5:** Train all architectures with identical hyperparameters and data preprocessing
- [ ] **Step 6:** Compare accuracy, parameter count, training time, and convergence curves
- [ ] **Step 7:** Analyze learned filters and feature maps using visualization techniques

---

### Module 0.4: Recurrent Neural Networks and Sequence Modeling

**Temporal pattern recognition and sequence-to-sequence learning**

#### Core Concepts Checklist

- [ ] **RNN Fundamentals:**
  - [ ] Vanilla RNN architecture and hidden state computation
  - [ ] Backpropagation Through Time (BPTT) algorithm
  - [ ] Vanishing gradient problem in long sequences
- [ ] **Advanced RNN Architectures:**
  - [ ] **LSTM:** Long Short-Term Memory with gating mechanisms
  - [ ] **GRU:** Gated Recurrent Unit as simplified LSTM alternative
  - [ ] Bidirectional RNNs for utilizing future context
- [ ] **Sequence Learning Tasks:**
  - [ ] **Many-to-One:** Sequence classification (sentiment analysis)
  - [ ] **One-to-Many:** Sequence generation (image captioning)
  - [ ] **Many-to-Many:** Sequence-to-sequence (machine translation)
- [ ] **Attention Mechanism:**
  - [ ] Motivation: fixed-size bottleneck problem in encoder-decoder
  - [ ] Attention weights computation and context vector creation
  - [ ] Different attention variants: additive, multiplicative, self-attention

#### Knowledge Checks

- [ ] **Exercise 1:** Implement vanilla RNN from scratch and demonstrate BPTT on simple sequence task
- [ ] **Exercise 2:** Build LSTM cell with forget, input, and output gates step-by-step
- [ ] **Exercise 3:** Code GRU architecture and compare parameter efficiency with LSTM
- [ ] **Exercise 4:** Implement basic attention mechanism for sequence-to-sequence model
- [ ] **Exercise 5:** Compare RNN variants on sequence length sensitivity using synthetic data

#### Capstone Project: Language Modeling with RNNs

- [ ] **Objective:** Build character-level language model using different RNN architectures
- [ ] **Step 1:** Prepare character-level dataset from text corpus (Shakespeare, Wikipedia)
- [ ] **Step 2:** Implement vanilla RNN for character prediction with teacher forcing
- [ ] **Step 3:** Build LSTM-based language model with same architecture size
- [ ] **Step 4:** Create GRU variant and attention-enhanced LSTM model
- [ ] **Step 5:** Train all models with identical hyperparameters and sequence lengths
- [ ] **Step 6:** Evaluate perplexity and generate sample text from each model
- [ ] **Step 7:** Compare training stability, convergence speed, and generation quality

---

### Module 0.5: Transformer Architecture Deep Dive

**Attention mechanisms and the foundation of modern language models**

#### Core Concepts Checklist

- [ ] **Self-Attention Mechanism:**
  - [ ] Query, Key, Value matrices and their roles
  - [ ] Scaled dot-product attention computation
  - [ ] Attention matrix interpretation and visualization
- [ ] **Multi-Head Attention:**
  - [ ] Parallel attention heads with different learned representations
  - [ ] Head concatenation and output projection
  - [ ] Relationship to ensemble methods and representational diversity
- [ ] **Transformer Block Components:**
  - [ ] **Layer Normalization:** Pre-norm vs. post-norm configurations
  - [ ] **Position-wise Feed-Forward Networks:** Two-layer MLP with ReLU/GELU
  - [ ] **Residual Connections:** Skip connections around attention and FFN
- [ ] **Positional Encoding:**
  - [ ] Sinusoidal encoding for absolute positions
  - [ ] Learned positional embeddings
  - [ ] Relative position encoding variants
- [ ] **Encoder vs. Decoder Architecture:**
  - [ ] **Encoder-only:** BERT-style bidirectional models
  - [ ] **Decoder-only:** GPT-style autoregressive models
  - [ ] **Encoder-Decoder:** T5-style sequence-to-sequence models

#### Knowledge Checks

- [ ] **Exercise 1:** Implement scaled dot-product attention from scratch with proper masking
- [ ] **Exercise 2:** Build multi-head attention mechanism and visualize attention patterns
- [ ] **Exercise 3:** Code sinusoidal positional encoding and compare with learned embeddings
- [ ] **Exercise 4:** Implement complete transformer block with layer norm and residual connections
- [ ] **Exercise 5:** Compare causal (decoder) vs. bidirectional (encoder) attention masks

#### Capstone Project: Transformer Variants Implementation

- [ ] **Objective:** Build three transformer variants and compare on language understanding tasks
- [ ] **Step 1:** Implement encoder-only transformer (BERT-style) for sequence classification
- [ ] **Step 2:** Build decoder-only transformer (GPT-style) for language generation
- [ ] **Step 3:** Create encoder-decoder transformer for sequence-to-sequence tasks
- [ ] **Step 4:** Train encoder-only model on sentiment analysis (IMDB, SST-2)
- [ ] **Step 5:** Train decoder-only model on character-level language modeling
- [ ] **Step 6:** Train encoder-decoder model on simple translation task
- [ ] **Step 7:** Analyze attention patterns, compare parameter efficiency, and evaluate task performance

---

### Module 0.6: Advanced Deep Learning Concepts

**Modern techniques essential for contemporary deep learning practice**

#### Core Concepts Checklist

- [ ] **Normalization Techniques:**
  - [ ] **Batch Normalization:** Training vs. inference mode differences
  - [ ] **Layer Normalization:** Per-sample normalization for variable-length sequences
  - [ ] **Group Normalization:** Channel grouping for small batch sizes
  - [ ] **Instance Normalization:** Per-channel normalization for style transfer
- [ ] **Advanced Optimizers:**
  - [ ] **Adam:** Adaptive moment estimation with bias correction
  - [ ] **AdamW:** Weight decay vs. L2 regularization distinction
  - [ ] **Learning Rate Scheduling:** Cosine annealing, warm restarts, polynomial decay
- [ ] **Regularization Strategies:**
  - [ ] **Dropout variants:** Standard, DropConnect, Scheduled DropPath
  - [ ] **Data Augmentation:** Geometric, color, mixup, cutmix techniques
  - [ ] **Weight regularization:** L1, L2, and elastic net penalties
- [ ] **Loss Function Design:**
  - [ ] **Classification:** Cross-entropy, focal loss, label smoothing
  - [ ] **Regression:** MSE, MAE, Huber loss robustness properties
  - [ ] **Contrastive Learning:** InfoNCE, triplet loss, margin-based losses

#### Knowledge Checks

- [ ] **Exercise 1:** Implement all normalization techniques and compare training dynamics
- [ ] **Exercise 2:** Code Adam optimizer from scratch with proper bias correction
- [ ] **Exercise 3:** Build learning rate scheduler with cosine annealing and warm restarts
- [ ] **Exercise 4:** Implement focal loss and demonstrate class imbalance handling
- [ ] **Exercise 5:** Create mixup data augmentation and analyze regularization effects

#### Capstone Project: Training Optimization Laboratory

- [ ] **Objective:** Systematically study training optimization techniques on vision classification
- [ ] **Step 1:** Define baseline CNN architecture and CIFAR-10 training setup
- [ ] **Step 2:** Compare normalization techniques (Batch, Layer, Group) on same architecture
- [ ] **Step 3:** Evaluate optimizers (SGD, Adam, AdamW) with different learning rate schedules
- [ ] **Step 4:** Test regularization combinations (dropout, weight decay, data augmentation)
- [ ] **Step 5:** Implement advanced techniques (label smoothing, mixup, cutmix)
- [ ] **Step 6:** Create comprehensive comparison with statistical significance testing
- [ ] **Step 7:** Document best practices and optimization recipe for reproducible results

---

### Module 0.7: Implementing GPT from Scratch (Andrej Karpathy Neural Networks: Zero to Hero Series)

**Complete implementation journey following the legendary tutorial series**

#### Core Concepts Checklist

- [ ] **Lecture 1: The spelled-out intro to neural networks and backpropagation: building micrograd**
  - [ ] Automatic differentiation and computational graphs
  - [ ] Building backpropagation engine from scratch
  - [ ] Value class with gradient tracking and chain rule implementation
- [ ] **Lecture 2: The spelled-out intro to language modeling: building makemore**
  - [ ] Character-level language modeling fundamentals
  - [ ] N-gram models and their limitations
  - [ ] Transition to neural language models
- [ ] **Lecture 3: Building makemore Part 2: MLP**
  - [ ] Multi-layer perceptron for character prediction
  - [ ] Embedding layers and distributed representations
  - [ ] Training dynamics and hyperparameter sensitivity
- [ ] **Lecture 4: Building makemore Part 3: Activations & Gradients, BatchNorm**
  - [ ] Activation function analysis and gradient flow
  - [ ] Batch normalization implementation and benefits
  - [ ] Initialization strategies (Xavier, He, etc.)
- [ ] **Lecture 5: Building makemore Part 4: Becoming a Backprop Ninja**
  - [ ] Manual backpropagation through complex operations
  - [ ] Debugging gradient computation and numerical verification
  - [ ] Cross-entropy loss derivation and implementation
- [ ] **Lecture 6: Building makemore Part 5: Building a WaveNet**
  - [ ] Dilated causal convolutions for sequence modeling
  - [ ] Hierarchical feature learning in WaveNet architecture
  - [ ] Comparison with RNN-based approaches
- [ ] **Lecture 7: Let's build GPT: from scratch, in code, spelled out**
  - [ ] Transformer decoder architecture implementation
  - [ ] Self-attention mechanism with causal masking
  - [ ] Position embeddings and layer stacking
- [ ] **Lecture 8: Let's build the GPT Tokenizer**
  - [ ] Byte Pair Encoding (BPE) algorithm implementation
  - [ ] Tokenization strategies and vocabulary construction
  - [ ] Handling out-of-vocabulary tokens and special symbols
- [ ] **Lecture 9: Let's reproduce GPT-2 (124M)**
  - [ ] Exact GPT-2 architecture replication
  - [ ] Loading pre-trained weights and verification
  - [ ] Training from scratch vs. fine-tuning strategies
- [ ] **Lecture 10: Let's build the GPT Tokenizer**
  - [ ] Advanced tokenization techniques
  - [ ] SentencePiece and other modern tokenizers
  - [ ] Tokenization's impact on model performance

#### Implementation Milestones

- [ ] **Milestone 1: Micrograd Engine**
  - [ ] Complete automatic differentiation system
  - [ ] Multi-layer neural network training capability
  - [ ] Gradient checking and numerical verification
- [ ] **Milestone 2: Character-Level Models**
  - [ ] N-gram baseline implementation
  - [ ] MLP-based character prediction model
  - [ ] Training loop with proper evaluation metrics
- [ ] **Milestone 3: Advanced Training Techniques**
  - [ ] Batch normalization from scratch
  - [ ] Multiple initialization strategies
  - [ ] Learning rate scheduling and optimization
- [ ] **Milestone 4: Sequence Models**
  - [ ] WaveNet-style dilated convolutions
  - [ ] Causal masking and autoregressive generation
  - [ ] Comparison with simpler baseline models
- [ ] **Milestone 5: Transformer Implementation**
  - [ ] Complete self-attention mechanism
  - [ ] Multi-head attention and layer normalization
  - [ ] Position embeddings and transformer blocks
- [ ] **Milestone 6: Tokenization System**
  - [ ] BPE algorithm from scratch
  - [ ] Vocabulary building and encoding/decoding
  - [ ] Integration with transformer model
- [ ] **Milestone 7: GPT-2 Reproduction**
  - [ ] Exact architecture matching
  - [ ] Weight loading and inference verification
  - [ ] Training stability and convergence analysis

#### Knowledge Checks

- [ ] **Check 1:** Verify micrograd gradients match PyTorch autograd on complex expressions
- [ ] **Check 2:** Achieve target perplexity on character-level modeling tasks
- [ ] **Check 3:** Demonstrate batch normalization's training stabilization effects
- [ ] **Check 4:** Show WaveNet's superior modeling of long-range dependencies
- [ ] **Check 5:** Visualize and interpret transformer attention patterns
- [ ] **Check 6:** Build working tokenizer that handles diverse text inputs
- [ ] **Check 7:** Generate coherent text samples from trained GPT model

#### Capstone Project: Complete GPT Implementation and Training

- [ ] **Objective:** Build production-ready GPT implementation with full training pipeline
- [ ] **Phase 1: Foundation (Lectures 1-3)**
  - [ ] Implement micrograd with comprehensive test suite
  - [ ] Build character-level MLP with proper evaluation
  - [ ] Create modular codebase for easy experimentation
- [ ] **Phase 2: Advanced Training (Lectures 4-6)**
  - [ ] Add batch normalization and initialization strategies
  - [ ] Implement WaveNet architecture for sequence modeling
  - [ ] Compare different architectural choices systematically
- [ ] **Phase 3: Transformer Core (Lectures 7-8)**
  - [ ] Complete transformer decoder implementation
  - [ ] Build robust tokenization system with BPE
  - [ ] Create text generation interface with sampling strategies
- [ ] **Phase 4: Scale and Reproduce (Lectures 9-10)**
  - [ ] Implement GPT-2 (124M) architecture exactly
  - [ ] Load and verify pre-trained weights
  - [ ] Train smaller model from scratch on custom dataset
- [ ] **Phase 5: Analysis and Documentation**
  - [ ] Compare all implemented models on common benchmarks
  - [ ] Create comprehensive documentation with mathematical derivations
  - [ ] Build interactive demo showcasing different model capabilities

---

### Module 0.8: PyTorch Mastery and Best Practices

**Advanced PyTorch techniques for efficient deep learning development**

#### Core Concepts Checklist

- [ ] **Advanced PyTorch Features:**
  - [ ] **Custom Datasets and DataLoaders:** Efficient data pipeline creation
  - [ ] **Model Checkpointing:** State preservation and resumable training
  - [ ] **Mixed Precision Training:** FP16/BF16 for memory and speed optimization
  - [ ] **Distributed Training:** Multi-GPU and multi-node parallelism
- [ ] **Model Architecture Patterns:**
  - [ ] **nn.Module Subclassing:** Clean architecture definition
  - [ ] **Forward Hooks:** Intermediate feature extraction and analysis
  - [ ] **Parameter Groups:** Different learning rates for different layers
  - [ ] **Model Surgery:** Layer freezing, replacement, and fine-tuning
- [ ] **Training Loop Engineering:**
  - [ ] **Gradient Accumulation:** Effective batch size scaling
  - [ ] **Learning Rate Scheduling:** Cosine annealing, step decay, plateau reduction
  - [ ] **Early Stopping:** Overfitting prevention and training efficiency
  - [ ] **Metrics Tracking:** Comprehensive logging and visualization
- [ ] **Memory and Performance Optimization:**
  - [ ] **torch.compile:** JIT compilation for faster execution
  - [ ] **Gradient Checkpointing:** Trading compute for memory
  - [ ] **DataLoader Optimization:** num_workers, pin_memory, prefetch_factor tuning
  - [ ] **CUDA Optimization:** Device placement and tensor operations

#### Knowledge Checks

- [ ] **Exercise 1:** Create custom dataset class with proper transforms and lazy loading
- [ ] **Exercise 2:** Implement model checkpointing with optimizer state and resume capability
- [ ] **Exercise 3:** Set up mixed precision training with automatic loss scaling
- [ ] **Exercise 4:** Build flexible training loop with multiple metrics and early stopping
- [ ] **Exercise 5:** Optimize DataLoader performance and measure throughput improvements

#### Capstone Project: Production-Ready Training Framework

- [ ] **Objective:** Build comprehensive PyTorch training framework with modern best practices
- [ ] **Step 1:** Create modular dataset classes supporting multiple data formats
- [ ] **Step 2:** Implement flexible model factory with configuration-driven architecture
- [ ] **Step 3:** Build advanced training loop with all optimization techniques
- [ ] **Step 4:** Add comprehensive logging, checkpointing, and visualization
- [ ] **Step 5:** Implement distributed training support with proper synchronization
- [ ] **Step 6:** Create configuration system for reproducible experiments
- [ ] **Step 7:** Test framework on multiple tasks (classification, generation, regression)

---

### Module 0.9: Mathematical Deep Learning Theory

**Theoretical foundations underlying deep learning success**

#### Core Concepts Checklist

- [ ] **Universal Approximation Theory:**
  - [ ] Theoretical guarantees for neural network expressivity
  - [ ] Width vs. depth trade-offs in approximation power
  - [ ] Practical implications for architecture design
- [ ] **Optimization Landscape Analysis:**
  - [ ] Loss surface geometry and critical points
  - [ ] Saddle points vs. local minima in high dimensions
  - [ ] Lottery ticket hypothesis and network pruning theory
- [ ] **Generalization Theory:**
  - [ ] PAC-Bayes bounds and generalization guarantees
  - [ ] Rademacher complexity and uniform convergence
  - [ ] Double descent phenomenon and overparameterization benefits
- [ ] **Information Theory Perspectives:**
  - [ ] Information bottleneck principle in deep networks
  - [ ] Mutual information and representation learning
  - [ ] Compression and prediction trade-offs

#### Knowledge Checks

- [ ] **Exercise 1:** Prove universal approximation theorem for single hidden layer networks
- [ ] **Exercise 2:** Analyze loss landscape properties using Hessian eigenvalue analysis
- [ ] **Exercise 3:** Compute generalization bounds for specific network architectures
- [ ] **Exercise 4:** Implement information bottleneck analysis for trained networks
- [ ] **Exercise 5:** Demonstrate double descent with systematic overparameterization experiments

#### Capstone Project: Theoretical Analysis of Deep Networks

- [ ] **Objective:** Comprehensive theoretical analysis of neural network behavior
- [ ] **Step 1:** Implement tools for loss landscape visualization and analysis
- [ ] **Step 2:** Study generalization behavior across different model sizes
- [ ] **Step 3:** Analyze information flow through network layers during training
- [ ] **Step 4:** Compare theoretical predictions with empirical observations
- [ ] **Step 5:** Create visual demonstrations of key theoretical concepts
- [ ] **Step 6:** Write comprehensive report connecting theory to practice

---

### Module 0.10: Research Paper Implementation Marathon

**Implementing foundational papers to understand algorithmic innovations**

#### Core Papers to Implement

- [ ] **ResNet (He et al., 2015):** "Deep Residual Learning for Image Recognition"
  - [ ] Identity mapping implementation and gradient flow analysis
  - [ ] Comparison with plain networks and vanishing gradient demonstration
- [ ] **Attention Is All You Need (Vaswani et al., 2017):** Original Transformer paper
  - [ ] Complete encoder-decoder architecture from paper specifications
  - [ ] Multi-head attention and position encoding implementation
- [ ] **BERT (Devlin et al., 2018):** "BERT: Pre-training of Deep Bidirectional Transformers"
  - [ ] Masked language modeling and next sentence prediction objectives
  - [ ] Bidirectional attention mechanism and fine-tuning procedures
- [ ] **GPT (Radford et al., 2018):** "Improving Language Understanding by Generative Pre-Training"
  - [ ] Autoregressive language modeling with transformer decoder
  - [ ] Unsupervised pre-training and supervised fine-tuning pipeline
- [ ] **Adam (Kingma & Ba, 2014):** "Adam: A Method for Stochastic Optimization"
  - [ ] Adaptive moment estimation with bias correction
  - [ ] Comparison with SGD and other optimizers on multiple tasks

#### Implementation Standards

- [ ] **Paper Fidelity:** Match exact architectures, hyperparameters, and training procedures
- [ ] **Reproducibility:** Achieve reported performance within reasonable variance
- [ ] **Code Quality:** Clean, documented, and modular implementations
- [ ] **Analysis:** Detailed comparison with paper claims and ablation studies

#### Knowledge Checks

- [ ] **Check 1:** ResNet implementation matches paper architecture and achieves target accuracy
- [ ] **Check 2:** Transformer implementation successfully trains on machine translation task
- [ ] **Check 3:** BERT implementation achieves reasonable performance on GLUE tasks
- [ ] **Check 4:** GPT implementation generates coherent text and achieves target perplexity
- [ ] **Check 5:** Adam optimizer implementation matches theoretical algorithm exactly

#### Capstone Project: Paper Implementation Portfolio

- [ ] **Objective:** Create comprehensive portfolio of influential paper implementations
- [ ] **Step 1:** Select 5 foundational papers across different domains (vision, NLP, optimization)
- [ ] **Step 2:** Implement each paper with meticulous attention to detail
- [ ] **Step 3:** Reproduce key results and verify against paper claims
- [ ] **Step 4:** Create detailed analysis comparing implementations and identifying insights
- [ ] **Step 5:** Build interactive demos showcasing each implementation
- [ ] **Step 6:** Write technical blog posts explaining implementation challenges and solutions
- [ ] **Step 7:** Open-source implementations with comprehensive documentation and tutorials

---

## The Art of Model Miniaturization

Master the techniques that make models smaller, faster, and more efficient by transforming the model itself.

### Module 1.1: Fundamentals of Quantization Theory

**Foundation concepts for precision reduction and computational efficiency**

#### Core Concepts Checklist

- [ ] **Introduction to Quantization:** Understand goals (reduce computational demands and memory footprint) and benefits (model size reduction, faster inference, lower energy consumption, reduced memory bandwidth pressure)
- [ ] **Data Types:** Differentiate between FP32 and lower-precision formats (INT8, INT4, E4M3, E5M2)
- [ ] **Core Mechanics:**
  - [ ] Define and calculate **Scaling Factor (s):** Positive floating-point scalar defining quantization step size
  - [ ] Define and calculate **Zero-Point (Z):** Integer ensuring precise zero representation
- [ ] **Quantization Schemes:**
  - [ ] **Symmetric vs. Asymmetric:** Trade-offs between single scaling factor vs. scale + zero-point approaches
  - [ ] **Uniform vs. Non-Uniform:** Equal-sized intervals vs. variable-sized intervals (k-means clustering)
- [ ] **Quantization Error:** Difference between original and quantized-dequantized values
- [ ] **Granularity:**
  - [ ] **Per-tensor:** Single scale/zero-point for entire tensor
  - [ ] **Per-channel/Per-axis:** Separate parameters per channel/axis
  - [ ] **Group-wise/Block:** Intermediate approach balancing accuracy and overhead
- [ ] **Fundamental Trade-offs:** Accuracy vs. Efficiency, Complexity vs. Performance, Hardware Support constraints

#### Knowledge Checks

- [ ] **Exercise 1:** Implement symmetric and asymmetric INT8 quantization functions, calculate MSE between original and dequantized arrays
- [ ] **Exercise 2:** Compare per-tensor vs. per-channel quantization on normal distribution with 1% outliers
- [ ] **Exercise 3:** Research five hardware accelerators and identify their supported low-precision data types

#### Capstone Project: Building a Basic Quantizer

- [ ] **Objective:** Implement per-tensor and per-channel static quantization for CNN, evaluate accuracy and model size impact
- [ ] **Step 1:** Train baseline FP32 CNN (LeNet-5 variant) on MNIST
- [ ] **Step 2:** Prepare calibration dataset (100-200 images) for activation ranges
- [ ] **Step 3:** Implement per-tensor static quantization using calibration data
- [ ] **Step 4:** Implement per-channel static quantization for conv/linear weights
- [ ] **Step 5:** Create two INT8 versions: fully per-tensor and per-channel weights + per-tensor activations
- [ ] **Step 6:** Evaluate accuracy on MNIST test set, calculate theoretical size reduction
- [ ] **Step 7:** Analyze results with histograms and trade-off discussion

---

### Module 1.2: Advanced Quantization Techniques

**State-of-the-art methods for production-ready model compression**

#### Core Concepts Checklist

- [ ] **Post-Training Quantization (PTQ):**
  - [ ] **Static PTQ:** Offline quantization of weights and activations using calibration dataset
  - [ ] **Dynamic PTQ:** Offline weight quantization, runtime activation quantization
- [ ] **Quantization-Aware Training (QAT):**
  - [ ] Fake quantization nodes in forward pass
  - [ ] **Straight-Through Estimator (STE):** Gradient flow through non-differentiable rounding
- [ ] **SOTA LLM Quantization Techniques:**
  - [ ] **QLoRA:** 4-bit NormalFloat (NF4), Double Quantization, Paged Optimizers
  - [ ] **GPTQ:** Layer-wise quantization with error compensation via Optimal Brain Quantization
  - [ ] **AWQ:** Activation-aware weight protection through scaling transformations
  - [ ] **SmoothQuant:** Activation outlier smoothing for efficient W8A8 inference

#### Comparative Analysis Table

| Feature | QLoRA | GPTQ | AWQ | SmoothQuant |
|---------|-------|------|-----|-------------|
| **Core Mechanism** | 4-bit NF4 + LoRA adapters | Layer-wise with error compensation | Activation-aware channel scaling | Outlier smoothing transformation |
| **Training Required** | LoRA adapters only | No (PTQ) | No (PTQ) | No (PTQ) |
| **Bit-widths** | 4-bit (NF4) | 2-8 bit | 3-4 bit | 8-bit W8A8 |
| **Hardware Friendly** | Good (custom kernels) | High | Very High | Very High |
| **Key Innovation** | NF4 + Double Quantization | OBQ adaptation | Activation magnitude protection | Mathematical equivalence |

#### Knowledge Checks

- [ ] **Exercise 1:** Compare static PTQ, dynamic PTQ, and QAT methodologies and use cases
- [ ] **Exercise 2:** Explain STE's role in enabling QAT and why it's necessary
- [ ] **Exercise 3:** Detail QLoRA's NF4, Double Quantization, and Paged Optimizers
- [ ] **Exercise 4:** Explain GPTQ's layer-wise quantization and desc_act option
- [ ] **Exercise 5:** Describe AWQ's activation magnitude focus and channel scaling
- [ ] **Exercise 6:** Analyze SmoothQuant's alpha parameter effects and W8A8 benefits

#### Capstone Project: QLoRA vs. GPTQ Comparison

- [ ] **Objective:** Apply QLoRA and GPTQ to medium LLM (OPT-2.7B/Llama-7B), compare outcomes
- [ ] **Part A: QLoRA Implementation**
  - [ ] Load base model in 4-bit using bitsandbytes
  - [ ] Add trainable LoRA adapters (W = W₀ + BA)
  - [ ] Fine-tune adapters on instruction dataset
  - [ ] Evaluate perplexity and VRAM usage
- [ ] **Part B: GPTQ Implementation**
  - [ ] Apply layer-wise quantization with error compensation
  - [ ] Quantize to 4-bit using AutoGPTQ
  - [ ] Evaluate perplexity and model size
- [ ] **Analysis:** Compare VRAM, model sizes, perplexity, and use case suitability

---

### Module 1.3: Broader Model Compression Strategies

**Complementary techniques beyond quantization for model efficiency**

#### Core Concepts Checklist

- [ ] **Weight Pruning:**
  - [ ] **Unstructured Pruning:** Individual weight removal, requires sparse computation support
  - [ ] **Structured Pruning:** Entire component removal (neurons, channels, heads), hardware-friendly
  - [ ] **Methods:** Magnitude-based, Iterative, One-Shot approaches
  - [ ] **Lottery Ticket Hypothesis:** Dense networks contain sparse "winning ticket" subnetworks
- [ ] **Knowledge Distillation (KD):**
  - [ ] **Teacher/Student Paradigm:** Large teacher guides small student training
  - [ ] **Soft Targets:** Temperature-scaled logits provide richer inter-class information
  - [ ] **Distillation Loss:** Combined supervised + KL divergence loss functions
- [ ] **Low-Rank Adaptation (LoRA):**
  - [ ] Core mechanism: W = W₀ + BA where rank r << matrix dimensions
  - [ ] Benefits: 10,000x parameter reduction, zero inference latency after merging

#### Knowledge Checks

- [ ] **Exercise 1:** Compare unstructured vs. structured pruning impact on model size and GPU speedup
- [ ] **Exercise 2:** Explain soft targets' role and effectiveness vs. hard labels in knowledge distillation
- [ ] **Exercise 3:** Calculate LoRA parameter reduction for linear layer (d_in=2048, d_out=8192, r=16)
- [ ] **Exercise 4:** Describe iterative pruning process and fine-tuning importance
- [ ] **Exercise 5:** Assess knowledge distillation feasibility across different architectures

#### Capstone Project: Transformer Compression Pipeline

- [ ] **Objective:** Apply pruning and distillation to BERT-base for text classification
- [ ] **Step 1:** Fine-tune BERT-base on text classification (SST-2) as teacher baseline
- [ ] **Step 2:** Apply iterative magnitude-based pruning at 30%, 50%, 70% sparsity levels
- [ ] **Step 3:** Train smaller student (DistilBERT) using knowledge distillation
- [ ] **Step 4 (Optional):** Apply pruning to distilled student model
- [ ] **Step 5:** Compare all variants on parameters, accuracy, reduction %, and accuracy drop

---

### Module 1.4: Model Formats and Conversion Pipelines

**Production deployment formats and optimization toolchains**

#### Core Concepts Checklist

- [ ] **GGUF (GPT-Generated Unified Format):**
  - [ ] **Key Features:** Single-file format, metadata + weights + tokenizer, native quantization, memory-mappable
  - [ ] **File Structure:** Header (magic, version), Metadata KV Store, Tensor Info, Tensor Data
  - [ ] **Security Considerations:** Parser vulnerabilities, trusted source requirements
- [ ] **llama.cpp Ecosystem:**
  - [ ] Workflow: HuggingFace → GGUF (F16) → quantize tool → final model
- [ ] **ONNX/TensorRT Pipeline:**
  - [ ] **PyTorch → ONNX:** Interoperable standard, dynamic axes challenges
  - [ ] **ONNX → TensorRT:** Layer fusion, precision calibration, kernel auto-tuning

#### GGUF Structure Overview

| Section | Purpose | Data Types | Significance |
|---------|---------|------------|--------------|
| **Header** | File validation, counts | uint32, uint64 | Version check, memory allocation |
| **Metadata KV** | Architecture, tokenizer, quantization | Mixed types | Model initialization parameters |
| **Tensor Info** | Name, shape, type, offset per tensor | Strings, arrays, enums | Memory mapping setup |
| **Tensor Data** | Raw weight bytes with alignment | Byte arrays | Efficient mmap loading |

#### Knowledge Checks

- [ ] **Exercise 1:** Explain GGUF development motivation as GGML successor
- [ ] **Exercise 2:** Describe GGUF header magic number and version field purposes
- [ ] **Exercise 3:** Analyze GGUF metadata section tokenizer inclusion advantages
- [ ] **Exercise 4:** Explain memory mapping and GGUF tensor data alignment importance
- [ ] **Exercise 5:** Identify PyTorch → ONNX conversion challenges
- [ ] **Exercise 6:** Describe trtexec role in ONNX → TensorRT conversion
- [ ] **Exercise 7:** Explain why TensorRT → GGUF conversion isn't typical workflow

#### Capstone Project: HuggingFace to GGUF Pipeline

- [ ] **Objective:** Convert HuggingFace transformer to GGUF, run inference with llama.cpp
- [ ] **Step 1:** Clone llama.cpp, setup environment, compile executables
- [ ] **Step 2:** Select compatible HuggingFace model (OPT, GPT-2, Llama variant)
- [ ] **Step 3:** Convert to F16 GGUF using convert-hf-to-gguf.py
- [ ] **Step 4:** Understand GGUF file structure and contained information
- [ ] **Step 5:** Quantize F16 → Q4_K_M using quantize tool
- [ ] **Step 6:** Run inference with both F16 and Q4_K_M versions
- [ ] **Step 7:** Compare file sizes, loading times, and generation quality

---

### Module 1.5: CUDA Kernels for Quantized Operations

**Low-level GPU programming for custom quantization kernels**

#### Core Concepts Checklist

- [ ] **CUDA Programming Fundamentals:**
  - [ ] **Host/Device Model:** CPU manages application, launches GPU kernels
  - [ ] **Kernels:** `__global__` functions executed by many GPU threads
  - [ ] **Thread Hierarchy:** Threads → Thread Blocks → Grids
  - [ ] **Launch Syntax:** `kernel<<<blocks, threads>>>(args)`
- [ ] **CUDA Memory Model:**
  - [ ] **Registers:** Fastest, private per thread, limited
  - [ ] **Shared Memory:** Low-latency, block-shared, user-managed cache
  - [ ] **Global Memory:** Largest, slowest, requires coalescing optimization
  - [ ] **Constant/Texture:** Read-only, cached, specialized access patterns
  - [ ] **Unified Memory:** Single address space, runtime migration
- [ ] **Quantized Kernel Considerations:**
  - [ ] Integer arithmetic and bitwise operations for sub-byte types
  - [ ] Higher-precision accumulators to prevent overflow
  - [ ] Correct scaling factor and zero-point application
- [ ] **Optimization Techniques:**
  - [ ] **Memory Coalescing:** Contiguous warp memory access
  - [ ] **Shared Memory Tiling:** User-managed caching for data reuse
  - [ ] **Occupancy Maximization:** Active warps per SM ratio
  - [ ] **Divergence Minimization:** Avoid conditional branches within warps

#### Memory Hierarchy Table

| Memory Type | Location | Scope | Latency | Key Characteristics | Optimization Strategy |
|-------------|----------|-------|---------|-------------------|-------------------|
| **Registers** | On-chip | Thread | Very Low | Fastest, private, limited quantity | Maximize usage, avoid spills |
| **Shared Memory** | On-chip | Block | Low | User-managed cache, inter-thread communication | Use for tiling, avoid bank conflicts |
| **Global Memory** | Off-chip | Grid | High | Largest space, all-thread accessible | Coalesce accesses, minimize total |
| **Constant Memory** | Off-chip | Grid | Medium | Read-only, 64KB, broadcast efficient | Use for kernel parameters |
| **Unified Memory** | Variable | System | Variable | Single address space, runtime managed | Use prefetch for performance |

#### Knowledge Checks

- [ ] **Exercise 1:** Explain memory coalescing with coalesced vs. non-coalesced examples
- [ ] **Exercise 2:** Describe shared memory bank conflicts and avoidance strategies
- [ ] **Exercise 3:** Compare Unified Memory advantages vs. manual cudaMemcpy
- [ ] **Exercise 4:** Analyze thread divergence impact with conditional examples
- [ ] **Exercise 5:** Identify 4-bit quantization kernel challenges (loading, arithmetic, storage)

#### Capstone Project: Custom Quantized Vector Addition Kernel

- [ ] **Objective:** Implement CUDA kernel for INT8 vector addition with proper scaling
- [ ] **Step 1:** Create FP32 input vectors, implement CPU quantization/dequantization reference
- [ ] **Step 2:** Implement `quantized_vector_add_kernel` with per-element processing
- [ ] **Step 3:** Write host orchestration: memory allocation, transfers, kernel launch
- [ ] **Step 4:** Verify GPU results match CPU reference, optional Nsight Compute profiling

---

### Module 1.6: Profiling Quantized Models

**Performance analysis and optimization guidance for quantized models**

#### Core Concepts Checklist

- [ ] **Profiling Importance:** Bottleneck identification, speedup verification, resource utilization analysis, optimization guidance
- [ ] **NVIDIA Profiling Tools:**
  - [ ] **Nsight Systems:** System-wide timeline analysis, CPU-GPU interaction, pipeline bottlenecks
  - [ ] **Nsight Compute:** Detailed kernel analysis, instruction throughput, memory patterns, occupancy
- [ ] **TensorBoard:** Weight/activation distribution monitoring for QAT debugging and PTQ comparison
- [ ] **Quantized Model Specifics:** Kernel timing improvements, memory bandwidth reduction, Tensor Core utilization, Q/DQ operation overhead

#### Knowledge Checks

- [ ] **Exercise 1:** Compare Nsight Systems vs. Nsight Compute information and use cases
- [ ] **Exercise 2:** Analyze low SM occupancy causes and solutions (registers, shared memory, launch parameters)
- [ ] **Exercise 3:** Describe TensorBoard histogram usage for QAT monitoring and problem identification
- [ ] **Exercise 4:** Identify system-level bottlenecks beyond compute kernels for quantized LLM latency
- [ ] **Exercise 5:** Explain Tensor Core utilization importance and Nsight Compute indicators

#### Capstone Project: Iterative Optimization with Profiling

- [ ] **Objective:** Profile quantized model, identify bottlenecks, implement targeted optimizations
- [ ] **Step 1:** Baseline Nsight Systems profiling of inference pipeline phases
- [ ] **Step 2:** Deep-dive Nsight Compute analysis of time-consuming kernels
- [ ] **Step 3:** Identify 1-2 high-impact optimization targets from profiler data
- [ ] **Step 4:** Implement optimizations (pinned memory, CUDA streams, launch parameters)
- [ ] **Step 5:** Re-profile optimized code to quantify improvements
- [ ] **Step 6:** Document findings, optimizations, and before/after comparison

---

## The Science of Serving

Transition from model optimization to the complex systems that serve models efficiently at scale.

### Module 2.1: Modern AI Inference Stack Architecture

**Understanding the multi-layered infrastructure for production AI serving**

#### Core Concepts Checklist

- [ ] **The 5 Layers of Inference Stack:**
  - [ ] **GPU Clouds/Hardware:** Raw compute foundation (AWS, GCP, CoreWeave, on-premise)
  - [ ] **Compute Infrastructure Management:** Kubernetes orchestration for containerized services
  - [ ] **LLM Inference Engine:** Core runtime with batching and memory optimization (TensorRT-LLM, vLLM)
  - [ ] **KV Cache:** Transformer attention state management for context reuse
  - [ ] **Orchestration and Routing:** Intelligent request management and payload-aware routing
- [ ] **Payload-Aware Routing:** Request inspection for intelligent batching, caching, routing decisions
- [ ] **Deployment Models:** Build-Your-Own (flexibility), Integrated Deployment (balance), Inference-as-a-Service (abstraction)
- [ ] **Containerization:** Docker for portability, dependency management, scalability, reproducibility

#### Knowledge Checks

- [ ] **Exercise 1:** Diagram five inference stack layers with key functions per layer
- [ ] **Exercise 2:** Explain Kubernetes benefits for inference service deployment and scaling
- [ ] **Exercise 3:** Define payload-aware routing with two examples and benefits
- [ ] **Exercise 4:** Analyze containerization contributions to hybrid cloud portability and operations
- [ ] **Exercise 5:** Compare deployment model trade-offs and organizational suitability

#### Capstone Project: Application-Specific Stack Design

- [ ] **Objective:** Design inference stack for specific LLM application with justified component choices
- [ ] **Step 1:** Choose scenario (chatbot, summarization, code completion) with defined KPIs and constraints
- [ ] **Step 2:** Select and justify technology choices for each stack layer
- [ ] **Step 3:** Identify potential bottlenecks per layer with mitigation strategies
- [ ] **Step 4:** Create design document with stack diagram, justifications, and bottleneck analysis

---

### Module 2.2: Deep Dive into Serving Engines

**Comprehensive analysis of production-grade inference engines**

#### Core Concepts Checklist

- [ ] **Triton Inference Server:**
  - [ ] **Architecture:** Model Repository with config.pbtxt, HTTP/gRPC communication, Backend system flexibility
  - [ ] **Features:** Concurrent model execution, dynamic batching, sequence batching, model ensembling
- [ ] **TensorRT-LLM:**
  - [ ] **Workflow:** Python API model definition → TensorRT engine compilation
  - [ ] **Optimizations:** Kernel fusion, advanced quantization (FP8 on Hopper+), in-flight batching, multi-GPU parallelism
- [ ] **vLLM:**
  - [ ] **PagedAttention:** OS virtual memory concepts for KV cache management via non-contiguous pages
  - [ ] **Features:** State-of-the-art throughput via continuous batching, optimized CUDA kernels (FlashAttention)

#### Knowledge Checks

- [ ] **Self-Study 1:** Explain Triton's backend system making it a flexible "meta-server"
- [ ] **Self-Study 2:** Contrast TensorRT-LLM compilation vs. HuggingFace dynamic approaches
- [ ] **Self-Study 3:** Diagram PagedAttention memory waste avoidance vs. traditional contiguous allocation

---

## The Arsenal: Tools & Frameworks

Essential software ecosystem for the inference optimization journey:

**Quantization & Compression**
- bitsandbytes, Hugging Face Optimum, Hugging Face PEFT
- AutoGPTQ, llm-awq

**Deployment & Serving**
- llama.cpp, ONNX, NVIDIA TensorRT
- NVIDIA Triton Inference Server, vLLM

**Low-Level & Profiling**
- CUDA Toolkit, NVIDIA Nsight Systems, NVIDIA Nsight Compute
- Triton (Language)

**Core Frameworks**
- PyTorch, Hugging Face Transformers

---

## The Pantheon: Foundational Papers

Essential research papers underlying the techniques in this curriculum:

- [ ] **QLoRA:** "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- [ ] **GPTQ:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)
- [ ] **AWQ:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)
- [ ] **SmoothQuant:** "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2022)
- [ ] **PagedAttention (vLLM):** "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)
- [ ] **LoRA:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

---

## Captain's Log

My optimization journey with insights, breakthroughs, and challenges:

**Template Entry:**
```
YYYY-MM-DD: Module X.Y - Title
Progress: [Current status and completed work]
Key Insight: [Major understanding or breakthrough moment]
Challenge: [Current obstacles and approach to resolution]
Next Steps: [Planned actions and goals]
```
