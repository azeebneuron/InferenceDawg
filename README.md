## Progress Dashboard

| Category | Status | Completion | Modules |
|----------|--------|------------|---------|
| **Part I: The Art of Model Miniaturization** | Not Started | █░░░░░░░░░░░░░░░░░░░░░░░░ | 0 / 6 |
| **Part II: The Science of Serving** | Not Started | ░░░░░░░░░░░░░░░░░░░░░░░░░ | 0 / 2 |
| **Total Projects Completed** | Not Started | ░░░░░░░░░░░░░░░░░░░░░░░░░ | **0 / 8** |


## Part I: The Art of Model Miniaturization

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

## Part II: The Science of Serving

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
