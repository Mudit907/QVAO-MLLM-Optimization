Quantum-Inspired Optimization for Multi-Modal LLMs

This repository documents my research journey into enhancing multi-modal large language models (MLLMs) using quantum-inspired optimization techniques. As a third-year CSE student at SRM Institute, Chennai, I aimed to develop a Quantum-Inspired Variational Attention Optimizer (QVAO) to improve the efficiency and accuracy of MLLMs like LLaVA-1.5-7B, particularly for visual question answering (VQA) tasks using the VQAv2 dataset. Due to resource limitations, the project couldn’t be fully implemented, but this repo captures the theoretical framework, classical baseline setup, and planned quantum approach.

**Project Overview**
- **Goal**: Optimize attention mechanisms in MLLMs using quantum-inspired methods to enhance cross-modal alignment and reduce computational costs.
- **Tools**: Kaggle (classical baseline), IBM Quantum Labs (quantum phase), LLaVA-1.5-7B, VQAv2 dataset, Qiskit, PyTorch.
- **Status**: Incomplete due to resource constraints; serves as a proof-of-concept and learning resource.

**Classical Baseline**
The classical phase sets up a baseline on Kaggle using LLaVA-1.5-7B and VQAv2. Key steps include:
- Installing dependencies and loading the dataset.
- Testing the model with inference on a sample image.
- Evaluating baseline VQA accuracy.

**Quantum Phase (Planned)**
The quantum phase was intended to implement QVAO on IBM Quantum Labs, using Qiskit to simulate variational quantum circuits for attention optimization. This remains a theoretical outline due to unavailability of quantum resources.

**Files**
- `setup_kaggle_v3.py`: Installs dependencies for the classical setup on Kaggle.
- `load_vqav2_kaggle_v2.py`: Loads and saves the VQAv2 dataset.
- `load_llava_kaggle_v2.py`: Loads and tests LLaVA-1.5-7B with a sample image.
- `evaluate_baseline_kaggle_v2.py`: Evaluates baseline VQA accuracy.
- `qvao_quantum_outline.py`: Outlines the planned quantum optimization approach.

**Usage**
1. **Classical Setup**: Run the scripts in order on Kaggle (upload "apple.jpg" to `/kaggle/input/` for testing).
2. **Quantum Exploration**: Review `qvao_quantum_outline.py` and adapt for IBM Quantum Labs with Qiskit.
3. **Contributions**: Feel free to fork, improve, or suggest enhancements!
**
Acknowledgments**
This work was inspired by discussions and guidance from the xAI community. Special thanks to my peers and faculty at SRM for their support.

**Future Work**
With access to better resources (e.g., university HPC or IBM Quantum), I plan to complete the QVAO implementation.

