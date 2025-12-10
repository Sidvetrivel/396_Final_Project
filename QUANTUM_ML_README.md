# Quantum Machine Learning Comparative Study

## Overview

This project implements a comprehensive comparison between classical and quantum machine learning algorithms on the spirals dataset, a standard benchmark for QML research.

## Project Structure

### Datasets
- **Circles (N-spheres)**: 2D circular decision boundaries
- **Spirals**: 2D spiral patterns with adjustable difficulty (2-8 turns)
- **Bars and Stripes**: 4-bit binary classification

### Algorithms Implemented

#### Classical Algorithms (5 total)
1. **SVM (RBF Kernel)** - Radial basis function support vector machine
2. **SVM (Polynomial Kernel)** - Polynomial kernel SVM
3. **Neural Network** - Multi-layer perceptron with 50-30 hidden units
4. **Random Forest** - Ensemble of 100 decision trees
5. **K-Nearest Neighbors** - k=5 neighbors

#### Quantum Algorithms (4 total)

**Quantum Kernel Methods:**
1. **QSVC (ZZ Kernel)** - Quantum SVM with ZZ feature map
   - Uses quantum circuits to compute kernel matrices
   - ZZ gates create entanglement between qubits

2. **QSVC (Pauli Kernel)** - Quantum SVM with Pauli feature map
   - Uses Pauli rotations (Z, ZZ) for quantum encoding
   - Full entanglement structure

**Variational Quantum Classifiers:**
3. **VQC (COBYLA)** - Variational circuit with COBYLA optimizer
   - Gradient-free optimization
   - RealAmplitudes ansatz with 3 repetitions

4. **VQC (SPSA)** - Variational circuit with SPSA optimizer
   - Gradient-based optimization (more efficient)
   - Same circuit structure as COBYLA version

## Installation

### Required Packages

```bash
# Classical ML libraries
pip install scikit-learn scipy pandas matplotlib numpy

# Quantum computing libraries
pip install qiskit qiskit-machine-learning qiskit-algorithms
```

### Verify Installation

Run the first few cells in `demo_script.ipynb`. You should see:
- ✓ All required packages loaded successfully
- ✓ Qiskit and quantum ML packages loaded successfully

## Usage

### Quick Start

1. **Install packages** (see above)
2. **Open** `demo_script.ipynb`
3. **Run cells sequentially** from top to bottom

### Running Comparisons

#### Step 1: Generate Datasets
```python
# Runs automatically when you execute the dataset generation cells
# Creates Easy, Medium, Hard, and Very Hard difficulty levels
```

#### Step 2: Test Classical Algorithms
```python
# Already set up in the notebook
# Tests SVM, Neural Networks, Random Forest, KNN
# Shows decision boundaries and accuracy metrics
```

#### Step 3: Test Quantum Algorithms
```python
# Tests QSVC and VQC variants
# Note: Quantum simulation is slower!
# Recommended: Start with reduced dataset (200 samples)
```

#### Step 4: Generate Comparison
```python
# Uncomment the comparison code cell
# Creates side-by-side visualizations
# Generates statistical analysis
```

## Performance Expectations

### Classical Algorithms
- **Training time**: < 1 second per algorithm
- **Accuracy on Medium difficulty**: 85-95%
- **Best performer**: Typically SVM (RBF) or Neural Network

### Quantum Algorithms (Simulated)
- **Training time**: 5-15 minutes per algorithm
- **Accuracy on Medium difficulty**: 70-85% (varies)
- **Bottleneck**: Quantum circuit simulation overhead

### Why are quantum algorithms slower?
- Running on classical simulators (not real quantum hardware)
- Each kernel evaluation requires quantum circuit simulation
- O(n²) kernel matrix computations for n training samples

## Evaluation Metrics

The framework automatically computes:
- **Test Accuracy** - Primary performance metric
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of precision and recall
- **Training Time** - Wall-clock time for model training

## Visualization Tools

### Decision Boundaries
- 2D contour plots showing classification regions
- Color-coded by predicted class
- Data points overlaid with true labels

### Performance Plots
- Accuracy vs difficulty level
- Training time comparisons
- Bar charts for metric comparisons

### Statistical Analysis
- Mean and standard deviation across difficulty levels
- T-tests for statistical significance
- Best algorithm per difficulty level

## Tips for Success

### For Faster Experimentation
1. **Use smaller datasets**: Reduce `subset_size` to 100-200 samples
2. **Test one difficulty**: Start with Medium only
3. **Reduce iterations**: Lower VQC `maxiter` to 50
4. **Test sequentially**: Run one quantum algorithm at a time

### For Best Results
1. **Use full datasets**: All 700 training samples
2. **Test all difficulties**: Easy through Very Hard
3. **Multiple runs**: Average results over 3-5 runs
4. **Hyperparameter tuning**: Optimize VQC circuit depth and reps

## Writing Your Assessment

### Data Collection
After running comparisons, use:
```python
assessment_data = generate_assessment_data(all_classical_results, all_quantum_results)
print_assessment_summary(assessment_data)
```

This provides:
- Overall statistics (mean accuracy, std, times)
- Statistical significance tests (p-values)
- Best algorithms per difficulty
- Performance tables

### Assessment Template
The notebook includes a complete template covering:
1. **Executive Summary** - Key findings (2-3 sentences)
2. **Accuracy and Performance** - Algorithm-by-algorithm analysis
3. **Computational Efficiency** - Training time comparisons
4. **Robustness** - Performance across difficulty levels
5. **Decision Boundary Quality** - Visual analysis
6. **Practical Considerations** - NISQ limitations, implementation complexity
7. **Conclusions** - Current recommendations and future potential

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: `pip install scikit-learn`

```
ModuleNotFoundError: No module named 'qiskit'
```
**Solution**: `pip install qiskit qiskit-machine-learning qiskit-algorithms`

### Slow Performance
**Problem**: Quantum algorithms taking too long

**Solutions**:
- Reduce dataset size (use 100-200 samples)
- Lower VQC iterations (maxiter=50)
- Use only Medium difficulty
- Test one algorithm at a time

### Memory Issues
**Problem**: Kernel out of memory

**Solutions**:
- Restart kernel and clear outputs
- Reduce dataset size
- Close other applications
- Use coarser decision boundary resolution

### Accuracy Issues
**Problem**: Quantum algorithms showing poor accuracy

**Possible causes**:
- Too few training samples
- VQC not converging (increase maxiter)
- Feature map mismatch with problem structure
- Need hyperparameter tuning

## Expected Outcomes

Based on typical results:

### Classical Algorithms
- **Best**: SVM (RBF) ~90-95% accuracy
- **Good**: Neural Network, Random Forest ~85-90%
- **Baseline**: KNN ~80-85%

### Quantum Algorithms (Simulated)
- **QSVC**: 75-85% accuracy (competitive but slower)
- **VQC**: 70-80% accuracy (depends on optimization)
- **Trade-off**: Lower accuracy but explores quantum advantage potential

### Key Insights
1. Classical algorithms currently outperform quantum on simulated hardware
2. Quantum algorithms are limited by:
   - Simulation overhead
   - NISQ-era noise (if using real hardware)
   - Limited qubit count
3. Quantum advantage may emerge with:
   - Larger quantum computers
   - Better error correction
   - Quantum-friendly problem structures

## References

### Quantum Machine Learning
- [Qiskit Machine Learning Documentation](https://qiskit.org/ecosystem/machine-learning/)
- [Havlíček et al. - Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2)
- [Schuld & Killoran - Quantum machine learning in feature Hilbert spaces](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504)

### Classical Baselines
- [scikit-learn Documentation](https://scikit-learn.org/)
- [SVM Guide](https://scikit-learn.org/stable/modules/svm.html)
- [Neural Network Guide](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

### Datasets
- Spirals: [TensorFlow Playground](https://playground.tensorflow.org/)
- QML Benchmarks: [PennyLane Demos](https://pennylane.ai/qml/demos.html)

## Project Checklist

- [ ] Install scikit-learn and scipy
- [ ] Install Qiskit and quantum ML packages
- [ ] Generate spirals datasets (all difficulty levels)
- [ ] Test classical algorithms (5 algorithms)
- [ ] Visualize classical decision boundaries
- [ ] Test quantum algorithms (4 algorithms)
- [ ] Visualize quantum decision boundaries
- [ ] Run comparative analysis
- [ ] Generate statistical summary
- [ ] Write assessment using template
- [ ] Include visualizations in report
- [ ] Cite relevant papers

## Contact & Support

For issues with:
- **Qiskit**: [Qiskit GitHub Issues](https://github.com/Qiskit/qiskit/issues)
- **scikit-learn**: [sklearn Documentation](https://scikit-learn.org/)
- **This project**: Check notebook comments and markdown cells

## License

This project uses:
- Qiskit: Apache License 2.0
- scikit-learn: BSD License
- Dataset generators: MIT License (from Basic_Datasets repository)

---

**Good luck with your quantum machine learning comparative study!** 🚀🔬
