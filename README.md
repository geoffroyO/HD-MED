# Online EM for High-Dimensional Mixture Models

This repository implements online Expectation-Maximization (EM) algorithms for high-dimensional mixture models using the JAX AI stack. The focus is on efficient, factorized implementations that capture low-dimensional structure in high-dimensional data.

## ✨ Features

- **High-Dimensional Gaussian Mixture Model (HD-GMM)**: High-Dimensional Gaussian mixtures
- **High-Dimensional Student-t Mixture Model (HD-STM)**: High-Dimensional Student mixtures
- **High-Dimensional Laplace Mixture Model (HD-LM)**: High-Dimensional Laplace mixtures

## 📚 Tutorial

The main tutorial is provided in `tutorial_hd_models.ipynb`.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd onlineEM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Tutorial

```bash
jupyter notebook tutorial_hd_models.ipynb
```

Or run programmatically:
```bash
jupyter nbconvert --to notebook --execute tutorial_hd_models.ipynb
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{oudoumanessah2024scalable,
  title={Scalable magnetic resonance fingerprinting: Incremental inference of high dimensional elliptical mixtures from large data volumes},
  author={Oudoumanessah, Geoffroy and Coudert, Thomas and Lartizien, Carole and Dojat, Michel and Christen, Thomas and Forbes, Florence},
  journal={arXiv preprint arXiv:2412.10173},
  year={2024}
}
@inproceedings{oudoumanessah2025cluster,
  title={Cluster globally, Reduce locally: Scalable efficient dictionary compression for magnetic resonance fingerprinting},
  author={Oudoumanessah, Geoffroy and Coudert, Thomas and Meyer, Luc and Delphin, Aurelien and Christen, Thomas and Dojat, Michel and Lartizien, Carole and Forbes, Florence},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```