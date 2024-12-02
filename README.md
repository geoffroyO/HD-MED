# Online EM Algorithm for HD-MED

This is the code of the online EM algorithm for HD-MED and MED currently this package has online EM versions of:

- Standard Gaussian Mixture Model and t-distribution Mixture Model
- High-Dimensional Gaussian Mixture Model and High Dimensional t-distribtuion Mixture Model


## Basic Usage

### Training of the dataset X

```python
from onlineEM import em_config
from onlineEM.hd import hdgmm



config = {
    "n_components": 3, # Number of component of the mixture
    "num_features": 5,  # Number of features
    "num_epochs": 3, # Number of epochs
    "mini_batch_size": 256, # Size of the mini-batched
}

config = em_config(**config) # Create the configuration object

model_hd = hdgmm()

config_hd, params_hd, stats_hd = model_hd.burnin(X, config) # Initialize the model
params_hd, stats_hd = model_hd.online_epochs(X, params_hd, stats_hd, config_hd) # Train the model
```

### Dimension reduction
🚨 This only works for High-Dimensional models 🚨


```python
X_red, idx_clf = hdgmm.project(X, params_hd, config) # Project the initial high-dimensional data to lower different subspaces

# outputs low dimensional data X_red in a form of a list of arrays  for different clusters along with their associated index idx_clf in the original array

X_decompressed = hdgmm.inv_project(X_red, idx_clf)
```

##### 👩‍💻 Basic exemple in **notebook/**. 👨‍💻
### Todo

- [ ] Implement Ruppert-Polyak Averaging and find a way to average variables in the Stiefel manifold