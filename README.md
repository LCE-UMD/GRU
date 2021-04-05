# GRU

# Movie Watching Dataset

## Regression

__Model scripts to generate the results__

- `bhv_gru.py`: GRU model
- `bhv_cpm.py`: connectome-based predictive model
- `bhv_ff.py`: fully-connected layer model
- `bhv_tcn.py`: temporally connected network model

All of the above scripts accept the name of the behavioral measure as the `--bhv` argument. There are 7 behavioral measures and their names are as follows: 

1. PMAT24_A_CR
2. PicVocab_Unadj 
3. NEOFAC_A
4. NEOFAC_E
5. NEOFAC_C
6. NEOFAC_N
7. NEOFAC_O

The model scripts need to be run for each behavioral measure separately.

__Plotting notebooks__
- bhv_compare.ipynb: plots results  

## Classification

__Model scripts to generate the results__

- `clip_gru.py`: GRU model
- `clip_gruencoder.py`: GRU followed by dimensionality reduction FC (DRFC) layer
- `clip_ff.py`: fully-connected layer model
- `clip_tcn.py`: temporally connected network model
- `clip_gru_recon.py`: GRU decoder model to reconstruct original timeseries
- `clip_pca.py`: use PCA for clip predction and timeseries reconstruction

__GRU script to generate permutation results__
- `clip_gru_perm.ipynb`

__GRU script to generate saliency results__
- `clip_saliency.ipynb`
- `clip_null_saliency.ipynb`: generate null saliency based upon permutation testing

__Plotting notebooks__
- `clip_performance_compare.ipynb`: plots preformance comparisons of different models
- `clip_gru.ipynb`: plots results only for the GRU model. Includes cross-validation and permutation results
- `clip_gru_recon.ipynb`: compares GRU and PCA for classification accuracy and performance of original signal reconstruction
- `clip_trajectories.ipynb`: plots hidden representations (trajectories) learned by the GRU encoder and euclidean distance between trajectories
- `clip_saliency_rois.ipynb`: plots saliency timeseries for each ROI
- `clip_saliency_nilearn_movie.ipynb`: generates saliency movies for all input movie clips

# Moving Circles Dataset

__Model script to generate results__
- `emo_gru.py`

__GRU script to generate saliency results__
- `emo_saliency.ipynb`

__Plotting notebooks__
- `emo_gru.ipynb`: plots cross-validation and final results

# Additional Files
- `plot_utils`: functions for plotting
- `utils.py`: general functions
- `cc_utils`: functions for clip classification task
- `gru/cc_utils.py`: GRU functions for clip classification task
- `gru/rb_utils.py`: GRU functions for behavioral score regressiont task
- `gru/dataloader.py`: functions to load dataset
- `gru/models.py`: tensorflow classes for GRU, GRU encoder, FF, TCN etc.
- `gru/utils.py`: GRU general functions