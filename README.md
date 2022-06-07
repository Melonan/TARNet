# TARNet
This is the official PyTorch implementation of KDD 2022 paper "TARNet : Task-Aware Reconstruction for Time-Series Transformer."


## Datasets
The classification datasets can be found at [UEA Archive](https://www.timeseriesclassification.com/dataset.php), [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), and from [Towards automatic spatial verification of sensor placement in buildings](https://cseweb.ucsd.edu/~dehong/pdf/buildsys13-paper.pdf).

The regression datasets are available in [Monash University, UEA, UCR Time Series Regression Archive](http://tseregression.org/).

The data directory contains an example of a preprocessed classification dataset: [Atrial Fibrillation (AF)](https://www.timeseriesclassification.com/description.php?Dataset=AtrialFibrillation) and a preprocessed regression dataset: [Appliances Energy (AE)](https://zenodo.org/record/3902637), along with their corresponding preprocessing files. 

The `preprocessing.py` file under `/data/AF/` can be used to preprocess any classification dataset from [UEA Archive](https://www.timeseriesclassification.com/dataset.php) by changing the `filename` parameter. 

The `preprocessing.py` file under `/data/AE/` can be used to preprocess any regression dataset from [Monash University, UEA, UCR Time Series Regression Archive](http://tseregression.org/) by changing the `train_file` and `test_file` parameter.

After running the `preprocessing.py` files on the raw datasets, store the `X_train.npy, y_train.npy, X_test.npy, y_test.npy` for each dataset under the `/data/` directory.


## Quick Start
```
git clone https://github.com/ranakroychowdhury/TARNet.git
```

Run the `script.py` file to train and evaluate the model, like
```
python3 script.py --dataset AF --task_type classification
```

This will train the model on the Atrial Fibrillation (AF) dataset and report the accuracy. Similarly, to run a regression dataset, use `--task_type regression`. 

You can specify any hyper-parameter on the command line and train the model using those hyper-parameters. In this case, uncomment line `52` in `utils.py` file. Or you may use the optimized set of hyperparameters for each dataset, presented in the `hyperparameters.pkl` file. In that case, uncomment line `49` in `utils.py` file.


## File Structure
- `hyperparameters.pkl` stores the best set of hyperparameters for each dataset
- 'multitask_transformer_class.py` TARNet model file
- `script.py` is the main file that loads dataset, initializes, trains and evaluates the model
- 'transformer.py` transformer encoder, multihead attention
- `utils.py` contains all the helper functions.


## Citation
If you find the code useful, please cite our paper:
