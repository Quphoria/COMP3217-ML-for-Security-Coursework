# COMP3217 ML for Security Coursework

This coursework has 2 parts, using 2 different datasets  
 - Part A - This is all files containing Binary, where the dataset has 2 different labels, 0 and 1  
 - Part B - This is all files containing Multi, where the dataset has 3 different labels, 0, 1 and 2  

## This repo has the following structure

- datasets/ - This folder contains the Testing and Training datasets used
- models/ - This folder contains the saved models and metadata about the models
- output/ - This folder contains the outputted Testing datasets with calculated label column added
- research/ - This folder contains python scripts which were used to test different machine learning models (they are messy as they are for developement)
- X_param_search.py - These python scripts are used to tune the hyperparameters by uncommenting lines in the `param_grid` variable
- X_train.py - These python scripts are used to generate the model file used for testing using the training data
- X_test.py - These python scripts used the saved model files to compute the labels for the testing data and save the results in the `output` directory
- csv_to_bunch.py - This script is a helper script used to simplify loading/saving the CSV files
- requirements.txt - This contains the list of required python modules for the other scripts

## Requirements

The scripts have been tested to work on Python 3.8 or higher

To install the required libraries run
```sh
pip install -r requirements.txt
```

## Computing labels for the test data

To compute the labels for the testing data run
```sh
python binary_test.py
python multi_test.py
```

Note: These require models to be saved in the models directory (`binary.model` and `multi.model`)

## Creating the models

To create the models by training with the training data run:
```sh
python binary_train.py
python multi_train.py
```

## Testing Hyperparameters

To tune various hyperparameters you can modify the `binary_param_search.py` and `multi_param_search.py` files.  
By uncommenting/modifying lines in the `param_grid` variable, you can add various parameters to the grid.  
You can run the grid search with:
```sh
python binary_param_search.py
python multi_param_search.py
```

The script will output the best parameters it found, these can then be used to modify the parameters in the training scripts.  

