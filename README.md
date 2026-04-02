# StatLine Lab 8 Code

This folder contains the code used for Lab 8 of CS 4364.

## Project Summary

StatLine is an NBA prediction project with a pre-game prediction pipeline based on engineered team features such as blended season statistics, recent momentum, and Elo ratings.

For Lab 8, we implemented and tested a 1D CNN and compared it against the existing SVM baseline for pre-game NBA game prediction.

## Main Files

- `train_svm_momentum_svm.py`  
  Trains the original SVM-based pre-game model using last-season averages, current-season momentum, and Elo ratings.

- `predict_today_svm.py`  
  Uses the trained SVM model to generate pre-game predictions for a given date.

- `train_cnn_pregame_temporal.py`  
  Builds the same pre-game feature set, applies a chronological 80/20 split, trains both the SVM baseline and a 1D CNN, and compares them using accuracy, precision, recall, and F1.


## How to Run

### 1. Train the original SVM model
python3 train_svm_momentum_svm.py

### 2. Generate SVM predictions for a given date
python3 predict_today_svm.py --date YYYY-MM-DD 
or simply run 
python3 predict_today_svm.py (for today's game dirrectly)

### 3. Run the chronological CNN vs SVM comparison
python3 train_cnn_pregame_temporal.py