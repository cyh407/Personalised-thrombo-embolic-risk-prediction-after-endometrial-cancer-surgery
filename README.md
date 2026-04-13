# Personalised-thrombo-embolic-risk-prediction-after-endometrial-cancer-surgery
Personalised thrombo-embolic risk prediction after endometrial cancer surgery


## 🚀 How to Run (运行步骤)

Please execute the scripts in the following order to reproduce the results:
1. `python 01_train.py` - Initial model training.
2. `python 02_feature_elimination.py` - Perform feature selection/elimination.
3. `python 03_svm_optimization.py` - Hyperparameter tuning for the SVM model.
4. `python 04_calibrated_svm.py` - Model calibration.
5. `python 05_plot_dca_curve.py` - Plot the Decision Curve Analysis (DCA) to evaluate clinical utility.
