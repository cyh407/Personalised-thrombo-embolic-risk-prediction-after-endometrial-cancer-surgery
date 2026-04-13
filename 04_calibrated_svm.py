import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss

# ==========================================
# 1. Load data and split dataset
# ==========================================
# Define the 4 selected features and the target variable
selected_features = ['Postop_D_Dimer', 'Clinical_Stage', 'Pathological_Grade', 'Age']
target_col = 'Thrombosis'

# Read training and test data
df_main = pd.read_excel(r'data/data.xlsx')
X_main = df_main[selected_features]
y_main = df_main[target_col]

# ==========================================
# 2. Dataset splitting (Train 80%, Test 20%)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_main, y_main, test_size=0.2, random_state=42, stratify=y_main
)

print(f"Data loaded successfully:")
print(f"- Train set: {X_train.shape[0]} samples")
print(f"- Test set : {X_test.shape[0]} samples")

# ==========================================
# 3. Load the previously saved best model
# ==========================================
print("Loading original SVM model...")
# Ensure 'best_svm_model.joblib' is in the current working directory
best_svm_model = joblib.load('best_svm_model.joblib')

# ==========================================
# 4. Probability calibration using CalibratedClassifierCV
# ==========================================
print("Calibrating model probabilities (this may take a few seconds)...")

# Wrap the best model in CalibratedClassifierCV
# method='sigmoid' is generally robust for SVM and oversampled datasets
# cv=5 means using 5-fold cross-validation to refit and calibrate
calibrated_model = CalibratedClassifierCV(
    estimator=best_svm_model, 
    method='sigmoid', 
    cv=5, 
    n_jobs=-1
)

# Fit the calibrated model on the training set
calibrated_model.fit(X_train, y_train)
print("✅ Model probability calibration completed!")

# ==========================================
# 5. Evaluate calibration effect (Brier Score)
# ==========================================
# Brier Score measures the mean squared error between predicted probabilities and true labels
prob_uncalibrated = best_svm_model.predict_proba(X_test)[:, 1]
prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

brier_uncalibrated = brier_score_loss(y_test, prob_uncalibrated)
brier_calibrated = brier_score_loss(y_test, prob_calibrated)

print("-" * 30)
print(f"Brier Score before calibration: {brier_uncalibrated:.4f}")
print(f"Brier Score after calibration: {brier_calibrated:.4f}")
print("-" * 30)

# ==========================================
# 6. Plot calibration curve (Reliability Diagram)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

# Plot the reference line for perfect calibration (diagonal)
ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# Plot the curve before calibration
CalibrationDisplay.from_predictions(
    y_test, 
    prob_uncalibrated, 
    n_bins=10, 
    name="Uncalibrated SVM", 
    ax=ax, 
    color='red',
    marker='o'
)

# Plot the curve after calibration
CalibrationDisplay.from_predictions(
    y_test, 
    prob_calibrated, 
    n_bins=10, 
    name="Calibrated SVM", 
    ax=ax, 
    color='blue',
    marker='s'
)

ax.set_title('Calibration Curve Comparison (Test Set)', fontsize=14)
ax.set_xlabel('Mean predicted probability', fontsize=12)
ax.set_ylabel('Fraction of positives', fontsize=12)
ax.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)

# Save and show the plot
plt.savefig('Calibration_Curve.png', bbox_inches='tight')
plt.show()

# ==========================================
# 7. Save the new calibrated model
# ==========================================
calibrated_model_filename = 'calibrated_svm_model.joblib'
joblib.dump(calibrated_model, calibrated_model_filename)
print(f"✅ Calibrated model successfully saved as: {calibrated_model_filename}")
print("Tip: You can now load this new model to replot the DCA curve!")