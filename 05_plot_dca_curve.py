import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
# 3. Load the calibrated model and predict probabilities
# ==========================================
print("Loading calibrated SVM model...")
calibrated_model = joblib.load('calibrated_svm_model.joblib')

# Get predicted probabilities on the test set (positive class)
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# ==========================================
# 4. Define function to calculate DCA (Net Benefit)
# ==========================================
def calculate_net_benefit(y_true, y_proba, thresholds):
    """
    Calculate Net Benefit at different thresholds
    """
    net_benefits = []
    treat_all_nbs = []
    
    N = len(y_true)           # Total number of samples
    P = np.sum(y_true)        # Number of true positive samples
    
    for pt in thresholds:
        # If threshold is 1.0, denominator becomes 0, set to 0 directly
        if pt == 1.0:
            net_benefits.append(0.0)
            treat_all_nbs.append(0.0)
            continue
            
        # Classify based on the current threshold pt
        preds = (y_proba >= pt).astype(int)
        
        # Calculate True Positives (TP) and False Positives (FP)
        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))
        
        # Odds multiplier (weight brought by risk threshold)
        multiplier = pt / (1 - pt)
        
        # 1. Calculate the net benefit of the model
        nb = (TP / N) - (FP / N) * multiplier
        net_benefits.append(nb)
        
        # 2. Calculate the net benefit of "Treat All"
        # Assume all patients are predicted as positive
        TP_all = P
        FP_all = N - P
        nb_all = (TP_all / N) - (FP_all / N) * multiplier
        treat_all_nbs.append(nb_all)
            
    return np.array(net_benefits), np.array(treat_all_nbs)

# ==========================================
# 5. Calculate DCA data
# ==========================================
print("Calculating DCA curve data...")
# Set threshold range, from 0.01 to 0.99, step 0.01
thresholds = np.arange(0.01, 1.00, 0.01)

# Calculate net benefit
nb_model, nb_all = calculate_net_benefit(y_test.values, y_pred_proba, thresholds)

# The net benefit of "Treat None" is always 0
nb_none = np.zeros_like(thresholds)

# ==========================================
# 6. Plot DCA curve
# ==========================================
plt.figure(figsize=(10, 6), dpi=300)

# Plot the three core curves
plt.plot(thresholds, nb_model, color='red', linewidth=2, label='Calibrated SVM')
plt.plot(thresholds, nb_all, color='gray', linestyle='--', linewidth=1.5, label='Treat All')
plt.plot(thresholds, nb_none, color='black', linewidth=1.5, label='Treat None')

# Optimize chart display
plt.title('Decision Curve Analysis (DCA) - Calibrated Model', fontsize=16)
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)

# Set axis limits
# X-axis is usually 0 to 1
plt.xlim([0.0, 1.0])
# Y-axis lower limit slightly below 0, upper limit is max of "Treat All" plus margin
max_prevalence = np.max(nb_all)
plt.ylim([-0.05, max_prevalence + 0.05])

plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Save and show the plot
plt.savefig('DCA_Curve_Calibrated.png', bbox_inches='tight')
print("✅ DCA curve plotted successfully! Saved as 'DCA_Curve_Calibrated.png'")
plt.show()