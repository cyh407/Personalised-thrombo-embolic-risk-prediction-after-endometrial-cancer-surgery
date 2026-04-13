import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# 1. Data loading and feature selection
# ==========================================
# Define the 4 selected features and the target variable
selected_features = ['Postop_D_Dimer', 'Clinical_Stage', 'Pathological_Grade', 'Age']
target_col = 'Thrombosis'

# Read training and test data
df_main = pd.read_excel(r'data/data.xlsx')
X_main = df_main[selected_features]
y_main = df_main[target_col]

# Read external validation set data
df_val = pd.read_excel(r'data/extra_test.xlsx')
X_val = df_val[selected_features]
y_val = df_val[target_col]

# ==========================================
# 2. Dataset splitting (Train 80%, Test 20%)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_main, y_main, test_size=0.2, random_state=42, stratify=y_main
)

print(f"Data loaded successfully:")
print(f"- Train set: {X_train.shape[0]} samples")
print(f"- Test set : {X_test.shape[0]} samples")
print(f"- Validation set: {X_val.shape[0]} samples\n")

# ==========================================
# 3. Build model pipeline and hyperparameter grid
# ==========================================
# Use imblearn's Pipeline:
# RandomOverSampler only takes effect during training (fit), skipped during testing/validation.
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', RandomOverSampler(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)) # probability=True must be enabled to calculate AUC
])

# Define hyperparameter search grid for SVM
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01]
}

# ==========================================
# 4. Model training and optimization (GridSearchCV)
# ==========================================
print("Performing grid search to optimize SVM on the training set, please wait...")
grid_search = GridSearchCV(
    pipeline, 
    param_grid=param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1
)

# Fit and find the best parameters only on the training set
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"✅ Best parameter combination: {grid_search.best_params_}\n")

# ==========================================
# 5. Model evaluation function
# ==========================================
def evaluate_model(model, X_data, y_true, dataset_name):
    """Calculate and print evaluation metrics"""
    # Predict classes and probabilities
    y_pred = model.predict(X_data)
    y_prob = model.predict_proba(X_data)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Print results
    print(f"--- {dataset_name} Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}\n")

# Output metrics for train, test, and validation sets sequentially
evaluate_model(best_model, X_train, y_train, "Train Set")
evaluate_model(best_model, X_test, y_test, "Test Set")
evaluate_model(best_model, X_val, y_val, "Validation Set")

# ==========================================
# 6. Save the best model
# ==========================================
model_filename = 'best_svm_model.joblib'
joblib.dump(best_model, model_filename)
print(f"✅ Best model (including scaler and SVM) successfully saved as: {model_filename}")