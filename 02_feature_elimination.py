import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

np.random.seed(42)

# ==========================================
# New: Define Bootstrap function to calculate AUC 95% CI
# ==========================================
def calculate_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.05, random_state=42):
    """
    Calculate ROC-AUC confidence interval using Bootstrap method
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    for i in range(n_bootstraps):
        # Sampling with replacement
        indices = rng.randint(0, len(y_scores), len(y_scores))
        
        # Ensure the sampled data contains both positive and negative classes
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Calculate percentiles (e.g., for alpha=0.05, take 2.5% and 97.5%)
    lower_bound = np.percentile(sorted_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(sorted_scores, 100 * (1 - alpha / 2))
    
    return lower_bound, upper_bound

# ==========================================
# 1. Define feature list and known discrete variables
# ==========================================
# Ordered according to feature importance, from most to least important
ordered_features = [
    'Postop_D_Dimer', 'Clinical_Stage', 'Pathological_Grade', 'Age', 'Postop_Bed_Rest_Time',
    'Fibrinogen_Level', 'APTT', 'Postop_Lower_Limb_Massage', 'Max_Tumor_Diameter',
    'WBC_Count', 'Intraop_Blood_Loss_mL', 'Anticoagulant_Usage', 'Hematocrit', 'Hypertension',
    'RBC_Count', 'Central_Venous_Catheter', 'Operative_Time_min', 'Menopause', 'Platelet_Count',
    'Diabetes', 'Thrombin_Time', 'Intraop_Blood_Transfusion_mL', 'BMI', 'Intraop_Fluid_Infusion',
    'Vascular_Metastasis', 'Hemostatic_Usage', 'Lymph_Node_Metastasis'
]

# Previously defined discrete variables (used to distinguish preprocessing methods)
known_discrete_cols = [
    'Menopause', 'Diabetes', 'Hypertension', 'Lymph_Node_Metastasis', 
    'Vascular_Metastasis', 'Postop_Lower_Limb_Massage', 'Anticoagulant_Usage', 
    'Hemostatic_Usage', 'Central_Venous_Catheter'
]
target_col = 'Thrombosis'

# --- Load dataset ---
# Note: Ensure the path data/data.xlsx is correct
df = pd.read_excel('data/data.xlsx')
# --------------------------------------------------------

# ==========================================
# 2. Split global dataset (keep test set consistent)
# ==========================================
X = df[ordered_features]
y = df[target_col]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. Define models and simplified hyperparameter grids
# ==========================================
models_and_params = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {'classifier__C': [0.01, 0.1, 1, 10]}
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
    },
    'NearestCentroid': {
        'model': NearestCentroid(),
        'params': {'classifier__metric': ['euclidean', 'manhattan']}
    },
    'BernoulliNB': {
        'model': BernoulliNB(),
        'params': {'classifier__alpha': [0.1, 0.5, 1.0]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 5, 10]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.1, 1.0]}
    }
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store pure AUC scores for each model under different feature counts (for plotting)
performance_history = {model_name: [] for model_name in models_and_params.keys()}
# Store formatted "AUC (95% CI)" strings (for final table output)
performance_history_ci = {model_name: [] for model_name in models_and_params.keys()}
feature_counts = []

# ==========================================
# 4. Core process: Stepwise feature elimination
# ==========================================
print("Starting backward feature elimination experiment...\n")

# Start with all features, remove the least important one each time, until 1 feature is left
for num_features in range(len(ordered_features), 0, -1):
    # Slice currently retained features (those at the front are more important)
    current_features = ordered_features[:num_features]
    feature_counts.append(num_features)
    
    eliminated = ordered_features[num_features:] if num_features < len(ordered_features) else 'None'
    print(f"[{num_features}/{len(ordered_features)}] Current feature count: {num_features}. Eliminated feature: {eliminated}")
    
    # Get current training and test subsets
    X_train_sub = X_train_full[current_features]
    X_test_sub = X_test_full[current_features]
    
    # Dynamically distinguish discrete and continuous variables in current features
    current_discrete = [col for col in current_features if col in known_discrete_cols]
    current_continuous = [col for col in current_features if col not in known_discrete_cols]
    
    # Dynamically update preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), current_continuous),
            ('cat', 'passthrough', current_discrete)
        ])
    
    # Train and evaluate each model
    for model_name, mp in models_and_params.items():
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampler', RandomOverSampler(random_state=42)),
            ('classifier', mp['model'])
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid=mp['params'], 
            cv=cv_strategy, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_sub, y_train)
        best_model = grid_search.best_estimator_
        
        # Calculate ROC-AUC on the test set
        classifier_step = best_model.named_steps['classifier']
        
        # Check if the model supports probability output (NearestCentroid does not)
        if hasattr(classifier_step, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_sub)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            # Calculate confidence interval
            ci_lower, ci_upper = calculate_auc_ci(y_test, y_prob)
        else:
            # For NearestCentroid, only predicted labels can be used to calculate approximate AUC
            y_pred = best_model.predict(X_test_sub)
            auc_score = roc_auc_score(y_test, y_pred)
            # Calculate confidence interval
            ci_lower, ci_upper = calculate_auc_ci(y_test, y_pred)
            
        # Record pure numerical values for plotting
        performance_history[model_name].append(auc_score)
        
        # Record formatted strings with CI for table display
        ci_str = f"{auc_score:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
        performance_history_ci[model_name].append(ci_str)

print("\nExperiment completed! Generating performance trend chart...")

# ==========================================
# 5. Result visualization: Plot relationship between feature count and ROC-AUC
# ==========================================
plt.figure(figsize=(12, 8))

for model_name, auc_scores in performance_history.items():
    # Note: feature_counts is ordered from large to small, reverse it for plotting
    plt.plot(feature_counts[::-1], auc_scores[::-1], marker='o', label=model_name, linewidth=2)

plt.title('ROC-AUC Performance with Stepwise Feature Elimination', fontsize=16)
plt.xlabel('Number of Retained Features (Cumulative from most important)', fontsize=14)
plt.ylabel('Test Set ROC-AUC', fontsize=14)
plt.xticks(range(1, len(ordered_features) + 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Print final result table to view specific values and their 95% CI
results_df = pd.DataFrame(performance_history_ci, index=[f"{i} features" for i in feature_counts])
print("\nDetailed ROC-AUC and 95% CI data for each model under different feature counts:")
# Set pandas display options to ensure long strings are fully displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df)