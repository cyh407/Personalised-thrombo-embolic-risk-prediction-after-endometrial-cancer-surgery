import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine learning related libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Import 6 classifiers
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Import imblearn for handling imbalanced data and building safe Pipelines
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
np.random.seed(42)

# 1. Load data (Please replace with your actual file path)
df = pd.read_excel('data/data.xlsx')

# Define target and discrete columns based on the translated features
target_col = 'Thrombosis'
discrete_cols = [
    'Menopause', 'Diabetes', 'Hypertension', 'Lymph_Node_Metastasis', 
    'Vascular_Metastasis', 'Postop_Lower_Limb_Massage', 'Anticoagulant_Usage', 
    'Hemostatic_Usage', 'Central_Venous_Catheter'
]

# Automatically identify continuous variables (excluding target and discrete variables)
continuous_cols = [col for col in df.columns if col not in discrete_cols and col != target_col]

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Split dataset: 80% train, 20% test (using stratified sampling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set size: {X_train.shape}, Positive sample ratio: {y_train.mean():.2f}")
print(f"Test set size: {X_test.shape}, Positive sample ratio: {y_test.mean():.2f}\n")

# 3. Data preprocessing: Standardize continuous variables, keep discrete variables unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_cols),
        ('cat', 'passthrough', discrete_cols)
    ])

# 4. Define models and hyperparameter grids
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

# Cross-validation strategy: Stratified 5-fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = {}

# 5. Loop to train each model and perform grid search
for model_name, mp in models_and_params.items():
    print("="*40)
    print(f"Training and tuning model: {model_name}")
    
    # Core: Use imblearn's Pipeline
    # Workflow: Preprocessing -> RandomOverSampler -> Classifier
    # This ensures oversampling is only applied to the training folds during CV, preventing data leakage
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
        scoring='roc_auc', # ROC-AUC is a better metric for imbalanced data
        n_jobs=-1
    )
    
    # Fit on the training set
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Highest CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # 6. Validate on the test set (No augmentation on test set, Pipeline handles it automatically)
    y_pred = grid_search.predict(X_test)
    
    # Evaluation metrics
    print(f"\n[{model_name}] Test set performance:")
    print(classification_report(y_test, y_pred))
    
    # If the model supports probability prediction, calculate ROC-AUC on the test set
    if hasattr(grid_search.best_estimator_.named_steps['classifier'], "predict_proba"):
        y_prob = grid_search.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"Test set ROC-AUC: {auc_score:.4f}")
    print("="*40 + "\n")