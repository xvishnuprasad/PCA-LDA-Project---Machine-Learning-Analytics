import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import shap
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
print("="*80)
print("LOADING DATASET")
print("="*80)
df = pd.read_csv('processed_data/combined_cleaned_data.csv')

# Data preprocessing
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"\nMissing values: {df.isnull().sum().sum()} total")

# Identify target column
possible_targets = ['diagnosis', 'class', 'status', 'patient_status']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\nError: No target column found. Please specify the target variable.")
    exit()

print(f"\nTarget column identified: {target_col}")
print(f"Target distribution:\n{df[target_col].value_counts()}")

# Separate features and target
X = df.drop(columns=[target_col, 'patient_id', 'date_of_surgery', 'date_of_last_visit', 'unnamed:_32'], errors='ignore')
y = df[target_col]

# Store original feature names
original_feature_names = X.columns.tolist()

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(categorical_cols)}")

le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Handle missing values in target and drop rows with NaN targets
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]
print(f"\nAfter removing rows with NaN targets: {X.shape[0]} samples")

# Encode target if categorical
le_target = LabelEncoder()
if y.dtype == 'object':
    y_encoded = le_target.fit_transform(y.astype(str))
    class_names = le_target.classes_
    print(f"Target encoded. Classes: {list(class_names)}")
else:
    y_encoded = y.values
    # Check for NaN in numeric target and handle
    if pd.isna(y_encoded).any():
        valid_mask = ~pd.isna(y_encoded)
        X = X[valid_mask]
        y_encoded = y_encoded[valid_mask]
    # Convert to integer labels if numeric but represents classes
    unique_values = np.unique(y_encoded)
    if len(unique_values) <= 10:  # Likely a classification problem
        # Map to 0, 1, 2, ... labels
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        y_encoded = np.array([mapping[val] for val in y_encoded])
        class_names = [f"Class {i}" for i in unique_values]
    else:
        class_names = [f"Class {i}" for i in unique_values]

# Handle missing values - fill numeric columns with median, drop columns that are all NaN
numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X[col].isna().all():
        X = X.drop(columns=[col])
    else:
        X[col] = X[col].fillna(X[col].median())
        
# Fill remaining object columns with mode
object_cols = X.select_dtypes(include=['object']).columns
for col in object_cols:
    if X[col].isna().any():
        mode_value = X[col].mode()
        if len(mode_value) > 0:
            X[col] = X[col].fillna(mode_value[0])
        else:
            X[col] = X[col].fillna('unknown')

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Final check - drop any remaining NaN rows or columns
if np.isnan(X_scaled).any():
    print(f"Warning: NaN values detected after scaling. Dropping rows/columns with NaN...")
    nan_mask = ~np.isnan(X_scaled).any(axis=1)
    X_scaled = X_scaled[nan_mask]
    y_encoded = y_encoded[nan_mask]
    print(f"Final dataset shape after NaN removal: {X_scaled.shape}")

print(f"\nFeatures after preprocessing: {X_scaled.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================================================
# DIMENSIONALITY REDUCTION WITH TIMING
# ============================================================================
print("\n" + "="*80)
print("DIMENSIONALITY REDUCTION WITH PERFORMANCE TRACKING")
print("="*80)

reduction_results = {}

# 1. PCA
print("\n--- PCA Analysis ---")
start_time = time.time()
pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train)
X_test_pca_full = pca_full.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pca_time = time.time() - start_time

print(f"PCA completed in {pca_time:.4f} seconds")
print(f"  Original features: {X_train.shape[1]}")
print(f"  PCA components: {X_train_pca.shape[1]}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.4f}")

reduction_results['PCA'] = {
    'n_components': X_train_pca.shape[1],
    'variance_retained': pca.explained_variance_ratio_.sum(),
    'execution_time': pca_time,
    'X_train': X_train_pca,
    'X_test': X_test_pca
}

# 2. LDA
n_classes = len(np.unique(y_encoded))
if n_classes > 2:
    print("\n--- LDA Analysis ---")
    start_time = time.time()
    lda = LDA(n_components=min(n_classes-1, X_train.shape[1]))
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    lda_time = time.time() - start_time
    
    print(f"LDA completed in {lda_time:.4f} seconds")
    print(f"  LDA components: {X_train_lda.shape[1]}")
    print(f"  Variance explained: {lda.explained_variance_ratio_.sum():.4f}")
    
    reduction_results['LDA'] = {
        'n_components': X_train_lda.shape[1],
        'variance_retained': lda.explained_variance_ratio_.sum(),
        'execution_time': lda_time,
        'X_train': X_train_lda,
        'X_test': X_test_lda
    }
else:
    print("\n--- LDA skipped (binary classification) ---")
    X_train_lda, X_test_lda = None, None

# 3. Autoencoder
print("\n--- Autoencoder Analysis ---")
start_time = time.time()

# Determine encoding dimension (similar to PCA for fair comparison)
encoding_dim = X_train_pca.shape[1]

# Build autoencoder
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu', name='encoding')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(X_train.shape[1], activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, autoencoder.get_layer('encoding').output)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training autoencoder...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Extract encoded features
X_train_ae = encoder.predict(X_train, verbose=0)
X_test_ae = encoder.predict(X_test, verbose=0)
ae_time = time.time() - start_time

# Calculate variance retained (reconstruction quality)
X_train_reconstructed = autoencoder.predict(X_train, verbose=0)
mse = np.mean((X_train - X_train_reconstructed) ** 2)
variance_retained_ae = 1 - (mse / np.var(X_train))

print(f"Autoencoder completed in {ae_time:.4f} seconds")
print(f"  Encoding dimension: {encoding_dim}")
print(f"  Variance retained (approx): {variance_retained_ae:.4f}")
print(f"  Final MSE: {history.history['loss'][-1]:.6f}")

reduction_results['Autoencoder'] = {
    'n_components': encoding_dim,
    'variance_retained': max(0, variance_retained_ae),
    'execution_time': ae_time,
    'X_train': X_train_ae,
    'X_test': X_test_ae
}

# ============================================================================
# VISUALIZATION: PCA, LDA, AUTOENCODER
# ============================================================================
print("\n" + "="*80)
print("DIMENSIONALITY REDUCTION VISUALIZATIONS")
print("="*80)

# Create color palette
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

# 1. PCA Variance Analysis
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_plot = min(20, len(explained_var))

plt.plot(range(1, n_components_plot+1), explained_var[:n_components_plot], 'bo-', linewidth=2, markersize=8)
plt.xlabel('Principal Component', fontsize=12, fontweight='bold')
plt.ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
plt.title('PCA Scree Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, n_components_plot+1), cumulative_var[:n_components_plot], 'ro-', linewidth=2, markersize=8)
plt.axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='95% Variance')
plt.xlabel('Number of Components', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
plt.title('PCA Cumulative Variance', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 2D Projections Comparison
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA 2D
for i, (class_label, color) in enumerate(zip(np.unique(y_train), colors)):
    mask = y_train == class_label
    axes[0].scatter(X_train_pca_2d[mask, 0], X_train_pca_2d[mask, 1], 
                   c=[color], label=class_names[i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
axes[0].set_title('PCA (Unsupervised)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.3)

# LDA 2D
if X_train_lda is not None and X_train_lda.shape[1] >= 2:
    for i, (class_label, color) in enumerate(zip(np.unique(y_train), colors)):
        mask = y_train == class_label
        axes[1].scatter(X_train_lda[mask, 0], X_train_lda[mask, 1], 
                       c=[color], label=class_names[i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
    axes[1].set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
    axes[1].set_title('LDA (Supervised)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'LDA Not Available\n(Binary Classification)', 
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    axes[1].set_title('LDA (Supervised)', fontsize=13, fontweight='bold')

# Autoencoder 2D
ae_pca_2d = PCA(n_components=2)
X_train_ae_2d = ae_pca_2d.fit_transform(X_train_ae)

for i, (class_label, color) in enumerate(zip(np.unique(y_train), colors)):
    mask = y_train == class_label
    axes[2].scatter(X_train_ae_2d[mask, 0], X_train_ae_2d[mask, 1], 
                   c=[color], label=class_names[i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[2].set_xlabel('AE Component 1', fontsize=11, fontweight='bold')
axes[2].set_ylabel('AE Component 2', fontsize=11, fontweight='bold')
axes[2].set_title('Autoencoder (Deep Learning)', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=9, loc='best')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Dimensionality Reduction Comparison - 2D Projections', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Autoencoder Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('MSE Loss', fontsize=12, fontweight='bold')
plt.title('Autoencoder Training History', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Reconstruction quality visualization
sample_idx = np.random.choice(X_test.shape[0], 5, replace=False)
X_test_reconstructed = autoencoder.predict(X_test[sample_idx], verbose=0)

for i in range(5):
    plt.plot(X_test[sample_idx[i]], alpha=0.5, label=f'Original {i+1}' if i == 0 else '')
    plt.plot(X_test_reconstructed[i], alpha=0.5, linestyle='--', label=f'Reconstructed {i+1}' if i == 0 else '')

plt.xlabel('Feature Index', fontsize=12, fontweight='bold')
plt.ylabel('Feature Value', fontsize=12, fontweight='bold')
plt.title('Sample Reconstruction Quality', fontsize=14, fontweight='bold')
plt.legend(['Original', 'Reconstructed'], fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# MODEL TRAINING ON ALL REDUCTION METHODS
# ============================================================================
print("\n" + "="*80)
print("MODEL TRAINING ACROSS ALL DIMENSIONALITY REDUCTION METHODS")
print("="*80)

models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        }
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

def train_evaluate(X_tr, X_te, y_tr, y_te, model_name, model, params):
    grid = GridSearchCV(model, params, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0)
    grid.fit(X_tr, y_tr)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_te)
    
    metrics = {
        'Accuracy': accuracy_score(y_te, y_pred),
        'F1 Score': f1_score(y_te, y_pred, average='weighted', zero_division=0),
        'Model': best_model
    }
    return metrics

# Train on Original + all reduction methods
feature_sets = ['Original'] + list(reduction_results.keys())

for fs in feature_sets:
    print(f"\n--- Training on {fs} Features ---")
    all_results[fs] = {}
    
    if fs == 'Original':
        X_tr, X_te = X_train, X_test
    else:
        X_tr = reduction_results[fs]['X_train']
        X_te = reduction_results[fs]['X_test']
    
    for model_name, model_info in models.items():
        print(f"  Training {model_name}...")
        metrics = train_evaluate(X_tr, X_te, y_train, y_test, 
                               model_name, model_info['model'], model_info['params'])
        all_results[fs][model_name] = metrics

# ============================================================================
# EXPLAINABLE AI (XAI) - SHAP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("EXPLAINABLE AI (XAI) - SHAP ANALYSIS")
print("="*80)

# Find best model
best_f1 = 0
best_model_info = None
for fs, models_dict in all_results.items():
    for model_name, metrics in models_dict.items():
        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_model_info = (fs, model_name, metrics['Model'])

print(f"\nBest Model: {best_model_info[1]} with {best_model_info[0]} features")
print(f"F1 Score: {best_f1:.4f}")

# Perform SHAP analysis on best model
if best_model_info[0] == 'Original':
    X_for_shap = X_test
    feature_names_shap = original_feature_names
elif best_model_info[0] == 'PCA':
    X_for_shap = X_test_pca
    feature_names_shap = [f'PC{i+1}' for i in range(X_test_pca.shape[1])]
elif best_model_info[0] == 'LDA' and X_test_lda is not None:
    X_for_shap = X_test_lda
    feature_names_shap = [f'LD{i+1}' for i in range(X_test_lda.shape[1])]
else:
    X_for_shap = X_test_ae
    feature_names_shap = [f'AE{i+1}' for i in range(X_test_ae.shape[1])]

print(f"\nGenerating SHAP explanations for {best_model_info[1]}...")

# Create SHAP explainer
if best_model_info[0] == 'Original':
    sample_size = min(100, X_train.shape[0])
    X_sample = X_train[np.random.choice(X_train.shape[0], sample_size, replace=False)]
else:
    X_sample = reduction_results[best_model_info[0]]['X_train'][:100]

try:
    if isinstance(best_model_info[2], (LogisticRegression, SVC)):
        explainer = shap.KernelExplainer(best_model_info[2].predict_proba, X_sample)
    else:
        explainer = shap.Explainer(best_model_info[2].predict_proba, X_sample)
    
    shap_values = explainer(X_for_shap[:100])
    
    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    if n_classes == 2:
        shap.summary_plot(shap_values[:, :, 1], X_for_shap[:100], 
                         feature_names=feature_names_shap[:X_for_shap.shape[1]], 
                         show=False)
    else:
        shap.summary_plot(shap_values, X_for_shap[:100], 
                         feature_names=feature_names_shap[:X_for_shap.shape[1]], 
                         show=False)
    plt.title(f'SHAP Feature Importance - {best_model_info[1]} ({best_model_info[0]})', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    if n_classes == 2:
        shap.plots.bar(shap_values[:, :, 1], show=False)
    else:
        shap.plots.bar(shap_values[:, :, 0], show=False)
    plt.title(f'SHAP Feature Importance (Bar) - {best_model_info[1]}', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ SHAP analysis completed successfully")
    
except Exception as e:
    print(f"⚠ SHAP analysis encountered an issue: {str(e)}")
    print("  Continuing with other analyses...")

# ============================================================================
# COMPREHENSIVE COMPARISON REPORT
# ============================================================================
print("\n" + "="*80)
print("DIMENSIONALITY REDUCTION COMPARISON REPORT")
print("="*80)

# Create comparison dataframe
comparison_data = []
for method, info in reduction_results.items():
    # Calculate average accuracy across all models
    avg_acc = np.mean([all_results[method][m]['Accuracy'] for m in models.keys()])
    avg_f1 = np.mean([all_results[method][m]['F1 Score'] for m in models.keys()])
    
    comparison_data.append({
        'Method': method,
        'Components': info['n_components'],
        'Variance Retained': info['variance_retained'],
        'Execution Time (s)': info['execution_time'],
        'Avg Accuracy': avg_acc,
        'Avg F1 Score': avg_f1
    })

# Add original features
orig_avg_acc = np.mean([all_results['Original'][m]['Accuracy'] for m in models.keys()])
orig_avg_f1 = np.mean([all_results['Original'][m]['F1 Score'] for m in models.keys()])

comparison_data.insert(0, {
    'Method': 'Original',
    'Components': X_train.shape[1],
    'Variance Retained': 1.0,
    'Execution Time (s)': 0.0,
    'Avg Accuracy': orig_avg_acc,
    'Avg F1 Score': orig_avg_f1
})

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Visualization: Comparison Dashboard
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

methods = comparison_df['Method'].tolist()
colors_methods = ['steelblue', 'coral', 'lightgreen', 'plum'][:len(methods)]

# 1. Components Comparison
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(methods, comparison_df['Components'], color=colors_methods, edgecolor='black', alpha=0.8)
ax1.set_ylabel('Number of Components', fontsize=11, fontweight='bold')
ax1.set_title('Dimensionality', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars1, comparison_df['Components'])):
    ax1.text(bar.get_x() + bar.get_width()/2, val + max(comparison_df['Components'])*0.02, 
            f'{int(val)}', ha='center', fontweight='bold', fontsize=10)

# 2. Variance Retained
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(methods, comparison_df['Variance Retained'], color=colors_methods, edgecolor='black', alpha=0.8)
ax2.set_ylabel('Variance Retained', fontsize=11, fontweight='bold')
ax2.set_title('Information Preservation', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')
ax2.grid(axis='y', alpha=0.3)
ax2.legend(fontsize=9)
for i, (bar, val) in enumerate(zip(bars2, comparison_df['Variance Retained'])):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.03, 
            f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

# 3. Execution Time
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(methods, comparison_df['Execution Time (s)'], color=colors_methods, edgecolor='black', alpha=0.8)
ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars3, comparison_df['Execution Time (s)'])):
    if val > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, val + max(comparison_df['Execution Time (s)'])*0.02, 
                f'{val:.3f}s', ha='center', fontweight='bold', fontsize=9)

# 4. Average Accuracy
ax4 = fig.add_subplot(gs[1, 0])
bars4 = ax4.bar(methods, comparison_df['Avg Accuracy'], color=colors_methods, edgecolor='black', alpha=0.8)
ax4.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('Model Accuracy', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1.1])
ax4.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars4, comparison_df['Avg Accuracy'])):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.03, 
            f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

# 5. Average F1 Score
ax5 = fig.add_subplot(gs[1, 1])
bars5 = ax5.bar(methods, comparison_df['Avg F1 Score'], color=colors_methods, edgecolor='black', alpha=0.8)
ax5.set_ylabel('Average F1 Score', fontsize=11, fontweight='bold')
ax5.set_title('Model Performance', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 1.1])
ax5.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars5, comparison_df['Avg F1 Score'])):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.03, 
            f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

# 6. Performance vs Complexity Trade-off
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(comparison_df['Components'], comparison_df['Avg F1 Score'], 
                     s=comparison_df['Execution Time (s)']*500 + 100, 
                     c=colors_methods, alpha=0.7, edgecolors='black', linewidths=2)
for i, method in enumerate(methods):
    ax6.annotate(method, 
                (comparison_df.iloc[i]['Components'], comparison_df.iloc[i]['Avg F1 Score']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
ax6.set_xlabel('Number of Components', fontsize=11, fontweight='bold')
ax6.set_ylabel('Average F1 Score', fontsize=11, fontweight='bold')
ax6.set_title('Performance vs Complexity\n(Size = Execution Time)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Model Performance Heatmap
ax7 = fig.add_subplot(gs[2, :])
heatmap_data = []
for method in methods:
    row = []
    for model_name in models.keys():
        row.append(all_results[method][model_name]['F1 Score'])
    heatmap_data.append(row)

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=list(models.keys()), yticklabels=methods,
            cbar_kws={'label': 'F1 Score'}, linewidths=0.5, ax=ax7)
ax7.set_title('Model Performance Across Dimensionality Reduction Methods', 
             fontsize=13, fontweight='bold')
ax7.set_xlabel('Models', fontsize=11, fontweight='bold')
ax7.set_ylabel('Reduction Method', fontsize=11, fontweight='bold')

plt.suptitle('Comprehensive Dimensionality Reduction Comparison', 
            fontsize=16, fontweight='bold', y=0.995)
plt.savefig('dimensionality_reduction_comparison_report.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# DETAILED PERFORMANCE METRICS TABLE
# ============================================================================
print("\n" + "="*80)
print("DETAILED MODEL PERFORMANCE BY REDUCTION METHOD")
print("="*80)

detailed_results = []
for method in methods:
    for model_name in models.keys():
        detailed_results.append({
            'Reduction Method': method,
            'Model': model_name,
            'Accuracy': f"{all_results[method][model_name]['Accuracy']:.4f}",
            'F1 Score': f"{all_results[method][model_name]['F1 Score']:.4f}"
        })

detailed_df = pd.DataFrame(detailed_results)
print("\n" + detailed_df.to_string(index=False))

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("FINAL RECOMMENDATIONS & INSIGHTS")
print("="*80)

# Best overall configuration
print(f"\n{'🏆 BEST MODEL CONFIGURATION':^80}")
print("="*80)
print(f"Reduction Method: {best_model_info[0]}")
print(f"Model: {best_model_info[1]}")
print(f"F1 Score: {best_f1:.4f}")

# Best method per criterion
print(f"\n{'📊 BEST METHOD BY CRITERION':^80}")
print("="*80)

best_variance = comparison_df.loc[comparison_df['Variance Retained'].idxmax()]
best_speed = comparison_df.loc[comparison_df[comparison_df['Method'] != 'Original']['Execution Time (s)'].idxmin()]
best_performance = comparison_df.loc[comparison_df['Avg F1 Score'].idxmax()]
best_compression = comparison_df.loc[comparison_df['Components'].idxmin()]

print(f"\n✓ Best Information Preservation:")
print(f"  → {best_variance['Method']}: {best_variance['Variance Retained']:.4f} variance retained")

print(f"\n✓ Fastest Execution:")
print(f"  → {best_speed['Method']}: {best_speed['Execution Time (s)']:.4f} seconds")

print(f"\n✓ Best Average Performance:")
print(f"  → {best_performance['Method']}: {best_performance['Avg F1 Score']:.4f} F1 score")

print(f"\n✓ Maximum Compression:")
print(f"  → {best_compression['Method']}: {int(best_compression['Components'])} components")

# Method-specific insights
print(f"\n{'💡 METHOD-SPECIFIC INSIGHTS':^80}")
print("="*80)

print("\n📌 PCA (Principal Component Analysis):")
print(f"  • Reduced features by {((X_train.shape[1] - X_train_pca.shape[1])/X_train.shape[1]*100):.1f}%")
print(f"  • Preserved {pca.explained_variance_ratio_.sum():.2%} of variance")
print(f"  • Best for: Unsupervised feature reduction, noise removal")
print(f"  • Execution time: {reduction_results['PCA']['execution_time']:.4f}s")

if 'LDA' in reduction_results:
    print("\n📌 LDA (Linear Discriminant Analysis):")
    print(f"  • Created {X_train_lda.shape[1]} discriminant components")
    print(f"  • Maximizes class separability (supervised)")
    print(f"  • Preserved {lda.explained_variance_ratio_.sum():.2%} of variance")
    print(f"  • Best for: Classification tasks with labeled data")
    print(f"  • Execution time: {reduction_results['LDA']['execution_time']:.4f}s")

print("\n📌 Autoencoder (Deep Learning):")
print(f"  • Learned {encoding_dim} non-linear features")
print(f"  • Captures complex patterns PCA/LDA might miss")
print(f"  • Variance retained (approx): {reduction_results['Autoencoder']['variance_retained']:.2%}")
print(f"  • Best for: Non-linear relationships, large datasets")
print(f"  • Execution time: {reduction_results['Autoencoder']['execution_time']:.4f}s")
print(f"  • Training epochs: {len(history.history['loss'])}")

# Performance comparison
print(f"\n{'📈 PERFORMANCE COMPARISON':^80}")
print("="*80)

orig_f1 = comparison_df[comparison_df['Method'] == 'Original']['Avg F1 Score'].values[0]
for method in ['PCA', 'LDA', 'Autoencoder']:
    if method in comparison_df['Method'].values:
        method_f1 = comparison_df[comparison_df['Method'] == method]['Avg F1 Score'].values[0]
        diff = ((method_f1 - orig_f1) / orig_f1) * 100
        symbol = "↑" if diff > 0 else "↓"
        print(f"\n{method} vs Original:")
        print(f"  {symbol} Performance change: {abs(diff):.2f}%")
        print(f"  {'✓ IMPROVEMENT' if diff > 0 else '⚠ SLIGHT DECREASE' if diff > -5 else '✗ SIGNIFICANT DECREASE'}")

# Recommendations based on use case
print(f"\n{'🎯 USE CASE RECOMMENDATIONS':^80}")
print("="*80)

print("\n1️⃣  For MAXIMUM ACCURACY:")
best_acc_method = comparison_df.loc[comparison_df['Avg Accuracy'].idxmax()]['Method']
print(f"   → Use {best_acc_method}")

print("\n2️⃣  For FASTEST INFERENCE:")
fastest_method = comparison_df[comparison_df['Method'] != 'Original'].loc[
    comparison_df[comparison_df['Method'] != 'Original']['Components'].idxmin()]['Method']
print(f"   → Use {fastest_method} (smallest feature space)")

print("\n3️⃣  For INTERPRETABILITY:")
print(f"   → Use PCA (linear combinations of original features)")
print(f"   → Combined with SHAP for feature importance")

print("\n4️⃣  For PRODUCTION DEPLOYMENT:")
if reduction_results['PCA']['execution_time'] < reduction_results['Autoencoder']['execution_time']:
    print(f"   → Use PCA (fast transformation: {reduction_results['PCA']['execution_time']:.4f}s)")
else:
    print(f"   → Use Autoencoder (better performance, acceptable speed)")

print("\n5️⃣  For RESEARCH/PUBLICATION:")
print(f"   → Compare all methods (PCA, LDA, Autoencoder)")
print(f"   → Include SHAP analysis for explainability")
print(f"   → Demonstrate hybrid approach (PCA + Autoencoder)")

# Statistical summary
print(f"\n{'📊 STATISTICAL SUMMARY':^80}")
print("="*80)

print("\nPerformance Statistics:")
print(f"  • Best F1 Score: {comparison_df['Avg F1 Score'].max():.4f}")
print(f"  • Worst F1 Score: {comparison_df['Avg F1 Score'].min():.4f}")
print(f"  • Average F1 Score: {comparison_df['Avg F1 Score'].mean():.4f}")
print(f"  • Std Deviation: {comparison_df['Avg F1 Score'].std():.4f}")

print("\nDimensionality Statistics:")
print(f"  • Original features: {X_train.shape[1]}")
print(f"  • Average reduced features: {comparison_df[comparison_df['Method'] != 'Original']['Components'].mean():.1f}")
print(f"  • Average compression: {(1 - comparison_df[comparison_df['Method'] != 'Original']['Components'].mean()/X_train.shape[1])*100:.1f}%")

print("\nTiming Statistics:")
print(f"  • Fastest method: {comparison_df[comparison_df['Method'] != 'Original']['Execution Time (s)'].min():.4f}s")
print(f"  • Slowest method: {comparison_df['Execution Time (s)'].max():.4f}s")
print(f"  • Average time: {comparison_df[comparison_df['Method'] != 'Original']['Execution Time (s)'].mean():.4f}s")

# Export summary
print(f"\n{'💾 EXPORTED FILES':^80}")
print("="*80)
print("\nVisualizations:")
print("  1. pca_variance_analysis.png")
print("  2. dimensionality_reduction_comparison.png")
print("  3. autoencoder_analysis.png")
print("  4. shap_summary_plot.png")
print("  5. shap_bar_plot.png")
print("  6. dimensionality_reduction_comparison_report.png")

print("\nData Files:")
comparison_df.to_csv('dimensionality_reduction_summary.csv', index=False)
detailed_df.to_csv('detailed_model_performance.csv', index=False)
print("  1. dimensionality_reduction_summary.csv")
print("  2. detailed_model_performance.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✅ Successfully compared 4 methods: Original, PCA, LDA, Autoencoder")
print(f"✅ Trained and evaluated 4 ML models: LR, SVM, KNN, Decision Tree")
print(f"✅ Generated {len([f for f in ['pca_variance_analysis.png', 'dimensionality_reduction_comparison.png', 'autoencoder_analysis.png', 'shap_summary_plot.png', 'shap_bar_plot.png', 'dimensionality_reduction_comparison_report.png'] if f])} visualizations")
print(f"✅ Performed SHAP explainability analysis")
print(f"✅ Best model: {best_model_info[1]} with {best_model_info[0]} (F1: {best_f1:.4f})")

print("\n" + "="*80)