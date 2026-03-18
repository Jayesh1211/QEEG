import os
import time
import joblib
import pandas as pd
import numpy as np

# Classical ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb

# Quantum Imports (Qiskit)
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Quantum Imports (PennyLane & PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INITIALIZING MASTER TRAINING PIPELINE (ALL MODELS)")
print("="*80)

# ============================================================================
# HELPER: PROPER 50/50 DISTRIBUTION SAMPLER (For Quantum Models)
# ============================================================================
def get_balanced_subset(X, y, total_samples):
    """
    Ensures an exact 50/50 split of class 0 and class 1.
    """
    half = total_samples // 2
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    # Check if we have enough data to fulfill the request
    if len(idx_0) < half or len(idx_1) < half:
        actual_half = min(len(idx_0), len(idx_1))
        print(f"      [WARNING] Not enough data for {total_samples} samples. Using {actual_half*2} instead.")
        half = actual_half
        
    np.random.seed(42)
    sel_0 = np.random.choice(idx_0, half, replace=False)
    sel_1 = np.random.choice(idx_1, half, replace=False)
    
    selected_indices = np.concatenate([sel_0, sel_1])
    np.random.shuffle(selected_indices)
    
    return X[selected_indices], y[selected_indices]

# ============================================================================
# 1. CLASSICAL MODEL DEFINITIONS (UNCHANGED)
# ============================================================================

class EEGXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, scale_pos_weight=1.5, eval_metric='logloss',
            tree_method='hist', random_state=42
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class EEGLightGBM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            num_leaves=50, min_child_samples=20, subsample=0.8,
            class_weight='balanced', random_state=42, verbose=-1
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class EEGRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=700, max_depth=30, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced', n_jobs=-1, random_state=42
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class EEGClassicalSVM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class EEGVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, scale_pos_weight=1.5, random_state=42, eval_metric='logloss')
        lgb_clf = lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)
        rf_clf = RandomForestClassifier(n_estimators=700, max_depth=30, class_weight='balanced', n_jobs=-1, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('rf', rf_clf)],
            voting='soft', weights=[2, 2, 1]
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class EEGStackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        base_models = [
            ('xgb', xgb.XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.05, random_state=42, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=400, max_depth=9, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=25, class_weight='balanced', n_jobs=-1, random_state=42))
        ]
        meta_learner = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
        
        self.model = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=5, n_jobs=-1)
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

# ============================================================================
# 2. QUANTUM MODEL DEFINITIONS (FIXED PICKLING & BALANCING)
# ============================================================================

class EEGQSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_dimension=12, max_samples=1000):
        self.feature_dimension = feature_dimension
        self.max_samples = max_samples
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if len(X_scaled) > self.max_samples:
            print(f"      [QSVM] Balancing training data to exact {self.max_samples} samples (50/50)...")
            X_scaled, y = get_balanced_subset(X_scaled, y, self.max_samples)
        else:
            print(f"      [QSVM] Balancing existing {len(X_scaled)} samples (50/50)...")
            X_scaled, y = get_balanced_subset(X_scaled, y, len(X_scaled))

        feature_map = ZZFeatureMap(feature_dimension=self.feature_dimension, reps=2, entanglement='linear')
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        self.model = QSVC(quantum_kernel=kernel)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# --- VQC Helper Network ---
class HybridVQC_TorchModule(nn.Module):
    def __init__(self, n_qubits, n_layers, use_gpu):
        super().__init__()
        self.n_qubits = n_qubits
        
        dev_name = "lightning.gpu" if (use_gpu and torch.cuda.is_available()) else "default.qubit"
        try:
            dev = qml.device(dev_name, wires=n_qubits)
        except:
            dev = qml.device("default.qubit", wires=n_qubits)
            
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.RZ(inputs[i] * 0.5, wires=i)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                    qml.RX(weights[layer, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))
            
        self.quantum_circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.post_quantum = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(8, 1), nn.Sigmoid()
        )
        
    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        q_outs = [self.quantum_circuit(x[i], self.weights) for i in range(x.shape[0])]
        q_tensor = torch.stack(q_outs).unsqueeze(-1).float()
        return self.post_quantum(q_tensor).squeeze(-1)

class EEGVQC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_layers=3, epochs=30, batch_size=32, max_samples=1000, use_gpu=True):
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        self.n_qubits = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.n_qubits = X.shape[1]
        
        if len(X_scaled) > self.max_samples:
            print(f"      [VQC] Balancing training data to exact {self.max_samples} samples (50/50)...")
            X_scaled, y = get_balanced_subset(X_scaled, y, self.max_samples)
        else:
            print(f"      [VQC] Balancing existing {len(X_scaled)} samples (50/50)...")
            X_scaled, y = get_balanced_subset(X_scaled, y, len(X_scaled))

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        self.model = HybridVQC_TorchModule(self.n_qubits, self.n_layers, self.use_gpu).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        print(f"      [VQC] Starting PyTorch training loop on {self.device}...")
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_t), self.batch_size):
                bx, by = X_t[i:i+self.batch_size], y_t[i:i+self.batch_size]
                optimizer.zero_grad()
                outputs = self.model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            preds = (outputs > 0.5).float().cpu().numpy()
        return preds.astype(int)

    # Pickling fix for PyTorch/PennyLane
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            self.model.to("cpu")
            state['torch_weights'] = self.model.state_dict()
            state['model'] = None 
        return state

    def __setstate__(self, state):
        torch_weights = state.pop('torch_weights', None)
        self.__dict__.update(state)
        if torch_weights is not None and getattr(self, 'n_qubits', None) is not None:
            self.device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
            self.model = HybridVQC_TorchModule(self.n_qubits, self.n_layers, self.use_gpu)
            self.model.load_state_dict(torch_weights)
            self.model.to(self.device)

# ============================================================================
# 3. DATA LOADING & PIPELINE EXECUTION
# ============================================================================

def main():
    FILE_PATH = "new_data_EEG_CNN_8Features_D4D5.csv"
    SAVE_DIR = "trained_api_models"
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"\nLoading data from {FILE_PATH}...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Cannot find {FILE_PATH}. Make sure it is in the same directory.")
        return

    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.\n")

    # Extract exactly 200 balanced samples for evaluating Quantum models quickly
    print(f"[SETUP] Preparing 200 balanced TEST samples specifically for Quantum Evaluation...")
    X_test_200, y_test_200 = get_balanced_subset(X_test, y_test, 200)

    # Define all models
    models_to_train = {
        "Classical_SVM": EEGClassicalSVM(),
        "XGBoost": EEGXGBoost(),
        "LightGBM": EEGLightGBM(),
        "RandomForest": EEGRandomForest(),
        "Voting_Ensemble": EEGVotingEnsemble(),
        "Stacking_Ensemble": EEGStackingEnsemble(),
        "QSVM": EEGQSVM(feature_dimension=X.shape[1], max_samples=1000),
        "VQC": EEGVQC(n_layers=3, epochs=30, max_samples=1000, use_gpu=True)
    }

    results = {}

    for name, clf in models_to_train.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        
        # Fit Model
        clf.fit(X_train, y_train)
        
        # Evaluate 
        if name in ["QSVM", "VQC"]:
            # Evaluate quantum models on the smaller 200-sample balanced subset
            preds = clf.predict(X_test_200)
            acc = accuracy_score(y_test_200, preds)
            t_elapsed = time.time() - t0
            print(f"    ✓ Accuracy (on 200 balanced samples): {acc*100:.2f}% (Time: {t_elapsed:.1f}s)")
        else:
            # Evaluate classical models on the full test set
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            t_elapsed = time.time() - t0
            print(f"    ✓ Accuracy (on full test set): {acc*100:.2f}% (Time: {t_elapsed:.1f}s)")
        
        results[name] = acc
        
        # Save Model
        save_path = os.path.join(SAVE_DIR, f"{name}.pkl")
        joblib.dump(clf, save_path)
        print(f"    ✓ Saved to {save_path}")

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20s} | {acc*100:.2f}%")
        
    print(f"\nAll models are packaged and saved in the '{SAVE_DIR}/' folder.")

if __name__ == "__main__":
    main()
