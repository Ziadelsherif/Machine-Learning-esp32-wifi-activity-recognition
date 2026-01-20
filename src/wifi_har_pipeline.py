import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Pipeline Configuration
# ==========================================
CONFIG = {
    'sampling_rate': 50,       # Hz (Typical for Wi-Fi sampling)
    'window_size_sec': 2.0,    # Seconds per window
    'overlap': 0.5,            # 50% overlap
    'seed': 42,
    'data_path': 'data/wifi_har_data.csv',
    'model_save_path': 'models/wifi_har_model.pkl'
}

# Ensure reproducibility
np.random.seed(CONFIG['seed'])

# ==========================================
# 2. Data Acquisition / Simulation
# ==========================================
def generate_synthetic_data(duration_sec=300):
    """
    Generates synthetic Wi-Fi RSSI data for Walking, Standing, and Jumping.
    In a real scenario, this would be replaced by: df = pd.read_csv('dataset.csv')
    
    Activities:
    - Standing: Low variance, stable mean.
    - Walking: Periodic variation (sine wave) + noise.
    - Jumping: High variance, spikes.
    """
    print("Generating synthetic dataset...")
    t = np.linspace(0, duration_sec, int(duration_sec * CONFIG['sampling_rate']))
    
    data = []
    
    # Simulate Standing
    rssi_stand = -50 + np.random.normal(0, 1.0, len(t))
    df_stand = pd.DataFrame({'Timestamp': t, 'RSSI': rssi_stand, 'Activity': 'Standing'})
    
    # Simulate Walking (Periodic fading 2Hz + Noise)
    rssi_walk = -55 + 5 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 2.0, len(t))
    df_walk = pd.DataFrame({'Timestamp': t, 'RSSI': rssi_walk, 'Activity': 'Walking'})
    
    # Simulate Jumping ( Bursts + High Noise)
    # Spikes every 1.5 seconds approx
    rssi_jump = -60 + np.random.normal(0, 3.0, len(t))
    spikes = np.where(np.random.rand(len(t)) > 0.95, 10, 0) # Random spikes
    rssi_jump += spikes
    df_jump = pd.DataFrame({'Timestamp': t, 'RSSI': rssi_jump, 'Activity': 'Jumping'})
    
    # Combine
    df = pd.concat([df_stand, df_walk, df_jump], ignore_index=True)
    
    # Save to CSV for realism
    os.makedirs(os.path.dirname(CONFIG['data_path']), exist_ok=True)
    df.to_csv(CONFIG['data_path'], index=False)
    print(f"Dataset generated and saved to {CONFIG['data_path']}")
    return df

def load_data(path):
    if not os.path.exists(path):
        return generate_synthetic_data()
    # Load data
    print(f"Loading data from {CONFIG['data_path']}...")
    data = pd.read_csv(CONFIG['data_path'])
    
    # Debug info
    print("Data types before cleaning:")
    print(data.dtypes)

    # Force RSSI to numeric, coercing errors to NaN (handles any accidental strings)
    data['RSSI'] = pd.to_numeric(data['RSSI'], errors='coerce')
    
    # Drop any rows with NaN RSSI or empty Activity
    initial_len = len(data)
    data.dropna(subset=['RSSI', 'Activity'], inplace=True)
    if len(data) < initial_len:
        print(f"Dropped {initial_len - len(data)} invalid rows.")

    # Convert Activity to string just in case
    data['Activity'] = data['Activity'].astype(str)
    return data

# ==========================================
# 3. Preprocessing & Windowing
# ==========================================
def create_windows(df, window_size, overlap):
    """
    Splits time-series data into overlapping windows.
    Returns: X (n_windows, window_points), y (n_windows)
    """
    print("Segmenting data into windows...")
    window_points = int(window_size * CONFIG['sampling_rate'])
    step_size = int(window_points * (1 - overlap))
    
    X = []
    y = []
    
    # Process each activity separately to avoid mixing labels in a window
    for activity in df['Activity'].unique():
        act_data = df[df['Activity'] == activity]['RSSI'].values
        
        for i in range(0, len(act_data) - window_points, step_size):
            window = act_data[i : i + window_points]
            X.append(window)
            y.append(activity)
            
    return np.array(X), np.array(y)

# ==========================================
# 4. Feature Extraction
# ==========================================
def extract_features(windows):
    """
    Extracts time-domain and frequency-domain features for each window.
    Features: Mean, Std, Min, Max, Energy, Dominant Freq, Spectral Entropy
    """
    print("Extracting features...")
    features = []
    
    for w in windows:
        # Time-domain
        mean = np.mean(w)
        std = np.std(w)
        var = np.var(w)
        min_val = np.min(w)
        max_val = np.max(w)
        energy = np.sum(w**2) / len(w)
        
        # Frequency-domain (FFT)
        freqs, psd = welch(w, fs=CONFIG['sampling_rate'])
        dominant_freq = freqs[np.argmax(psd)]
        
        # Spectral Entropy
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        features.append([mean, std, var, min_val, max_val, energy, dominant_freq, spectral_entropy])
        
    feature_names = ['Mean', 'Std', 'Variance', 'Min', 'Max', 'Energy', 'Dominant_Freq', 'Spectral_Entropy']
    return pd.DataFrame(features, columns=feature_names)

# ==========================================
# 5. Model Training & Evaluation
# ==========================================
def train_evaluate_pipeline(X_features, y_labels):
    print("Training models...")
    
    # Encoder labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_enc, test_size=0.3, random_state=CONFIG['seed'])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['seed'])
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    # Evaluation
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\nRandom Forest Accuracy: {acc_rf:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=le.classes_))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rf.feature_importances_, y=X_features.columns)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    print("Feature importance saved to feature_importance.png")

    return rf, scaler, le

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    df = load_data(CONFIG['data_path'])
    
    # 2. Visualize Raw Data Sample
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=df.iloc[:500], x='Timestamp', y='RSSI', hue='Activity')
    plt.title('Raw RSSI Signal Segment')
    plt.savefig('raw_signal_sample.png')
    
    # 3. Windowing
    X_windows, y_labels = create_windows(df, CONFIG['window_size_sec'], CONFIG['overlap'])
    print(f"Created {len(X_windows)} windows.")
    
    # 4. Feature Extraction
    X_features = extract_features(X_windows)
    
    # 5. Train & Eval
    model, scaler, le = train_evaluate_pipeline(X_features, y_labels)
    
    print("\nPipeline Complete. Models and plots saved.")
