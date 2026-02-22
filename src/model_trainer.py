"""
Entraînement du modèle IA pour la prédiction de robustesse des mots de passe
Auteur : Abderemane Attoumani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


# 1. Chargement et préparation

def load_and_prepare(csv_path):
    """Charge le dataset et prépare les features pour le modèle."""
    
    df = pd.read_csv(csv_path)
    print(f"Dataset chargé : {len(df):,} mots de passe")
    
    # Features utilisées pour l'entraînement
    # On exclut : password (texte), strength_name (texte), crack_time (trop corrélé)
    FEATURES = [
    # Longueur seulement (pas charset_size qui encode la complexité)
    'length',
    
    # Compteurs bruts (ce que le modèle OBSERVE)
    'num_lowercase', 'num_uppercase', 'num_digits', 'num_special',
    'num_unique_chars',
    
    # Patterns booléens (observations directes)
    'has_sequential_nums', 'has_sequential_alpha', 'has_keyboard_pattern',
    'is_all_same_char', 'has_date_pattern',
    'starts_with_capital', 'ends_with_number', 'ends_with_special',
]
    
    # Conversion booléens en int (sklearn préfère les nombres)
    bool_cols = ['has_sequential_nums', 'has_sequential_alpha', 'has_keyboard_pattern',
                 'is_all_same_char', 'has_date_pattern', 'starts_with_capital',
                 'ends_with_number', 'ends_with_special']
    
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    X = df[FEATURES]
    y = df['strength_label']
    
    print(f"Features utilisées : {len(FEATURES)}")
    print(f"Classes : {sorted(y.unique())} → {df['strength_name'].unique().tolist()}")
    
    return X, y, FEATURES



# 2. Entrainement et comparaison de modèles

def train_and_compare(X, y):
    """Entraîne plusieurs modèles et compare leurs performances."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nSplit : {len(X_train):,} entraînement / {len(X_test):,} test")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEntraînement : {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Validation croisée (5 folds) — plus fiable qu'un simple test
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"   Accuracy test    : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   CV score (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results, X_train, X_test, y_train, y_test



# 3. Visualisations des résultats

def plot_confusion_matrix(results, figures_path):
    """Matrice de confusion pour le meilleur modèle."""
    
    rf_result = results['Random Forest']
    
    # Détecter dynamiquement les classes présentes
    classes_present = sorted(rf_result['y_test'].unique())
    label_map = {0: 'Très faible', 1: 'Faible', 2: 'Moyen', 3: 'Fort', 4: 'Très fort'}
    labels = [label_map[c] for c in classes_present]
    
    cm = confusion_matrix(rf_result['y_test'], rf_result['y_pred'], labels=classes_present)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Matrice de confusion — Random Forest\nPrédiction de robustesse des mots de passe',
                 fontweight='bold', pad=15)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '05_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Matrice de confusion sauvegardée")


def plot_feature_importance(model, feature_names, figures_path):
    """Importance des features — graphique clé pour expliquer le modèle."""
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Adapter top_n au nombre réel de features disponibles
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n))[::-1]
    ax.barh(range(top_n), top_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features[::-1]])
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Features les plus importantes\npour prédire la robustesse d'un mot de passe",
                 fontweight='bold')
    
    for i, imp in enumerate(top_importances[::-1]):
        ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '06_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Graphique d'importance sauvegardé")


def plot_model_comparison(results, figures_path):
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    model_names = list(results.keys())
    cv_means = [results[m]['cv_mean'] for m in model_names]
    cv_stds = [results[m]['cv_std'] for m in model_names]
    
    colors = ['#3498db', '#e67e22']
    bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=8,
                  color=colors, edgecolor='white', linewidth=1.5,
                  error_kw={'linewidth': 2})
    
    ax.set_ylabel('Accuracy (Cross-Validation 5-fold)')
    ax.set_title('Comparaison des modèles\n(moyenne ± écart-type sur 5 folds)',
                 fontweight='bold')
    
    # Axe Y adapté dynamiquement aux vraies valeurs
    min_val = min(cv_means) - 0.05
    ax.set_ylim(max(0, min_val), min(1.0, max(cv_means) + 0.05))
    
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.003,
                f'{mean*100:.2f}%', ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '07_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Graphique de comparaison sauvegardé")



# 4. Suvegarde du modèle

def save_best_model(results, feature_names, models_path):
    """Sauvegarde le meilleur modèle avec ses métadonnées."""
    
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best = results[best_name]
    
    model_data = {
        'model': best['model'],
        'feature_names': feature_names,
        'accuracy': best['accuracy'],
        'cv_mean': best['cv_mean'],
        'cv_std': best['cv_std'],
        'model_name': best_name,
        'label_map': {0: 'Très faible', 1: 'Faible', 2: 'Moyen', 3: 'Fort'}
    }
    
    save_path = os.path.join(models_path, 'password_model.pkl')
    joblib.dump(model_data, save_path)
    
    print(f"\nMeilleur modèle : {best_name}")
    print(f"   Accuracy : {best['accuracy']*100:.2f}%")
    print(f"   CV Score : {best['cv_mean']*100:.2f}% ± {best['cv_std']*100:.2f}%")
    print(f"   Sauvegardé dans : {save_path}")
    
    return model_data



# 5. Rapport Texte

def print_full_report(results):
    """Affiche le rapport complet de classification."""
    
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION — RANDOM FOREST")
    print("="*60)
    
    rf = results['Random Forest']
    
    # Détecter dynamiquement les classes présentes
    classes_present = sorted(rf['y_test'].unique())
    label_map = {0: 'Très faible', 1: 'Faible', 2: 'Moyen', 3: 'Fort', 4: 'Très fort'}
    label_names = [label_map[c] for c in classes_present]
    
    print(classification_report(rf['y_test'], rf['y_pred'],
                                target_names=label_names))



# Main

if __name__ == "__main__":
    
    BASE_PATH = r"E:\Password_Analyzer"
    CSV_PATH = os.path.join(BASE_PATH, "data", "processed", "passwords_features.csv")
    FIGURES_PATH = os.path.join(BASE_PATH, "reports", "figures")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    
    # 1. Chargement
    X, y, feature_names = load_and_prepare(CSV_PATH)
    
    # 2. Entraînement
    results, X_train, X_test, y_train, y_test = train_and_compare(X, y)
    
    # 3. Visualisations
    plot_confusion_matrix(results, FIGURES_PATH)
    plot_feature_importance(results['Random Forest']['model'], feature_names, FIGURES_PATH)
    plot_model_comparison(results, FIGURES_PATH)
    
    # 4. Rapport texte
    print_full_report(results)
    
    # 5. Sauvegarde
    model_data = save_best_model(results, feature_names, MODELS_PATH)
    
    print("\nPhase IA terminée ! Modèle prêt pour la Phase Rapport.")