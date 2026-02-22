"""
Extraction de features pour l'analyse de mots de passe
Auteur : Abderemane Attoumani
Projet : Password Analyzer by AI
"""

import math
import re
import pandas as pd
import numpy as np
from zxcvbn import zxcvbn

# 1. Chargement des données

def load_rockyou(filepath, sample_size=100000, random_seed=42):
    """
    Charge un échantillon de rockyou.txt de
    100 000 mots de passe (fichier complet = 32M, trop lourd)
    """
    passwords = []
    
    print(f"Chargement de rockyou.txt...")
    
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if pwd and 4 <= len(pwd) <= 30:  # Filtrer les extrêmes
                passwords.append(pwd)
    
    print(f"{len(passwords):,} mots de passe valides trouvés")
    
    # Échantillon aléatoire reproductible
    np.random.seed(random_seed)
    if len(passwords) > sample_size:
        passwords = list(np.random.choice(passwords, sample_size, replace=False))
    
    return passwords


# 2. Calcul des features

def get_charset_size(password):
    """Taille de l'alphabet utilisé dans le mot de passe."""
    size = 0
    if re.search(r'[a-z]', password): size += 26
    if re.search(r'[A-Z]', password): size += 26
    if re.search(r'[0-9]', password): size += 10
    if re.search(r'[^a-zA-Z0-9]', password): size += 32
    return max(size, 1)


def calculate_entropy(password):
    """
    Entropie théorique en bits = log2(N) * L
    N = taille alphabet, L = longueur
    """
    N = get_charset_size(password)
    L = len(password)
    return round(math.log2(N) * L, 2)


def detect_patterns(password):
    """Détecte les patterns faibles connus."""
    pwd_lower = password.lower()
    
    patterns = {
        'has_sequential_nums': bool(re.search(r'(012|123|234|345|456|567|678|789|890)', password)),
        'has_sequential_alpha': bool(re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', pwd_lower)),
        'has_keyboard_pattern': bool(re.search(r'(qwerty|azerty|qwert|asdf|zxcv|1q2w|pass|admin|login)', pwd_lower)),
        'is_all_same_char': len(set(password)) == 1,
        'has_date_pattern': bool(re.search(r'(19|20)\d{2}|[0-3]\d[0-1]\d\d{2}', password)),
        'starts_with_capital': password[0].isupper() if password else False,
        'ends_with_number': password[-1].isdigit() if password else False,
        'ends_with_special': bool(re.search(r'[^a-zA-Z0-9]$', password)),
    }
    return patterns


def extract_features(password):
    """
    Extrait TOUTES les features d'un mot de passe.
    C'est cette fonction qui sera appelée pour chaque mot de passe.
    """
    length = len(password)
    
    # Features de base
    features = {
        'password': password,
        'length': length,
        'charset_size': get_charset_size(password),
        'entropy': calculate_entropy(password),
        
        # Composition
        'num_lowercase': sum(1 for c in password if c.islower()),
        'num_uppercase': sum(1 for c in password if c.isupper()),
        'num_digits': sum(1 for c in password if c.isdigit()),
        'num_special': sum(1 for c in password if not c.isalnum()),
        'num_unique_chars': len(set(password)),
        
        # Ratios (plus parlants que les valeurs brutes)
        'ratio_uppercase': sum(1 for c in password if c.isupper()) / length,
        'ratio_digits': sum(1 for c in password if c.isdigit()) / length,
        'ratio_special': sum(1 for c in password if not c.isalnum()) / length,
        'ratio_unique': len(set(password)) / length,
    }
    
    # Patterns faibles
    patterns = detect_patterns(password)
    features.update(patterns)
    
    # Score zxcvbn (le meilleur estimateur existant, utilisé par Dropbox)
    try:
        zxcvbn_result = zxcvbn(password)
        features['zxcvbn_score'] = zxcvbn_result['score']  # 0 à 4
        features['crack_time_seconds'] = zxcvbn_result['crack_times_seconds']['offline_fast_hashing_1e10_per_second']
    except:
        features['zxcvbn_score'] = 0
        features['crack_time_seconds'] = 0
    
    return features


# 3. Labellisation ("vérité terrain")

def assign_strength_label(row):
    """
    Label = zxcvbn_score directement (0 à 4).
    zxcvbn est notre 'expert externe', le modèle n'y a PAS accès.
    C'est la séparation propre : label externe, features internes.
    """
    return int(row['zxcvbn_score'])


# 4. Pipeline complet

def build_dataset(rockyou_path, sample_size=50000):
    """
    Pipeline complet : charge → extrait features → labellise → sauvegarde.
    """
    # Chargement
    passwords = load_rockyou(rockyou_path, sample_size)
    
    # Extraction des features (avec barre de progression)
    print("Extraction des features...")
    
    try:
        from tqdm import tqdm
        features_list = [extract_features(pwd) for pwd in tqdm(passwords)]
    except ImportError:
        features_list = [extract_features(pwd) for pwd in passwords]
        print("Features extraites")
    
    # Création du DataFrame
    df = pd.DataFrame(features_list)
    
    # Labellisation
    print("Attribution des labels de robustesse...")
    df['strength_label'] = df.apply(assign_strength_label, axis=1)
    df['strength_name'] = df['strength_label'].map({
        0: 'Très faible',
        1: 'Faible', 
        2: 'Moyen',
        3: 'Fort',
        4: 'Très fort'
    })
    
    print(f"\nDistribution des labels :")
    print(df['strength_name'].value_counts())
    
    return df


if __name__ == "__main__":
    df = build_dataset(r"E:\Password_Analyzer\data\raw\rockyou.txt", sample_size=50000)
    
    # Sauvegarde
    df.to_csv(r"E:\Password_Analyzer\data\processed\passwords_features.csv", index=False)
    print(f"\nDataset sauvegardé ! {len(df):,} mots de passe avec {len(df.columns)} features")