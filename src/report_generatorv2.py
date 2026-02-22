"""
Génération automatique d'un rapport PDF professionnel.
Auteur : Abderemane Attoumani - Projet Personnel Cybersécurité & IA
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, HRFlowable,
                                 KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import warnings
warnings.filterwarnings('ignore')


# Couleurs et Styles

BLEU_CYBER = colors.HexColor('#1a237e')
BLEU_CLAIR = colors.HexColor('#3498db')
ROUGE      = colors.HexColor('#e74c3c')
VERT       = colors.HexColor('#2ecc71')
ORANGE     = colors.HexColor('#e67e22')
GRIS_FOND  = colors.HexColor('#f5f6fa')
GRIS_TEXTE = colors.HexColor('#2c3e50')

PAGE_WIDTH = A4[0] - 4*cm  # largeur utile (marges 2cm de chaque côté)

def get_styles():
    styles = getSampleStyleSheet()

    custom = {
        'titre_principal': ParagraphStyle(
            'titre_principal',
            fontSize=26, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, alignment=TA_CENTER,
            spaceAfter=8
        ),
        'sous_titre': ParagraphStyle(
            'sous_titre',
            fontSize=13, fontName='Helvetica',
            textColor=BLEU_CLAIR, alignment=TA_CENTER,
            spaceAfter=6
        ),
        'section': ParagraphStyle(
            'section',
            fontSize=14, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, spaceBefore=16, spaceAfter=8,
        ),
        'corps': ParagraphStyle(
            'corps',
            fontSize=10, fontName='Helvetica',
            textColor=GRIS_TEXTE, alignment=TA_JUSTIFY,
            spaceAfter=6, leading=16
        ),
        'highlight': ParagraphStyle(
            'highlight',
            fontSize=10, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, spaceAfter=4
        ),
        'footer': ParagraphStyle(
            'footer',
            fontSize=8, fontName='Helvetica',
            textColor=colors.grey, alignment=TA_CENTER
        ),
        # Style pour texte dans les cellules de tableau — wrapping automatique
        'cell': ParagraphStyle(
            'cell',
            fontSize=9, fontName='Helvetica',
            textColor=GRIS_TEXTE, leading=13,
            wordWrap='CJK'
        ),
        'cell_bold': ParagraphStyle(
            'cell_bold',
            fontSize=9, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, leading=13,
            wordWrap='CJK'
        ),
        'cell_header': ParagraphStyle(
            'cell_header',
            fontSize=10, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_CENTER,
            leading=13, wordWrap='CJK'
        ),
    }
    return custom


# Composants réutilisables

def separateur():
    return HRFlowable(width="100%", thickness=1, color=BLEU_CLAIR,
                      spaceAfter=10, spaceBefore=4)


def P(text, style):
    """Raccourci : crée un Paragraph avec wrapping automatique."""
    return Paragraph(str(text), style)


def tableau_stats(data_raw, col_widths, styles_dict):
    """
    Crée un tableau stylisé avec wrapping automatique dans toutes les cellules.
    data_raw : liste de listes de strings (ou Paragraphs)
    col_widths : liste de largeurs en points
    """
    # Convertir toutes les cellules en Paragraph pour le wrapping automatique
    data = []
    for row_idx, row in enumerate(data_raw):
        new_row = []
        for cell in row:
            if isinstance(cell, str):
                if row_idx == 0:
                    new_row.append(P(cell, styles_dict['cell_header']))
                else:
                    new_row.append(P(cell, styles_dict['cell']))
            else:
                new_row.append(cell)
        data.append(new_row)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        # En-tête
        ('BACKGROUND',    (0, 0), (-1, 0),  BLEU_CYBER),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
        ('ALIGN',         (0, 0), (-1, 0),  'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        # Corps
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRIS_FOND, colors.white]),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
    ]))
    return t


# Construction du rapport

def build_report(df_path, figures_path, output_path, model_results):

    df = pd.read_csv(df_path)
    styles = get_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Rapport — Analyseur de Mots de Passe par IA",
        author="Projet Personnel Cybersécurité & IA"
    )

    story = []
    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")

    # Page de garde
    story.append(Spacer(1, 2*cm))
    story.append(P("ANALYSEUR DE MOTS DE PASSE", styles['titre_principal']))
    story.append(Spacer(1, 0.5*cm))
    story.append(P("par Intelligence Artificielle", styles['sous_titre']))
    story.append(Spacer(1, 0.5*cm))
    story.append(separateur())
    story.append(Spacer(1, 0.5*cm))

    info_data = [
        ['Projet',    'Analyseur de Mots de Passe par IA'],
        ['Contexte',  'Projet personnel — Cybersécurité & Machine Learning'],
        ['Auteur',    'Abderemane Attoumani'],
        ['GitHub',    'github.com/abderemaneattoumani'],
        ['Généré le', date_str],
        ['Dataset',   f"{len(df):,} mots de passe (rockyou.txt)"],
        ['Modèle',    'Gradient Boosting Classifier'],
        ['Accuracy',  f"{model_results['cv_mean']*100:.2f}% (CV 5-fold)"],
    ]

    t = Table(
        [[P(r[0], styles['cell_bold']), P(r[1], styles['cell'])] for r in info_data],
        colWidths=[4.5*cm, 11.5*cm]
    )
    t.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [GRIS_FOND, colors.white]),
        ('GRID',           (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('TOPPADDING',     (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 7),
        ('LEFTPADDING',    (0, 0), (-1, -1), 10),
        ('VALIGN',         (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t)
    story.append(PageBreak())

    # 1. Contexte et Obejectifs
    story.append(P("1. Contexte et Objectifs", styles['section']))
    story.append(separateur())

    story.append(P(
        "Ce projet personnel vise à développer un système d'analyse automatique "
        "de la robustesse des mots de passe par intelligence artificielle. "
        "La sécurité des mots de passe représente l'un des enjeux majeurs de la "
        "cybersécurité moderne : selon le rapport Verizon DBIR 2023, plus de 80% "
        "des violations de données impliquent des mots de passe compromis.",
        styles['corps']))

    story.append(P(
        "L'approche combine l'analyse structurelle des mots de passe (longueur, "
        "composition, patterns) avec un modèle de Machine Learning entraîné sur "
        "50 000 mots de passe réels issus de la fuite RockYou (2009), référence "
        "mondiale en sécurité informatique.",
        styles['corps']))

    story.append(Spacer(1, 0.3*cm))
    story.append(P("Objectifs du projet :", styles['highlight']))

    objectifs = [
        ["#", "Objectif", "Statut"],
        ["1", "Analyser et visualiser la distribution des mots de passe réels", "Réalisé"],
        ["2", "Extraire des features pertinentes (entropie, patterns, composition)", "Réalisé"],
        ["3", "Entraîner un modèle IA de classification de robustesse", "Réalisé"],
        ["4", "Identifier et corriger le Data Leakage", "Réalisé"],
        ["5", "Générer ce rapport PDF automatiquement", "Réalisé"],
        ["6", "Publier le projet sur GitHub avec Colab interactif", "En cours"],
    ]
    story.append(tableau_stats(objectifs, [1*cm, 12*cm, 3*cm], styles))
    story.append(PageBreak())

    # 2. Aanalyse des données
    story.append(P("2. Analyse du Dataset", styles['section']))
    story.append(separateur())

    story.append(P("2.1 Statistiques générales", styles['highlight']))

    stats_data = [
        ["Métrique", "Valeur", "Interprétation"],
        ["Mots de passe analysés",    f"{len(df):,}",                          "Échantillon de rockyou.txt"],
        ["Longueur moyenne",          f"{df['length'].mean():.2f} caractères", "En dessous du seuil recommandé (12+)"],
        ["Entropie moyenne",          f"{df['entropy'].mean():.2f} bits",      "Sous le seuil sécurisé (60+ bits)"],
        ["Caractères uniques (moy.)", f"{df['num_unique_chars'].mean():.2f}",  "Faible diversité globale"],
        ["Score zxcvbn moyen",        f"{df['zxcvbn_score'].mean():.2f} / 4",  "Robustesse globalement faible"],
    ]
    story.append(tableau_stats(stats_data, [5*cm, 4*cm, 7*cm], styles))
    story.append(Spacer(1, 0.5*cm))

    story.append(P("2.2 Distribution de la robustesse", styles['highlight']))
    label_map = {0: 'Très faible', 1: 'Faible', 2: 'Moyen', 3: 'Fort', 4: 'Très fort'}
    dist = df['strength_label'].value_counts().sort_index()

    dist_data = [["Niveau", "Nombre", "Pourcentage", "Signification"]]
    for label_id, count in dist.items():
        pct = count / len(df) * 100
        signification = (
            "Cracké en moins d'une seconde"  if label_id == 0 else
            "Cracké en quelques minutes"     if label_id == 1 else
            "Résistance modérée"             if label_id == 2 else
            "Bonne résistance"               if label_id == 3 else
            "Très haute résistance"
        )
        dist_data.append([label_map.get(label_id, str(label_id)),
                          f"{count:,}", f"{pct:.1f}%", signification])
    story.append(tableau_stats(dist_data, [3*cm, 2.5*cm, 3*cm, 7.5*cm], styles))
    story.append(Spacer(1, 0.4*cm))

    fig1 = os.path.join(figures_path, '01_distribution_robustesse.png')
    if os.path.exists(fig1):
        story.append(P("Figure 1 — Distribution de la robustesse des mots de passe réels",
                       styles['footer']))
        story.append(Image(fig1, width=16*cm, height=6*cm))

    story.append(PageBreak())

    # 3. Feature engineering
    story.append(P("3. Feature Engineering", styles['section']))
    story.append(separateur())

    story.append(P(
        "Le feature engineering consiste à transformer un mot de passe brut "
        "(une chaîne de caractères) en un vecteur numérique que le modèle peut "
        "interpréter. 14 features ont été extraites, organisées en 3 catégories :",
        styles['corps']))

    features_data = [
        ["Catégorie", "Features", "Rôle"],
        ["Structure",    "length, num_unique_chars",
                         "Longueur et diversité des caractères"],
        ["Composition",  "num_lowercase, num_uppercase, num_digits, num_special",
                         "Types de caractères utilisés"],
        ["Ratios",       "ratio_uppercase, ratio_digits, ratio_special, ratio_unique",
                         "Proportions normalisées par la longueur"],
        ["Patterns",     "has_sequential_nums, has_keyboard_pattern, has_date_pattern, is_all_same_char",
                         "Détection de patterns faibles connus"],
        ["Position",     "starts_with_capital, ends_with_number, ends_with_special",
                         "Habitudes humaines de construction"],
    ]
    story.append(tableau_stats(features_data, [3*cm, 7*cm, 6*cm], styles))
    story.append(Spacer(1, 0.4*cm))

    story.append(P(
        "Note importante — Prévention du Data Leakage : Les features 'entropy', "
        "'charset_size' et 'zxcvbn_score' ont volontairement été exclues de "
        "l'entraînement car elles ont servi à construire les labels cibles. "
        "Les inclure aurait constitué une fuite de données menant à une accuracy "
        "artificielle de 100%.", styles['corps']))

    fig_fi = os.path.join(figures_path, '06_feature_importance.png')
    if os.path.exists(fig_fi):
        story.append(P("Figure 2 — Importance des features selon le modèle Random Forest",
                       styles['footer']))
        story.append(Image(fig_fi, width=16*cm, height=7*cm))

    story.append(PageBreak())

    # 4. Modèle IA
    story.append(P("4. Modèle d'Intelligence Artificielle", styles['section']))
    story.append(separateur())

    story.append(P("4.1 Choix et comparaison des algorithmes", styles['highlight']))
    story.append(P(
        "Deux algorithmes de classification ont été comparés via une validation "
        "croisée à 5 folds, méthode plus fiable qu'un simple découpage train/test :",
        styles['corps']))

    modeles_data = [
        ["Modèle", "Accuracy Test", "CV Score (5-fold)", "Écart-type", "Verdict"],
        ["Random Forest",
         f"{model_results['rf_acc']*100:.2f}%",
         f"{model_results['rf_cv']*100:.2f}%",
         f"+/-{model_results['rf_std']*100:.2f}%",
         "Bon"],
        ["Gradient Boosting",
         f"{model_results['gb_acc']*100:.2f}%",
         f"{model_results['gb_cv']*100:.2f}%",
         f"+/-{model_results['gb_std']*100:.2f}%",
         "Retenu"],
    ]
    table_modeles = tableau_stats(modeles_data, [4.5*cm, 2.8*cm, 3.2*cm, 2.5*cm, 3*cm], styles)

    story.append(P("4.2 Analyse des performances", styles['highlight']))
    story.append(P(
        f"Le Gradient Boosting obtient {model_results['cv_mean']*100:.2f}% de précision "
        "en validation croisée. Ce résultat, bien qu'inférieur aux 99%+ obtenus lors "
        "des premières tentatives, est le seul résultat valide : les scores élevés "
        "précédents étaient dus au Data Leakage. Un modèle aléatoire sur 5 classes "
        "obtiendrait 20% — notre modèle apporte donc une valeur ajoutée réelle de "
        f"+{(model_results['cv_mean']-0.2)*100:.1f} points de pourcentage.",
        styles['corps']))

    story.append(P(
        "La classe 'Très faible' présentait un recall nul avant correction. "
        "Le paramètre class_weight='balanced' a permis d'améliorer la détection "
        "de cette classe critique — compromis pertinent en cybersécurité où manquer "
        "un mot de passe faible est plus coûteux qu'une fausse alerte.",
        styles['corps']))

    fig_cm = os.path.join(figures_path, '05_confusion_matrix.png')
    fig_comp = os.path.join(figures_path, '07_model_comparison.png')

    elements_section4 = [table_modeles, Spacer(1, 0.3*cm)]

    if os.path.exists(fig_comp):
        elements_section4.append(Image(fig_comp, width=14*cm, height=6*cm))
        elements_section4.append(Spacer(1, 0.3*cm))

    if os.path.exists(fig_cm):
        elements_section4.append(
            P("Figure 3 — Matrice de confusion (Random Forest avec class_weight='balanced')",
              styles['footer']))
        elements_section4.append(Image(fig_cm, width=13*cm, height=8.5*cm))

    # KeepTogether essaie de garder tout ça sur la même page
    story.append(KeepTogether(elements_section4))

    story.append(PageBreak())

    # 5. Analyse des patterns faibles
    story.append(P("5. Analyse des Patterns Faibles", styles['section']))
    story.append(separateur())

    story.append(P(
        "L'analyse des mots de passe réels de rockyou.txt révèle des comportements "
        "humains prévisibles que les attaquants exploitent systématiquement :",
        styles['corps']))

    bool_cols = [
        ('has_sequential_nums',   'Séquence numérique (123, 456...)'),
        ('has_sequential_alpha',  'Séquence alphabétique (abc, xyz...)'),
        ('has_keyboard_pattern',  'Pattern clavier (qwerty, azerty...)'),
        ('is_all_same_char',      'Caractère unique répété (aaaa, 1111...)'),
        ('has_date_pattern',      'Pattern date (1994, 2001...)'),
        ('ends_with_number',      'Se termine par un chiffre'),
    ]

    patterns_data = [["Pattern détecté", "Occurrences", "% dataset", "Risque"]]
    for col, name in bool_cols:
        if col in df.columns:
            count = int(df[col].sum())
            pct = count / len(df) * 100
            risque = ("Très élevé" if pct > 30 else
                      "Élevé"      if pct > 15 else "Modéré")
            patterns_data.append([name, f"{count:,}", f"{pct:.1f}%", risque])

    story.append(tableau_stats(patterns_data, [7*cm, 2.5*cm, 2.5*cm, 4*cm], styles))
    story.append(Spacer(1, 0.3*cm))

    fig4 = os.path.join(figures_path, '04_patterns_faibles.png')
    if os.path.exists(fig4):
        story.append(Image(fig4, width=16*cm, height=7*cm))

    story.append(PageBreak())

    # 6. Recommandations de Sécurité
    story.append(P("6. Recommandations de Sécurité", styles['section']))
    story.append(separateur())

    story.append(P(
        "Sur la base de l'analyse de 50 000 mots de passe réels et des résultats "
        "du modèle IA, voici les recommandations issues du standard NIST SP 800-63B :",
        styles['corps']))

    reco_data = [
        ["Priorité", "Recommandation", "Justification"],
        ["Critique",
         "Longueur minimum 12 caractères",
         "Feature la plus importante selon le modèle"],
        ["Critique",
         "Éviter les mots du dictionnaire",
         "Attaques par dictionnaire en moins d'une seconde"],
        ["Élevée",
         "Inclure chiffres ET caractères spéciaux",
         "Augmente considérablement l'espace de recherche"],
        ["Élevée",
         "Ne pas terminer par un chiffre (ex: pass1)",
         f"{df['ends_with_number'].mean()*100:.1f}% des mots de passe font cette erreur"],
        ["Modérée",
         "Éviter les dates de naissance",
         f"{df['has_date_pattern'].mean()*100:.1f}% des mots de passe contiennent une date"],
        ["Modérée",
         "Utiliser un gestionnaire de mots de passe",
         "Seule solution viable pour des mots de passe forts et uniques sur chaque site"],
        ["Bonne pratique",
         "Activer l'authentification à 2 facteurs",
         "Sécurité complémentaire même si le mot de passe est compromis"],
    ]
    story.append(tableau_stats(reco_data, [2.5*cm, 6*cm, 7.5*cm], styles))

    story.append(PageBreak())

    # 7. Conclusion
    story.append(P("7. Conclusion et Perspectives", styles['section']))
    story.append(separateur())

    story.append(P(
        "Ce projet démontre l'application concrète du Machine Learning à un problème "
        "de cybersécurité réel. Au-delà des performances du modèle, il illustre des "
        "compétences essentielles : la gestion du Data Leakage, le traitement du "
        "déséquilibre de classes, et la capacité à expliquer et nuancer des résultats "
        "plutôt que de présenter des scores artificiellement parfaits.",
        styles['corps']))

    story.append(P(
        f"L'analyse de {len(df):,} mots de passe réels confirme que plus de 75% "
        "des utilisateurs emploient des mots de passe faibles ou très faibles, "
        "soulignant l'importance d'une sensibilisation continue à la sécurité.",
        styles['corps']))

    story.append(Spacer(1, 0.3*cm))
    story.append(P("Perspectives d'amélioration :", styles['highlight']))

    perspectives_data = [
        ["Amélioration", "Impact attendu"],
        ["Intégrer un dictionnaire de prénoms et mots communs comme feature",
         "Accuracy estimée +10 à +15%"],
        ["Augmenter le dataset à 500 000 mots de passe",
         "Meilleure représentation des classes rares"],
        ["Déployer via une API Flask ou FastAPI",
         "Utilisable dans une vraie application web"],
        ["Ajouter l'estimation du temps de craquage en clair",
         "Interface utilisateur plus compréhensible"],
    ]
    story.append(tableau_stats(perspectives_data, [10*cm, 6*cm], styles))

    story.append(Spacer(1, 1*cm))
    story.append(separateur())
    story.append(P(
        f"Rapport généré automatiquement le {date_str} | Abderemane Attoumani | "
        "github.com/abderemaneattoumani",
        styles['footer']))

    # Génération du PDF
    doc.build(story)
    print(f"Rapport PDF généré : {output_path}")


# Main

if __name__ == "__main__":
    import joblib

    BASE = r"E:\Password_Analyzer"

    model_data = joblib.load(os.path.join(BASE, "models", "password_model.pkl"))

    model_results = {
        'rf_acc':  0.6242,
        'rf_cv':   0.6273,
        'rf_std':  0.0030,
        'gb_acc':  0.6625,
        'gb_cv':   0.6694,
        'gb_std':  0.0018,
        'cv_mean': model_data['cv_mean'],
    }

    build_report(
        df_path=os.path.join(BASE, "data", "processed", "passwords_features.csv"),
        figures_path=os.path.join(BASE, "reports", "figures"),
        output_path=os.path.join(BASE, "reports", "rapport_final.pdf"),
        model_results=model_results
    )
