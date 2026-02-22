# src/report_generator.py
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

BLEU_CYBER = colors.HexColor('#1a237e')
BLEU_CLAIR = colors.HexColor('#3498db')
GRIS_FOND  = colors.HexColor('#f5f6fa')
GRIS_TEXTE = colors.HexColor('#2c3e50')

MARGE = 2.2 * cm
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 2 * MARGE

def get_styles():
    return {
        'titre_principal': ParagraphStyle('titre_principal', fontSize=26,
            fontName='Helvetica-Bold', textColor=BLEU_CYBER, alignment=TA_CENTER, spaceAfter=8),
        'sous_titre': ParagraphStyle('sous_titre', fontSize=13,
            fontName='Helvetica', textColor=BLEU_CLAIR, alignment=TA_CENTER, spaceAfter=6),
        'section': ParagraphStyle('section', fontSize=14, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, spaceBefore=14, spaceAfter=6),
        'corps': ParagraphStyle('corps', fontSize=10, fontName='Helvetica',
            textColor=GRIS_TEXTE, alignment=TA_JUSTIFY, spaceAfter=6, leading=16),
        'highlight': ParagraphStyle('highlight', fontSize=10, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, spaceAfter=4),
        'footer': ParagraphStyle('footer', fontSize=8, fontName='Helvetica',
            textColor=colors.grey, alignment=TA_CENTER),
        'cell': ParagraphStyle('cell', fontSize=9, fontName='Helvetica',
            textColor=GRIS_TEXTE, leading=13),
        'cell_bold': ParagraphStyle('cell_bold', fontSize=9, fontName='Helvetica-Bold',
            textColor=BLEU_CYBER, leading=13),
        'cell_header': ParagraphStyle('cell_header', fontSize=9, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_CENTER, leading=13),
        'cell_center': ParagraphStyle('cell_center', fontSize=9, fontName='Helvetica',
            textColor=GRIS_TEXTE, alignment=TA_CENTER, leading=13),
    }

def sep():
    return HRFlowable(width="100%", thickness=1, color=BLEU_CLAIR, spaceAfter=8, spaceBefore=2)

def P(text, style):
    return Paragraph(str(text), style)

def make_table(data_raw, col_widths, S, center_cols=None):
    center_cols = center_cols or []
    data = []
    for row_idx, row in enumerate(data_raw):
        new_row = []
        for col_idx, cell in enumerate(row):
            if row_idx == 0:
                new_row.append(P(str(cell), S['cell_header']))
            elif col_idx in center_cols:
                new_row.append(P(str(cell), S['cell_center']))
            else:
                new_row.append(P(str(cell), S['cell']))
        data.append(new_row)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  BLEU_CYBER),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [GRIS_FOND, colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 7),
        ('RIGHTPADDING',  (0,0), (-1,-1), 7),
    ]))
    return t

def build_report(df_path, figures_path, output_path, model_results):
    df = pd.read_csv(df_path)
    S  = get_styles()

    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=MARGE, rightMargin=MARGE, topMargin=2*cm, bottomMargin=2*cm,
        title="Rapport — Analyseur de Mots de Passe par IA", author="Abderemane Attoumani")

    story = []
    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")

    # PAGE DE GARDE
    story.append(Spacer(1, 2*cm))
    story.append(P("ANALYSEUR DE MOTS DE PASSE", S['titre_principal']))
    story.append(P("par Intelligence Artificielle", S['sous_titre']))
    story.append(Spacer(1, 0.4*cm))
    story.append(sep())
    story.append(Spacer(1, 0.4*cm))
    info_rows = [
        ('Projet',    'Analyseur de Mots de Passe par IA'),
        ('Contexte',  'Projet personnel — Cybersécurité & Machine Learning'),
        ('Auteur',    'Abderemane Attoumani'),
        ('GitHub',    'github.com/abderemaneattoumani'),
        ('Généré le', date_str),
        ('Dataset',   f"{len(df):,} mots de passe (rockyou.txt)"),
        ('Modèle',    'Gradient Boosting Classifier'),
        ('Accuracy',  f"{model_results['cv_mean']*100:.2f}% (CV 5-fold)"),
    ]
    t_garde = Table([[P(r[0], S['cell_bold']), P(r[1], S['cell'])] for r in info_rows],
                    colWidths=[4*cm, CONTENT_W-4*cm])
    t_garde.setStyle(TableStyle([
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[GRIS_FOND, colors.white]),
        ('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),
        ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LEFTPADDING',(0,0),(-1,-1),10), ('VALIGN',(0,0),(-1,-1),'TOP'),
    ]))
    story.append(t_garde)
    story.append(PageBreak())

    # 1. CONTEXTE ET OBJECTIFS
    story.append(P("1. Contexte et Objectifs", S['section']))
    story.append(sep())
    story.append(P("Ce projet personnel vise à développer un système d'analyse automatique "
        "de la robustesse des mots de passe par intelligence artificielle. La sécurité "
        "des mots de passe représente l'un des enjeux majeurs de la cybersécurité moderne : "
        "selon le rapport Verizon DBIR 2023, plus de 80% des violations de données "
        "impliquent des mots de passe compromis.", S['corps']))
    story.append(P("L'approche combine l'analyse structurelle des mots de passe (longueur, "
        "composition, patterns) avec un modèle de Machine Learning entraîné sur 50 000 "
        "mots de passe réels issus de la fuite RockYou (2009), référence mondiale en "
        "sécurité informatique.", S['corps']))
    story.append(Spacer(1, 0.2*cm))
    story.append(P("Objectifs du projet :", S['highlight']))
    obj = [
        ["#","Objectif","Statut"],
        ["1","Analyser et visualiser la distribution des mots de passe réels","Réalisé"],
        ["2","Extraire des features pertinentes (entropie, patterns, composition)","Réalisé"],
        ["3","Entraîner un modèle IA de classification de robustesse","Réalisé"],
        ["4","Identifier et corriger le Data Leakage","Réalisé"],
        ["5","Générer ce rapport PDF automatiquement","Réalisé"],
        ["6","Publier le projet sur GitHub avec Colab interactif","En cours"],
    ]
    story.append(make_table(obj, [1*cm, CONTENT_W-4.5*cm, 3.5*cm], S, center_cols=[0,2]))
    story.append(PageBreak())

    # 2. ANALYSE DES DONNÉES
    story.append(P("2. Analyse du Dataset", S['section']))
    story.append(sep())
    story.append(P("2.1 Statistiques générales", S['highlight']))
    stats = [
        ["Métrique","Valeur","Interprétation"],
        ["Mots de passe analysés", f"{len(df):,}", "Échantillon de rockyou.txt"],
        ["Longueur moyenne", f"{df['length'].mean():.2f} caractères", "En dessous du seuil recommandé (12+)"],
        ["Entropie moyenne", f"{df['entropy'].mean():.2f} bits", "Sous le seuil sécurisé (60+ bits)"],
        ["Caractères uniques (moy.)", f"{df['num_unique_chars'].mean():.2f}", "Faible diversité globale"],
        ["Score zxcvbn moyen", f"{df['zxcvbn_score'].mean():.2f} / 4", "Robustesse globalement faible"],
    ]
    story.append(make_table(stats, [5*cm, 3.5*cm, CONTENT_W-8.5*cm], S))
    story.append(Spacer(1, 0.4*cm))
    story.append(P("2.2 Distribution de la robustesse", S['highlight']))
    label_map = {0:'Très faible',1:'Faible',2:'Moyen',3:'Fort',4:'Très fort'}
    dist = df['strength_label'].value_counts().sort_index()
    dist_data = [["Niveau","Nombre","%","Signification"]]
    for lid, count in dist.items():
        pct = count/len(df)*100
        sig = ("Cracké en moins d'une seconde" if lid==0 else
               "Cracké en quelques minutes" if lid==1 else
               "Résistance modérée" if lid==2 else
               "Bonne résistance" if lid==3 else "Très haute résistance")
        dist_data.append([label_map.get(lid,str(lid)), f"{count:,}", f"{pct:.1f}%", sig])
    story.append(make_table(dist_data, [3*cm,2.5*cm,2*cm,CONTENT_W-7.5*cm], S, center_cols=[1,2]))
    story.append(Spacer(1, 0.4*cm))
    fig1 = os.path.join(figures_path, '01_distribution_robustesse.png')
    if os.path.exists(fig1):
        story.append(P("Figure 1 — Distribution de la robustesse des mots de passe réels", S['footer']))
        story.append(Image(fig1, width=CONTENT_W, height=6*cm))
    story.append(PageBreak())

    # 3. FEATURE ENGINEERING
    story.append(P("3. Feature Engineering", S['section']))
    story.append(sep())
    story.append(P("Le feature engineering consiste à transformer un mot de passe brut "
        "(une chaîne de caractères) en un vecteur numérique que le modèle peut interpréter. "
        "14 features ont été extraites, organisées en 3 catégories :", S['corps']))
    feat = [
        ["Catégorie","Features","Rôle"],
        ["Structure","length, num_unique_chars","Longueur et diversité des caractères"],
        ["Composition","num_lowercase, num_uppercase, num_digits, num_special","Types de caractères utilisés"],
        ["Ratios","ratio_uppercase, ratio_digits, ratio_special, ratio_unique","Proportions normalisées par la longueur"],
        ["Patterns","has_sequential_nums, has_keyboard_pattern, has_date_pattern, is_all_same_char","Détection de patterns faibles connus"],
        ["Position","starts_with_capital, ends_with_number, ends_with_special","Habitudes humaines de construction"],
    ]
    story.append(make_table(feat, [2.8*cm, 7.2*cm, CONTENT_W-10*cm], S))
    story.append(Spacer(1, 0.3*cm))
    story.append(P("Note — Prévention du Data Leakage : Les features 'entropy', 'charset_size' "
        "et 'zxcvbn_score' ont été exclues de l'entraînement car elles ont servi à construire "
        "les labels cibles. Les inclure aurait mené à une accuracy artificielle de 100%.", S['corps']))
    fig_fi = os.path.join(figures_path, '06_feature_importance.png')
    if os.path.exists(fig_fi):
        story.append(P("Figure 2 — Importance des features selon le modèle Random Forest", S['footer']))
        story.append(Image(fig_fi, width=CONTENT_W, height=7*cm))
    story.append(PageBreak())

    # 4. MODÈLE IA — tout lié sur la même page
    story.append(P("4. Modèle d'Intelligence Artificielle", S['section']))
    story.append(sep())

    intro_41 = [
        P("4.1 Choix et comparaison des algorithmes", S['highlight']),
        P("Deux algorithmes ont été comparés via une validation croisée à 5 folds, "
          "méthode plus fiable qu'un simple découpage train/test car elle évalue "
          "le modèle sur l'ensemble du dataset de manière tournante :", S['corps']),
    ]
    modeles = [
        ["Modèle","Accuracy Test","CV Score (5-fold)","Écart-type","Verdict"],
        ["Random Forest",
         f"{model_results['rf_acc']*100:.2f}%", f"{model_results['rf_cv']*100:.2f}%",
         f"+/- {model_results['rf_std']*100:.2f}%", "Bon"],
        ["Gradient Boosting",
         f"{model_results['gb_acc']*100:.2f}%", f"{model_results['gb_cv']*100:.2f}%",
         f"+/- {model_results['gb_std']*100:.2f}%", "Retenu"],
    ]
    t_modeles = make_table(modeles, [4*cm,2.8*cm,3.2*cm,2.5*cm,CONTENT_W-12.5*cm], S, center_cols=[1,2,3,4])

    fig_comp = os.path.join(figures_path, '07_model_comparison.png')
    elems_comp = ([Spacer(1,0.3*cm), Image(fig_comp, width=CONTENT_W, height=6*cm)]
                  if os.path.exists(fig_comp) else [])

    texte_42 = [
        Spacer(1, 0.3*cm),
        P("4.2 Analyse des performances", S['highlight']),
        P(f"Le Gradient Boosting obtient {model_results['cv_mean']*100:.2f}% de précision "
          "en validation croisée. Ce résultat est le seul résultat valide : les scores "
          "élevés précédents (99%+) étaient dus au Data Leakage. Un modèle aléatoire sur "
          "5 classes obtiendrait 20% — notre modèle apporte donc une valeur ajoutée réelle "
          f"de +{(model_results['cv_mean']-0.2)*100:.1f} points de pourcentage.", S['corps']),
        P("La classe 'Très faible' présentait un recall nul avant correction. "
          "Le paramètre class_weight='balanced' a permis d'améliorer sa détection — "
          "compromis pertinent en cybersécurité où manquer un mot de passe faible "
          "est plus coûteux qu'une fausse alerte.", S['corps']),
    ]
    fig_cm = os.path.join(figures_path, '05_confusion_matrix.png')
    elems_cm = ([Spacer(1,0.3*cm),
                 P("Figure 3 — Matrice de confusion (Random Forest avec class_weight='balanced')", S['footer']),
                 Image(fig_cm, width=CONTENT_W*0.82, height=8.5*cm)]
                if os.path.exists(fig_cm) else [])

    story.append(KeepTogether(intro_41 + [t_modeles] + elems_comp + texte_42 + elems_cm))
    story.append(PageBreak())

    # 5. PATTERNS FAIBLES
    story.append(P("5. Analyse des Patterns Faibles", S['section']))
    story.append(sep())
    story.append(P("L'analyse des mots de passe réels de rockyou.txt révèle des comportements "
        "humains prévisibles que les attaquants exploitent systématiquement :", S['corps']))
    bool_cols = [
        ('has_sequential_nums',  'Séquence numérique (123, 456...)'),
        ('has_sequential_alpha', 'Séquence alphabétique (abc, xyz...)'),
        ('has_keyboard_pattern', 'Pattern clavier (qwerty, azerty...)'),
        ('is_all_same_char',     'Caractère unique répété (aaaa, 1111...)'),
        ('has_date_pattern',     'Pattern date (1994, 2001...)'),
        ('ends_with_number',     'Se termine par un chiffre'),
    ]
    pat_data = [["Pattern détecté", "Nb", "% dataset", "Risque"]]
    for col, name in bool_cols:
        if col in df.columns:
            count = int(df[col].sum())
            pct   = count/len(df)*100
            risque = "Très élevé" if pct>30 else "Élevé" if pct>15 else "Modéré"
            pat_data.append([name, f"{count:,}", f"{pct:.1f}%", risque])
    story.append(make_table(pat_data, [7.5*cm,2*cm,2.5*cm,CONTENT_W-12*cm], S, center_cols=[1,2,3]))
    story.append(Spacer(1, 0.3*cm))
    fig4 = os.path.join(figures_path, '04_patterns_faibles.png')
    if os.path.exists(fig4):
        story.append(Image(fig4, width=CONTENT_W, height=7*cm))
    story.append(PageBreak())

    # 6. RECOMMANDATIONS
    story.append(P("6. Recommandations de Sécurité", S['section']))
    story.append(sep())
    story.append(P("Sur la base de l'analyse de 50 000 mots de passe réels et des résultats "
        "du modèle IA, voici les recommandations issues du standard NIST SP 800-63B :", S['corps']))
    reco = [
        ["Priorité","Recommandation","Justification"],
        ["Critique","Longueur minimum 12 caractères","Feature la plus importante selon le modèle"],
        ["Critique","Éviter les mots du dictionnaire","Attaques par dictionnaire en moins d'une seconde"],
        ["Élevée","Inclure chiffres ET caractères spéciaux","Augmente considérablement l'espace de recherche"],
        ["Élevée","Ne pas terminer par un chiffre (ex: pass1)",
         f"{df['ends_with_number'].mean()*100:.1f}% des mots de passe font cette erreur"],
        ["Modérée","Éviter les dates de naissance",
         f"{df['has_date_pattern'].mean()*100:.1f}% des mots de passe contiennent une date"],
        ["Modérée","Utiliser un gestionnaire de mots de passe",
         "Seule solution viable pour des mots de passe forts et uniques sur chaque site"],
        ["Bonne pratique","Activer l'authentification à 2 facteurs",
         "Sécurité complémentaire même si le mot de passe est compromis"],
    ]
    story.append(make_table(reco, [2.8*cm, 6.2*cm, CONTENT_W-9*cm], S))
    story.append(PageBreak())

    # 7. CONCLUSION
    story.append(P("7. Conclusion et Perspectives", S['section']))
    story.append(sep())
    story.append(P("Ce projet démontre l'application concrète du Machine Learning à un problème "
        "de cybersécurité réel. Au-delà des performances du modèle, il illustre des "
        "compétences essentielles : la gestion du Data Leakage, le traitement du "
        "déséquilibre de classes, et la capacité à expliquer et nuancer des résultats "
        "plutôt que de présenter des scores artificiellement parfaits.", S['corps']))
    story.append(P(f"L'analyse de {len(df):,} mots de passe réels confirme que plus de 75% "
        "des utilisateurs emploient des mots de passe faibles ou très faibles, soulignant "
        "l'importance d'une sensibilisation continue à la sécurité.", S['corps']))
    story.append(Spacer(1, 0.2*cm))
    story.append(P("Perspectives d'amélioration :", S['highlight']))
    persp = [
        ["Amélioration","Impact attendu"],
        ["Intégrer un dictionnaire de prénoms et mots communs comme feature","Accuracy estimée +10 à +15%"],
        ["Augmenter le dataset à 500 000 mots de passe","Meilleure représentation des classes rares"],
        ["Déployer via une API Flask ou FastAPI","Utilisable dans une vraie application web"],
        ["Ajouter l'estimation du temps de craquage en clair","Interface utilisateur plus compréhensible"],
    ]
    story.append(make_table(persp, [CONTENT_W*0.65, CONTENT_W*0.35], S))
    story.append(Spacer(1, 1*cm))
    story.append(sep())
    story.append(P(f"Rapport généré automatiquement le {date_str}  |  "
                   "Abderemane Attoumani  |  github.com/abderemaneattoumani", S['footer']))

    doc.build(story)
    print(f"Rapport PDF genere : {output_path}")


if __name__ == "__main__":
    import joblib
    BASE = r"E:\Password_Analyzer"
    model_data = joblib.load(os.path.join(BASE, "models", "password_model.pkl"))
    model_results = {
        'rf_acc': 0.6242, 'rf_cv': 0.6273, 'rf_std': 0.0030,
        'gb_acc': 0.6625, 'gb_cv': 0.6694, 'gb_std': 0.0018,
        'cv_mean': model_data['cv_mean'],
    }
    build_report(
        df_path=os.path.join(BASE, "data", "processed", "passwords_features.csv"),
        figures_path=os.path.join(BASE, "reports", "figures"),
        output_path=os.path.join(BASE, "reports", "rapport_final.pdf"),
        model_results=model_results
    )
