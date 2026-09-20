# -*- coding: utf-8 -*-
# ============================================================================
#  CHURN ANALYTICS — AL BARID BANK (PFA) — v3
#  Analyse et prédiction de l'attrition des clients (Customer Churn)
#  ---------------------------------------------------------------------------
#  Chaîne complète : données (simulées ou CSV réel) → exploration →
#  modélisation (Pipeline scikit-learn, split train/validation/test,
#  validation croisée) → évaluation (ROC, calibration, seuil métier) →
#  scoring 300-900 → tableau de bord → portefeuille filtrable →
#  simulateur client avec explication individuelle → recommandations
#  personnalisées → limites du modèle.
#
#  v3 : refonte visuelle complète (glassmorphism institutionnel, charte
#  Al Barid Bank, suppression des emojis). La logique statistique,
#  économétrique et Machine Learning est STRICTEMENT identique à la v2 :
#  mêmes formules, mêmes seuils, mêmes hyperparamètres, mêmes résultats.
#
#  Auteur : Walid BEN ABID — FST Errachidia (IFA)
# ============================================================================

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False


# ============================================================================
# 0. CONFIGURATION GÉNÉRALE
# ============================================================================
st.set_page_config(
    page_title="Churn Analytics — Al Barid Bank",
    page_icon="assets/albaridbank.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Charte Al Barid Bank — couleurs inchangées par rapport à la v2
# ----------------------------------------------------------------------------
JAUNE = "#F5B800"          # jaune Al Barid — accent principal
JAUNE_FONCE = "#D9A200"    # hover / état pressé
JAUNE_CLAIR = "#FFF6DC"
BRUN = "#4D3F37"           # brun Al Barid
BRUN_FONCE = "#3A2F29"
TXT = "#2B2320"
MUT = "#8A7F78"
VERT = "#1E8449"
ROUGE = "#B03A2E"
ORANGE = "#CA6F1E"
BLEU = "#003366"
GRIS_CARTE = "#F6F3F0"

COULEURS_CLASSES = {"Faible": "#1E8449", "Modéré": "#B7950B",
                    "Élevé": "#CA6F1E", "Critique": "#B03A2E"}
FONDS_CLASSES = {"Faible": "#D5F5E3", "Modéré": "#FCF3CF",
                 "Élevé": "#FAD7A0", "Critique": "#F5B7B1"}

# Logo officiel Al Barid Bank (identique à la v2, embarqué : aucun fichier requis)
LOGO_B64 = "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"

ASSETS = Path(__file__).parent / "assets"
FOND_FICHIER = "albaridbank.jpg"


# ============================================================================
# 0.1 ARRIÈRE-PLAN : chargement local en base64 (robuste sur Streamlit Cloud)
# ============================================================================
@st.cache_data(show_spinner=False)
def load_background_image(nom_fichier: str = FOND_FICHIER) -> str:
    """Encode l'image d'arrière-plan en base64.

    Renvoie une chaîne vide si le fichier est absent : l'application
    fonctionne alors normalement avec un fond dégradé de repli, sans lever
    d'exception (exigence de robustesse sur Streamlit Community Cloud).
    """
    for candidat in (ASSETS / nom_fichier, Path(nom_fichier),
                     Path("assets") / nom_fichier):
        try:
            if candidat.is_file():
                return base64.b64encode(candidat.read_bytes()).decode("utf-8")
        except Exception:
            continue
    return ""


FOND_B64 = load_background_image()

if FOND_B64:
    # Voile volontairement léger : il protège la lisibilité des rares textes
    # posés hors carte sans masquer la photographie.
    _COUCHE_FOND = (
        "linear-gradient(180deg, rgba(52,41,33,0.20) 0%, "
        "rgba(52,41,33,0.12) 45%, rgba(52,41,33,0.30) 100%), "
        f"url('data:image/jpeg;base64,{FOND_B64}')"
    )
else:
    _COUCHE_FOND = ("linear-gradient(140deg, #AD7A4F 0%, #C89A72 50%, "
                    "#FBE0CF 100%)")


# ============================================================================
# 0.2 FEUILLE DE STYLE CENTRALISÉE
# ============================================================================
CSS = f"""
<style>
/* ===========================================================================
   Palette de verre bronze — échantillonnée sur la référence fournie :
   cœur #C3AA8C (195,170,140), haut #C9AF96, bas #B69B7E.
   La zone de résultats reçoit cette teinte ; la barre latérale reprend la
   même famille en plus sombre pour distinguer les deux plans.
   =========================================================================== */
:root {{
    --abb-jaune: {JAUNE};
    --abb-jaune-fonce: {JAUNE_FONCE};
    --abb-brun: #3A2F29;
    --abb-brun-fonce: {BRUN_FONCE};
    --abb-txt: #2B2320;
    --abb-mut: #3F3226;

    /* les deux couleurs de référence, appliquées telles quelles */
    --creme: #FBE0CF;                             /* zone qui affiche les résultats */
    --brun: #AD7A4F;                              /* barre latérale */
    --verre-feuille: rgba(251, 224, 207, 0.34);   /* grande surface : laisse voir la photo */
    --verre: #FBE0CF;                             /* cartes de résultats */
    --verre-fort: #FBE0CF;                        /* tableaux et texte dense */
    --verre-bord: rgba(173, 122, 79, 0.28);
    --verre-ombre: 0 8px 26px rgba(48, 36, 28, 0.18);

    --rayon: 18px;
    --police: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
}}

/* ---------------------------------------------------------------- FOND -- */
[data-testid="stAppViewContainer"] {{
    background-image: {_COUCHE_FOND};
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0); }}
[data-testid="stToolbar"] {{ right: 1rem; }}

/* ------------------------------------------------- SURFACE PRINCIPALE -- */
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] .block-container,
.main .block-container {{
    background: var(--verre-feuille);
    -webkit-backdrop-filter: blur(6px) saturate(112%);
    backdrop-filter: blur(6px) saturate(112%);
    border: 1px solid var(--verre-bord);
    border-radius: 24px;
    box-shadow: 0 12px 38px rgba(38, 29, 22, 0.20);
    padding: 2.1rem 2.4rem 3rem 2.4rem;
    margin-top: 1.1rem;
    margin-bottom: 2rem;
    max-width: 1500px;
}}

/* ------------------------------------------------------- TYPOGRAPHIE -- */
html, body, [class*="css"], .stMarkdown, .stApp {{
    font-family: var(--police);
    color: var(--abb-txt);
}}
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4,
.main h1, .main h2, .main h3, .main h4 {{
    font-family: var(--police);
    color: var(--abb-brun);
    letter-spacing: -0.01em;
}}
[data-testid="stMain"] h1, .main h1 {{ font-size: 1.62rem; font-weight: 700; margin-bottom: .2rem; }}
[data-testid="stMain"] h2, .main h2 {{ font-size: 1.24rem; font-weight: 650; margin-top: 1.5rem; }}
[data-testid="stMain"] h3, .main h3 {{ font-size: 1.05rem; font-weight: 600; margin-top: 1.1rem; }}
[data-testid="stMain"] p, [data-testid="stMain"] li,
.main p, .main li {{ font-size: 0.945rem; line-height: 1.62; color: var(--abb-txt); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: var(--abb-mut) !important;
}}
hr {{ border-color: rgba(58, 47, 41, 0.20); }}

/* ------------------------------------------------------ EN-TÊTE PAGE -- */
.page-header {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(15px) saturate(118%);
    backdrop-filter: blur(15px) saturate(118%);
    border: 1px solid var(--verre-bord);
    border-radius: var(--rayon);
    box-shadow: var(--verre-ombre);
    padding: 20px 26px 18px 26px;
    margin: 0 0 22px 0;
}}
.page-header .ph-title {{
    font-size: 1.42rem; font-weight: 700; color: var(--abb-brun);
    letter-spacing: -0.015em; line-height: 1.25;
}}
.page-header .ph-sub {{
    font-size: 0.9rem; color: var(--abb-mut); margin-top: 5px; line-height: 1.5;
}}
.page-header .ph-rule {{
    width: 58px; height: 3px; border-radius: 2px; margin-top: 13px;
    background: linear-gradient(90deg, var(--abb-jaune) 0%,
                rgba(245, 184, 0, 0.18) 100%);
}}

/* ------------------------------------------------------------ CARTES -- */
.glass-card {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(14px) saturate(116%);
    backdrop-filter: blur(14px) saturate(116%);
    border: 1px solid var(--verre-bord);
    border-radius: var(--rayon);
    box-shadow: var(--verre-ombre);
    padding: 18px 22px;
    margin-bottom: 14px;
    color: var(--abb-txt);
    font-size: 0.94rem;
    line-height: 1.6;
}}
.glass-card.solid {{ background: var(--verre-fort); }}
.glass-card .gc-title {{
    font-weight: 650; color: var(--abb-brun); font-size: 0.99rem;
    margin-bottom: 6px;
}}

.step-card {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(14px);
    backdrop-filter: blur(14px);
    border: 1px solid var(--verre-bord);
    border-radius: var(--rayon);
    box-shadow: var(--verre-ombre);
    padding: 17px 18px 16px 18px;
    min-height: 152px;
    position: relative;
    transition: transform .18s ease, box-shadow .18s ease;
}}
.step-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(38, 29, 22, 0.24);
}}
.step-card .sc-num {{
    font-size: 0.73rem; font-weight: 700; letter-spacing: .1em;
    color: #8A6B00; text-transform: uppercase;
}}
.step-card .sc-title {{
    font-weight: 650; color: var(--abb-brun); font-size: 1rem;
    margin: 4px 0 7px 0;
}}
.step-card .sc-desc {{ font-size: 0.83rem; color: var(--abb-mut); line-height: 1.5; }}
.step-card .sc-rule {{
    width: 26px; height: 2px; background: var(--abb-jaune);
    border-radius: 2px; margin-bottom: 9px;
}}

/* ---------------------------------------------------------- KPI CARD -- */
.kpi {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(14px) saturate(116%);
    backdrop-filter: blur(14px) saturate(116%);
    border: 1px solid var(--verre-bord);
    border-radius: var(--rayon);
    box-shadow: var(--verre-ombre);
    padding: 18px 18px 16px 18px;
    min-height: 108px;
    display: flex; flex-direction: column; justify-content: center;
}}
.kpi .val {{
    font-size: 1.78rem; font-weight: 700; color: var(--abb-brun);
    line-height: 1.12; letter-spacing: -0.02em;
}}
.kpi .lab {{
    font-size: 0.79rem; color: var(--abb-mut); margin-top: 6px; line-height: 1.35;
}}
.kpi .accent {{
    width: 24px; height: 2px; background: var(--abb-jaune);
    border-radius: 2px; margin-bottom: 11px;
}}

/* --------------------------------------------------- INTERPRÉTATIONS -- */
.insight-card {{
    background: var(--creme);
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 250, 240, 0.50);
    border-left: 3px solid var(--abb-jaune);
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px 0 18px 0;
    font-size: 0.895rem; line-height: 1.6; color: var(--abb-txt);
}}
.insight-card .ic-label {{
    display: block; font-size: 0.7rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #8A6B00; margin-bottom: 5px;
}}
.warn-card {{
    background: var(--creme);
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 250, 240, 0.48);
    border-left: 3px solid {ROUGE};
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px 0 18px 0;
    font-size: 0.895rem; line-height: 1.6; color: var(--abb-txt);
}}
.warn-card .ic-label {{
    display: block; font-size: 0.7rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: {ROUGE}; margin-bottom: 5px;
}}

/* ------------------------------------------------------------ BADGES -- */
.badge {{
    display: inline-block; padding: 4px 13px; border-radius: 999px;
    font-weight: 650; font-size: 0.785rem; letter-spacing: .01em;
    border: 1px solid rgba(0, 0, 0, 0.07);
}}

/* ---------------------------------------------------------- SIDEBAR -- */
/* même famille bronze, volontairement plus sombre que la zone de résultats */
[data-testid="stSidebar"] {{
    background: var(--brun);
    border-right: 1px solid rgba(43, 35, 32, 0.20);
}}
[data-testid="stSidebar"] * {{ color: #1C1611; }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: .6rem; }}

.sb-logo {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 13px 10px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(43, 35, 32, 0.24);
    margin-bottom: 13px;
}}
.sb-logo img {{ width: 142px; max-width: 100%; }}
.sb-title {{
    text-align: center; font-weight: 700; font-size: 1.03rem;
    color: #1C1611; letter-spacing: -0.01em;
}}
.sb-sub {{
    text-align: center; font-size: 0.745rem; color: rgba(28, 22, 17, 0.72);
    letter-spacing: .05em; text-transform: uppercase; margin-top: 3px;
}}
.sb-sep {{ height: 1px; background: rgba(28, 22, 17, 0.20); margin: 16px 0 12px 0; }}
.sb-legend {{
    font-size: 0.665rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: rgba(28, 22, 17, 0.62);
    margin: 0 0 9px 2px;
}}

[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] div[role="radiogroup"] label {{
    display: flex; align-items: center;
    padding: 8px 12px; margin: 0 0 2px 0;
    border-radius: 10px;
    border-left: 2px solid transparent;
    cursor: pointer;
    transition: background .16s ease, border-color .16s ease, padding .16s ease;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(28, 22, 17, 0.09);
    border-left-color: rgba(245, 184, 0, 0.55);
}}
[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    font-size: 0.88rem; margin: 0; color: rgba(28, 22, 17, 0.92);
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: rgba(255, 224, 207, 0.55);
    border-left-color: var(--abb-jaune);
    padding-left: 14px;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
    color: #120E0B; font-weight: 650;
}}
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] > div:first-child {{
    display: none;
}}

.state-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px; border-radius: 9px; margin-bottom: 3px;
    background: rgba(28, 22, 17, 0.07);
}}
.state-row .sr-lab {{ font-size: 0.79rem; color: rgba(28, 22, 17, 0.88); }}
.state-row .sr-val {{
    font-size: 0.715rem; letter-spacing: .04em; display: flex;
    align-items: center; gap: 7px;
}}
.dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.dot-on {{ background: #0E5C33; box-shadow: 0 0 0 3px rgba(14, 92, 51, 0.20); }}
.dot-off {{ background: transparent; border: 1.5px solid rgba(28, 22, 17, 0.50); }}
.sr-on {{ color: #0E5C33; }}
.sr-off {{ color: rgba(28, 22, 17, 0.58); }}
.sb-foot {{ font-size: 0.71rem; line-height: 1.55; color: rgba(28, 22, 17, 0.66); padding: 0 2px; }}

/* ----------------------------------------------------------- BOUTONS -- */
.stButton > button {{
    background: var(--abb-jaune);
    color: {BRUN_FONCE};
    font-weight: 600; font-size: 0.895rem;
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 11px;
    padding: 0.52rem 1.15rem;
    box-shadow: 0 2px 8px rgba(245, 184, 0, 0.26);
    transition: background .16s ease, transform .12s ease, box-shadow .16s ease;
}}
.stButton > button:hover {{
    background: var(--abb-jaune-fonce); color: {BRUN_FONCE};
    border-color: rgba(0, 0, 0, 0.09);
    box-shadow: 0 4px 14px rgba(217, 162, 0, 0.32);
    transform: translateY(-1px);
}}
.stButton > button:focus {{
    box-shadow: 0 0 0 3px rgba(245, 184, 0, 0.32) !important; color: {BRUN_FONCE};
}}
.stButton > button:active {{ transform: translateY(0); }}

.stDownloadButton > button {{
    background: var(--creme);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    color: var(--abb-brun);
    font-weight: 600; font-size: 0.87rem;
    border: 1px solid rgba(58, 47, 41, 0.24);
    border-radius: 11px;
    padding: 0.5rem 1.05rem;
    box-shadow: none;
    transition: background .16s ease, border-color .16s ease;
}}
.stDownloadButton > button:hover {{
    background: #FFF1E7;
    border-color: var(--abb-jaune); color: var(--abb-brun);
}}
.stDownloadButton > button:focus {{
    box-shadow: 0 0 0 3px rgba(245, 184, 0, 0.28) !important; color: var(--abb-brun);
}}

/* ------------------------------------------------ CHAMPS & CONTRÔLES -- */
[data-testid="stMain"] label, .main label,
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{
    font-size: 0.855rem; font-weight: 550; color: var(--abb-brun) !important;
}}
[data-testid="stSidebar"] label p {{ color: #1C1611 !important; }}

[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] input, .main [data-baseweb="select"] > div, .main input {{
    background: var(--creme);
    border: 1px solid rgba(173, 122, 79, 0.42);
    border-radius: 10px;
    color: var(--abb-txt);
}}
[data-testid="stMain"] [data-baseweb="select"] > div:focus-within,
[data-testid="stMain"] input:focus {{
    border-color: var(--abb-jaune);
    box-shadow: 0 0 0 3px rgba(245, 184, 0, 0.20);
}}
/* jetons du multiselect — étiquette jaune Al Barid, jamais la couleur par défaut */
[data-baseweb="tag"] {{
    background-color: {JAUNE_CLAIR} !important;
    color: {BRUN} !important;
    border: 1px solid rgba(245, 184, 0, 0.55) !important;
    border-radius: 8px !important;
}}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {{ color: {BRUN} !important; fill: {BRUN} !important; }}

[data-testid="stFileUploaderDropzone"] {{
    background: var(--creme);
    border: 1.5px dashed rgba(173, 122, 79, 0.50);
    border-radius: 14px;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--abb-jaune); background: #FFF1E7;
}}
[data-testid="stExpander"] {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    border: 1px solid var(--verre-bord);
    border-radius: 14px;
    box-shadow: var(--verre-ombre);
    overflow: hidden;
}}
[data-testid="stExpander"] summary {{ font-weight: 600; color: var(--abb-brun); font-size: 0.9rem; }}
[data-testid="stExpander"] summary:hover {{ color: #8A6B00; }}

[data-testid="stMain"] div[role="radiogroup"] label, .main div[role="radiogroup"] label {{
    background: var(--creme);
    border: 1px solid rgba(173, 122, 79, 0.38);
    border-radius: 10px;
    padding: 6px 13px;
    margin-right: 7px;
    transition: border-color .15s ease, background .15s ease;
}}
[data-testid="stMain"] div[role="radiogroup"] label:hover {{
    border-color: rgba(245, 184, 0, 0.60); background: #FFF1E7;
}}
[data-testid="stMain"] div[role="radiogroup"] label:has(input:checked) {{
    border-color: var(--abb-jaune); background: {JAUNE_CLAIR};
}}

/* curseurs — piste et poignée à la charte, au lieu de la couleur par défaut */
[data-testid="stSlider"] [role="slider"] {{
    background-color: var(--abb-jaune) !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 5px rgba(48, 36, 28, 0.28) !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background: rgb"] {{
    background: var(--abb-jaune) !important;
}}
[data-testid="stTickBar"] {{ background: transparent; }}
[data-testid="stThumbValue"] {{ color: var(--abb-brun) !important; font-weight: 650; }}

/* cases à cocher */
[data-baseweb="checkbox"] [data-testid="stCheckbox"] > label > span,
[data-testid="stCheckbox"] input:checked + div {{
    background-color: var(--abb-jaune) !important;
    border-color: var(--abb-jaune) !important;
}}

[data-baseweb="tab-list"] {{
    gap: 5px; background: transparent; border-bottom: 1px solid rgba(58, 47, 41, 0.18);
}}
[data-testid="stMain"] [data-baseweb="tab"] {{
    background: rgba(251, 224, 207, 0.62);
    border-radius: 11px 11px 0 0;
    padding: 9px 17px;
    font-size: 0.885rem; font-weight: 550;
    color: var(--abb-mut);
}}
[data-testid="stMain"] [data-baseweb="tab"][aria-selected="true"] {{
    background: var(--verre-fort);
    color: var(--abb-brun);
    box-shadow: inset 0 -2px 0 0 var(--abb-jaune);
}}

/* ------------------------------------------------------- DATAFRAMES -- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background: var(--verre-fort);
    border: 1px solid var(--verre-bord);
    border-radius: 14px;
    box-shadow: var(--verre-ombre);
    padding: 5px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {{
    background: rgba(58, 47, 41, 0.30); border-radius: 8px;
}}
[data-testid="stDataFrame"] ::-webkit-scrollbar-track {{ background: transparent; }}

/* --------------------------------------------------------- MÉTRIQUES -- */
[data-testid="stMetric"] {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    border: 1px solid var(--verre-bord);
    border-radius: 14px;
    box-shadow: var(--verre-ombre);
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] p {{ font-size: 0.8rem; color: var(--abb-mut); }}
[data-testid="stMetricValue"] {{ color: var(--abb-brun); font-weight: 700; }}

/* ----------------------------------------------------------- ALERTES -- */
[data-testid="stAlert"] {{
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid var(--verre-bord);
    box-shadow: var(--verre-ombre);
    font-size: 0.9rem;
}}
[data-testid="stAlert"] p {{ color: var(--abb-txt); }}

/* --------------------------------------------------------- GRAPHIQUES -- */
[data-testid="stPlotlyChart"] {{
    background: var(--verre);
    -webkit-backdrop-filter: blur(13px) saturate(114%);
    backdrop-filter: blur(13px) saturate(114%);
    border: 1px solid var(--verre-bord);
    border-radius: var(--rayon);
    box-shadow: var(--verre-ombre);
    padding: 9px 11px 4px 11px;
    margin-bottom: 8px;
}}
[data-testid="stPlotlyChart"] .modebar {{ background: transparent !important; }}

/* ---------------------------------------------------------- RESPONSIF -- */
@media (max-width: 1400px) {{
    .kpi .val {{ font-size: 1.58rem; }}
}}
@media (max-width: 1100px) {{
    [data-testid="stMainBlockContainer"],
    [data-testid="stMain"] .block-container, .main .block-container {{
        padding: 1.5rem 1.4rem 2.2rem 1.4rem; border-radius: 18px;
    }}
    .kpi {{ min-height: 96px; }}
    .kpi .val {{ font-size: 1.42rem; }}
    .step-card {{ min-height: 0; }}
}}
@media (max-width: 760px) {{
    [data-testid="stAppViewContainer"] {{ background-attachment: scroll; }}
    [data-testid="stMainBlockContainer"],
    [data-testid="stMain"] .block-container, .main .block-container {{
        padding: 1.1rem 0.95rem 1.8rem 0.95rem;
        margin-top: .5rem; border-radius: 14px;
    }}
    .page-header {{ padding: 15px 17px 14px 17px; }}
    .page-header .ph-title {{ font-size: 1.16rem; }}
    [data-testid="stMain"] h1, .main h1 {{ font-size: 1.3rem; }}
    .kpi .val {{ font-size: 1.3rem; }}
    .kpi .lab {{ font-size: 0.73rem; }}
    .glass-card {{ padding: 14px 15px; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
# 0.3 THÈME PLOTLY — graphiques intégrés aux cartes vitrées
# ============================================================================
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, 'Segoe UI', Arial", color=TXT, size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=46, r=22, t=54, b=44),
    title=dict(font=dict(size=14.5, color=BRUN, family="Inter, system-ui, Arial"),
               x=0.01, xanchor="left"),
    xaxis=dict(gridcolor="rgba(58,47,41,0.16)", zerolinecolor="rgba(58,47,41,0.26)",
               linecolor="rgba(58,47,41,0.30)", tickfont=dict(size=11, color="#4A3C31"),
               title=dict(font=dict(size=11.5, color="#4A3C31"))),
    yaxis=dict(gridcolor="rgba(58,47,41,0.16)", zerolinecolor="rgba(58,47,41,0.26)",
               linecolor="rgba(58,47,41,0.30)", tickfont=dict(size=11, color="#4A3C31"),
               title=dict(font=dict(size=11.5, color="#4A3C31"))),
    legend=dict(bgcolor="rgba(251,224,207,0.92)", bordercolor="rgba(173,122,79,0.40)",
                borderwidth=1, font=dict(size=11, color=TXT)),
    hoverlabel=dict(bgcolor="#FBE0CF", bordercolor=JAUNE,
                    font=dict(color=TXT, size=11.5, family="Inter, Arial")),
)


def apply_plotly_theme(fig, **surcharges):
    """Applique la charte graphique à une figure Plotly.

    Les surcharges (titre, libellés d'axes, etc.) sont fusionnées avec le
    thème : elles ne modifient jamais les DONNÉES de la figure.
    """
    params = {k: v for k, v in PLOTLY_LAYOUT.items()}
    for cle, valeur in surcharges.items():
        if cle in ("xaxis", "yaxis") and isinstance(valeur, dict):
            fusion = dict(params.get(cle, {}))
            fusion.update(valeur)
            params[cle] = fusion
        else:
            params[cle] = valeur
    fig.update_layout(**params)
    return fig


# ============================================================================
# 0.4 COMPOSANTS D'INTERFACE RÉUTILISABLES
# ============================================================================
def render_page_header(titre: str, sous_titre: str = ""):
    """En-tête vitré d'une page — remplace les anciens bandeaux pleins."""
    sub = f"<div class='ph-sub'>{sous_titre}</div>" if sous_titre else ""
    st.markdown(
        f"<div class='page-header'><div class='ph-title'>{titre}</div>{sub}"
        f"<div class='ph-rule'></div></div>", unsafe_allow_html=True)


def render_glass_card(contenu: str, titre: str = "", solide: bool = False):
    """Carte vitrée générique. `solide` renforce l'opacité pour le texte dense."""
    classe = "glass-card solid" if solide else "glass-card"
    entete = f"<div class='gc-title'>{titre}</div>" if titre else ""
    st.markdown(f"<div class='{classe}'>{entete}{contenu}</div>",
                unsafe_allow_html=True)


def render_insight(html: str, label: str = "Interprétation"):
    """Bloc d'interprétation statistique — distinct sans être une alerte."""
    st.markdown(f"<div class='insight-card'><span class='ic-label'>{label}</span>"
                f"{html}</div>", unsafe_allow_html=True)


def render_warning(html: str, label: str = "Point de vigilance"):
    """Bloc d'avertissement méthodologique."""
    st.markdown(f"<div class='warn-card'><span class='ic-label'>{label}</span>"
                f"{html}</div>", unsafe_allow_html=True)


def render_kpi_card(valeur: str, libelle: str):
    """Une carte KPI isolée (à placer dans une colonne)."""
    return (f"<div class='kpi'><div class='accent'></div>"
            f"<div class='val'>{valeur}</div>"
            f"<div class='lab'>{libelle}</div></div>")


def render_kpi_row(items):
    """Ligne de KPI homogènes : items = [(valeur, libellé), ...]."""
    cols = st.columns(len(items))
    for c, (val, lib) in zip(cols, items):
        c.markdown(render_kpi_card(val, lib), unsafe_allow_html=True)


def render_badge(classe_risque: str, prefixe: str = "") -> str:
    """Capsule de classe de risque — fond très clair, texte foncé."""
    txt = f"{prefixe}{classe_risque}" if prefixe else classe_risque
    return (f"<span class='badge' style='background:{FONDS_CLASSES[classe_risque]};"
            f"color:{COULEURS_CLASSES[classe_risque]}'>{txt}</span>")


def render_status_indicator(libelle: str, actif: bool,
                            texte_actif: str, texte_inactif: str) -> str:
    """Ligne d'état du pipeline — points CSS, aucun emoji."""
    cls_dot = "dot dot-on" if actif else "dot dot-off"
    cls_txt = "sr-on" if actif else "sr-off"
    texte = texte_actif if actif else texte_inactif
    return (f"<div class='state-row'><span class='sr-lab'>{libelle}</span>"
            f"<span class='sr-val {cls_txt}'><span class='{cls_dot}'></span>"
            f"{texte}</span></div>")


# ---- compatibilité : anciens noms conservés pour ne rien casser ------------
def bandeau(titre, sous_titre=""):
    render_page_header(titre, sous_titre)


def interpretation(html):
    render_insight(html)


def alerte(html):
    render_warning(html)


def kpi_row(items):
    render_kpi_row(items)


def fr(x, dec=1):
    return f"{x*100:.{dec}f}".replace(".", ",") + " %"


def num_fr(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


# ============================================================================
# 1. MOTEUR DE DONNÉES : SIMULATION & IMPORT CSV (cible jamais imputée)
#    -- logique strictement identique à la v2 --
# ============================================================================
VARS_FR = {
    "age": "Âge", "anciennete": "Ancienneté (mois)",
    "nb_produits": "Nombre de produits", "solde_moyen": "Solde moyen (DH)",
    "ln_solde": "ln(Solde moyen)", "nb_transactions": "Transactions (6 mois)",
    "bbm": "Usage Barid Bank Mobile", "salaire_dom": "Salaire domicilié",
    "credit": "Crédit en cours", "reclamations": "Réclamations (12 mois)",
    "churn": "Churn (cible)",
}
FEATURES_DEFAUT = ["age", "anciennete", "nb_produits", "ln_solde",
                   "nb_transactions", "bbm", "salaire_dom", "credit",
                   "reclamations"]


def lab(col):
    return VARS_FR.get(col, col.replace("_", " ").capitalize())


def lab_colonne_encodee(col, cat_cols):
    """Rend lisible une colonne one-hot encodée : 'pays_Germany' -> 'Pays = Germany'."""
    for c in cat_cols:
        if col == c:
            return lab(c)
        if col.startswith(c + "_"):
            return f"{lab(c)} = {col[len(c) + 1:]}"
    return lab(col)


@st.cache_data(show_spinner=False)
def simuler_donnees(n: int, graine: int) -> pd.DataFrame:
    """Générateur identique à celui du rapport (n=4000, graine=42 -> chiffres
    du mémoire reproduits à l'identique)."""
    rng = np.random.default_rng(graine)
    age = np.clip(rng.normal(40, 13, n), 18, 78).round(0)
    anciennete = np.clip(rng.exponential(48, n), 1, 200).round(0)
    nb_produits = rng.choice([1, 2, 3, 4], n, p=[0.42, 0.33, 0.17, 0.08])
    solde = np.clip(rng.lognormal(8.3, 1.0, n), 0, 250000).round(0)
    nb_trans = np.clip(rng.poisson(14, n) + (nb_produits - 1) * 4, 0, None)
    bbm = rng.binomial(1, 0.52, n)
    sal = rng.binomial(1, 0.46, n)
    cred = rng.binomial(1, 0.22, n)
    recl = rng.poisson(0.5, n)
    lin = (0.30 - 0.014 * anciennete - 0.55 * (nb_produits - 1)
           - 0.85 * bbm - 0.75 * sal - 0.90 * cred + 0.72 * recl
           - 0.050 * nb_trans - 0.012 * (age - 40)
           - 0.45 * (np.log1p(solde) - 8.3))
    churn = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    return pd.DataFrame({
        "id_client": np.arange(1, n + 1),
        "age": age.astype(int), "anciennete": anciennete.astype(int),
        "nb_produits": nb_produits, "solde_moyen": solde.astype(int),
        "ln_solde": np.log1p(solde), "nb_transactions": nb_trans,
        "bbm": bbm, "salaire_dom": sal, "credit": cred,
        "reclamations": recl, "churn": churn})


def lire_csv(fichier) -> pd.DataFrame:
    contenu = fichier.getvalue()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contenu), sep=None, engine="python",
                               encoding=enc)
        except Exception:
            continue
    raise ValueError("Format de fichier non reconnu.")


def colonne_id(df):
    for c in df.columns:
        if c.lower() in ("id_client", "customerid", "id"):
            return c
    return None


# ============================================================================
# 2. PRÉTRAITEMENT : COLUMNTRANSFORMER RÉUTILISABLE (AUCUNE FUITE)
# ============================================================================
def types_variables(df, features):
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols


def construire_preprocesseur(num_cols, cat_cols, standardiser=False):
    """ColumnTransformer neuf (non ajusté) : imputation + encodage + mise à
    l'échelle optionnelle. Il doit être ajusté (fit) UNIQUEMENT sur le train ;
    partout ailleurs (validation, test, simulateur, portefeuille complet), on
    appelle exclusivement .transform() sur l'objet déjà ajusté."""
    etapes_num = [("imputer", SimpleImputer(strategy="median"))]
    if standardiser:
        etapes_num.append(("scaler", StandardScaler()))
    pipe_num = Pipeline(etapes_num)
    pipe_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first",
                                  sparse_output=False)),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", pipe_num, num_cols))
    if cat_cols:
        transformers.append(("cat", pipe_cat, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def transformer_en_df(preprocesseur_ajuste, X):
    """Applique .transform() (jamais .fit()) et restitue un DataFrame nommé."""
    arr = preprocesseur_ajuste.transform(X)
    cols = [c.split("__", 1)[1] if "__" in c else c
            for c in preprocesseur_ajuste.get_feature_names_out()]
    return pd.DataFrame(arr, columns=cols, index=X.index)


def contributions_client(x_row_enc, logit_sm, moyennes_train, cat_cols, top_n=4):
    """Décompose l'écart de logit du client par rapport à la moyenne du train,
    variable par variable — explication individuelle simple et fidèle au
    modèle retenu (régression logistique)."""
    beta = logit_sm.params.drop("const")
    ecarts = x_row_enc[beta.index] - moyennes_train[beta.index]
    contrib = (beta * ecarts).sort_values(key=lambda s: s.abs(), ascending=False)
    out = []
    for col, val in contrib.head(top_n).items():
        out.append((lab_colonne_encodee(col, cat_cols), float(val)))
    return out


SUGGESTIONS_FACTEUR = {
    "reclamations": "traiter en priorité ses réclamations récentes",
    "bbm": "l'accompagner dans l'activation de Barid Bank Mobile",
    "nb_produits": "lui proposer un produit complémentaire adapté",
    "anciennete": "renforcer le contact durant cette phase encore récente",
    "salaire": "l'inciter à domicilier son salaire ou sa pension",
    "credit": "explorer une offre de crédit adaptée",
    "solde": "revoir l'adéquation de son package à son niveau d'encours",
    "transaction": "encourager une utilisation plus régulière du compte",
    "age": "adapter le discours commercial à sa tranche d'âge",
}


def suggestion_pour(nom_lisible):
    n = nom_lisible.lower()
    for k, v in SUGGESTIONS_FACTEUR.items():
        if k in n:
            return v
    return "approfondir l'entretien sur ce facteur au prochain contact"


# ============================================================================
# 3. BARRE LATÉRALE : IDENTITÉ, NAVIGATION, ÉTAT DE LA CHAÎNE
# ============================================================================
PAGES = ["Accueil", "Données", "Exploration", "Modélisation",
         "Évaluation", "Scoring", "Dashboard", "Portefeuille clients",
         "Simulateur client", "Recommandations", "À propos"]

with st.sidebar:
    st.markdown(
        f"<div class='sb-logo'><img src='data:image/png;base64,{LOGO_B64}' "
        f"alt='Al Barid Bank'></div>"
        f"<div class='sb-title'>Churn Analytics</div>"
        f"<div class='sb-sub'>Customer Attrition Intelligence</div>"
        f"<div class='sb-sep'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-legend'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("<div class='sb-sep'></div>"
                "<div class='sb-legend'>État du pipeline</div>",
                unsafe_allow_html=True)
    ok_d = "data" in st.session_state
    ok_m = "modeles" in st.session_state
    ok_s = "scores" in st.session_state
    st.markdown(
        render_status_indicator("Données", ok_d, "Chargées", "En attente")
        + render_status_indicator("Modèles", ok_m, "Entraînés", "En attente")
        + render_status_indicator("Scoring", ok_s, "Calculé", "En attente"),
        unsafe_allow_html=True)

    st.markdown("<div class='sb-sep'></div>"
                "<div class='sb-foot'>Projet de Fin d'Année<br>"
                "Ingénierie Financière et Actuarielle<br>FST Errachidia — UMI"
                "<br><br>Stage : Al Barid Bank<br>Agence Drissia, Tanger</div>",
                unsafe_allow_html=True)


# ============================================================================
# 3.1 GARDES DE NAVIGATION
# ============================================================================
def exiger_donnees():
    if "data" not in st.session_state:
        st.info("Commencez par charger ou simuler des données dans le module "
                "**Données**.")
        st.stop()


def exiger_modele():
    exiger_donnees()
    if "modeles" not in st.session_state:
        st.info("Entraînez d'abord les modèles dans le module **Modélisation**.")
        st.stop()


def exiger_scores():
    exiger_modele()
    if "scores" not in st.session_state:
        st.info("Calculez d'abord le scoring du portefeuille dans le module "
                "**Scoring**.")
        st.stop()


# ============================================================================
# 4. PAGE ACCUEIL
# ============================================================================
if page == "Accueil":
    st.markdown(
        "<div class='page-header' style='padding:30px 32px 28px 32px;'>"
        "<div style='font-size:0.7rem;font-weight:700;letter-spacing:.14em;"
        f"text-transform:uppercase;color:{JAUNE_FONCE};margin-bottom:9px;'>"
        "Al Barid Bank — Agence Drissia, Tanger</div>"
        "<div style='font-size:2.1rem;font-weight:700;color:" + BRUN +
        ";letter-spacing:-0.025em;line-height:1.16;'>Customer Churn Analytics</div>"
        "<div style='font-size:1.02rem;color:" + MUT + ";margin-top:9px;"
        "max-width:680px;line-height:1.55;'>Analyse, prédiction et pilotage du "
        "risque d'attrition client</div>"
        "<div class='ph-rule' style='width:74px;margin-top:18px;'></div>"
        "<div style='font-size:0.79rem;color:" + MUT + ";margin-top:14px;'>"
        "Projet de Fin d'Année · Ingénierie Financière et Actuarielle</div>"
        "</div>", unsafe_allow_html=True)

    render_glass_card(
        "Comment identifier, de manière précoce et objective, les clients "
        "susceptibles de quitter Al Barid Bank, afin de mettre en place des "
        "actions préventives de rétention ciblées et efficaces&nbsp;?",
        titre="Problématique")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### Chaîne de traitement")
    cols = st.columns(5)
    etapes = [
        ("Étape 1", "Données", "Simulez un portefeuille ou importez votre CSV réel."),
        ("Étape 2", "Modélisation", "Pipeline scikit-learn, split train/validation/test, CV."),
        ("Étape 3", "Évaluation", "ROC, calibration, seuil métier vs seuil statistique."),
        ("Étape 4", "Scoring", "Score de fidélité 300-900 et 4 classes de risque."),
        ("Étape 5", "Actions", "Recommandations personnalisées par client."),
    ]
    for c, (num, t, d) in zip(cols, etapes):
        c.markdown(f"<div class='step-card'><div class='sc-rule'></div>"
                   f"<div class='sc-num'>{num}</div>"
                   f"<div class='sc-title'>{t}</div>"
                   f"<div class='sc-desc'>{d}</div></div>",
                   unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("### Repères issus du mémoire")
    st.caption("Base de référence : portefeuille de 4 000 clients")
    render_kpi_row([("13,9 %", "Taux d'attrition observé"),
                    ("0,805", "AUC — régression logistique"),
                    ("85,5 %", "Churners détectés (rappel)"),
                    ("× 2,04", "Cote de churn par réclamation")])
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    render_insight("Cette édition applique une discipline de validation "
                   "<b>train / validation / test</b> stricte, des "
                   "<b>Pipelines scikit-learn</b> sans fuite de données, une "
                   "<b>vérification de calibration</b> des probabilités, et "
                   "des explications <b>individuelles</b> par client — au-delà "
                   "de la simple classe de risque.")

# ============================================================================
# 5. PAGE DASHBOARD
# ============================================================================
elif page == "Dashboard":
    render_page_header("Tableau de bord",
                       "Vue d'ensemble du portefeuille scoré")
    exiger_scores()
    scored = st.session_state["scores"]
    cible = st.session_state["target"]
    comptes = scored["classe_risque"].value_counts()
    n = len(scored)

    render_kpi_row([
        (f"{n:,}".replace(",", " "), "Clients au portefeuille"),
        (fr(scored[cible].mean()), "Taux d'attrition global"),
        (f"{int(comptes.get('Élevé', 0)):,}".replace(",", " "),
         f"Risque élevé ({fr(comptes.get('Élevé', 0)/n)})"),
        (f"{int(comptes.get('Critique', 0)):,}".replace(",", " "),
         f"Risque critique ({fr(comptes.get('Critique', 0)/n)})"),
    ])
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        ordre = ["Faible", "Modéré", "Élevé", "Critique"]
        vals = [int(comptes.get(c, 0)) for c in ordre]
        fig = go.Figure(go.Pie(labels=ordre, values=vals, hole=0.58,
                               marker=dict(colors=[COULEURS_CLASSES[c] for c in ordre],
                                           line=dict(color="rgba(255,255,255,0.85)",
                                                     width=2)),
                               textinfo="label+percent",
                               textfont=dict(size=11.5)))
        apply_plotly_theme(fig, title="Portefeuille par classe de risque",
                           showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        g = scored.groupby("classe_risque", observed=True)[cible].mean().reindex(
            ["Faible", "Modéré", "Élevé", "Critique"])
        fig = go.Figure(go.Bar(x=g.index, y=g.values * 100,
                               marker_color=[COULEURS_CLASSES[c] for c in g.index],
                               marker_line=dict(color="rgba(255,255,255,0.6)", width=1),
                               text=[f"{v*100:.1f}%" for v in g.values],
                               textposition="outside",
                               textfont=dict(size=11, color=BRUN)))
        apply_plotly_theme(fig, title="Taux de churn observé par classe",
                           yaxis=dict(title=dict(text="Taux de churn (%)")))
        st.plotly_chart(fig, width="stretch")
    render_insight("Le tableau de bord se met à jour automatiquement à chaque "
                   "nouveau calcul du scoring — c'est la vue destinée à un "
                   "comité de pilotage ou à un directeur d'agence, sans détail "
                   "technique.")

# ============================================================================
# 6. PAGE DONNÉES
# ============================================================================
elif page == "Données":
    render_page_header("Données",
                       "Portefeuille simulé paramétrable ou import d'un CSV réel")
    mode = st.radio("Source des données", ["Simulation (paramétrable)",
                    "Import d'un fichier CSV réel"], horizontal=True)

    if mode == "Simulation (paramétrable)":
        c1, c2, c3 = st.columns([2, 1, 1])
        n = c1.slider("Nombre de clients simulés", 500, 20000, 4000, step=500)
        graine = c2.number_input("Graine aléatoire", 0, 9999, 42)
        c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if c3.button("Générer le portefeuille", width="stretch"):
            df = simuler_donnees(n, int(graine))
            st.session_state["data"] = df
            st.session_state["source"] = f"Simulation — {n} clients (graine {graine})"
            st.session_state["target"] = "churn"
            st.session_state["features"] = FEATURES_DEFAUT.copy()
            for k in ("modeles", "scores"):
                st.session_state.pop(k, None)
            st.success("Portefeuille généré (n = 4 000, graine = 42 reproduit "
                       "exactement les chiffres du mémoire).")
    else:
        render_glass_card(
            "Le fichier doit contenir une colonne cible binaire (0/1) et des "
            "variables explicatives. <b>Les lignes dont la cible est manquante "
            "ou non binaire sont supprimées</b>, jamais imputées à 0 — imputer "
            "une cible inconnue biaiserait l'apprentissage.")
        up = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
        if up is not None:
            try:
                brut = lire_csv(up)
                st.dataframe(brut.head(8), width="stretch")
                colonnes = list(brut.columns)
                cible_defaut = "churn" if "churn" in colonnes else colonnes[-1]
                cible = st.selectbox("Colonne cible (churn : 0 = actif, "
                                     "1 = churner)", colonnes,
                                     index=colonnes.index(cible_defaut))
                candidates = [c for c in colonnes if c != cible]
                feats = st.multiselect("Variables explicatives", candidates,
                                       default=candidates)
                if st.button("Valider ce jeu de données"):
                    y_brut = pd.to_numeric(brut[cible], errors="coerce")
                    valides = y_brut.isin([0, 1])
                    n_perdues = (~valides).sum()
                    if len(feats) == 0:
                        st.error("Sélectionnez au moins une variable explicative.")
                    elif valides.sum() == 0:
                        st.error("Aucune ligne n'a une cible binaire (0/1) exploitable.")
                    else:
                        df = brut.loc[valides].copy()
                        df[cible] = y_brut.loc[valides].astype(int)
                        st.session_state["data"] = df
                        st.session_state["source"] = f"CSV importé — {up.name}"
                        st.session_state["target"] = cible
                        st.session_state["features"] = feats
                        for k in ("modeles", "scores"):
                            st.session_state.pop(k, None)
                        if n_perdues > 0:
                            st.warning(f"{n_perdues} ligne(s) avec une cible "
                                       f"manquante ou non binaire ont été "
                                       f"**supprimées** (non imputées).")
                        st.success(f"Jeu de données validé — {len(df)} lignes "
                                   f"conservées.")
            except Exception as e:
                st.error(f"Lecture impossible : {e}")

    if "data" in st.session_state:
        df = st.session_state["data"]
        cible = st.session_state["target"]
        st.markdown("---")
        st.markdown("### Aperçu du jeu de données actif")
        st.caption(st.session_state["source"])
        render_kpi_row([(f"{len(df):,}".replace(",", " "), "Clients"),
                        (str(len(st.session_state['features'])),
                         "Variables explicatives"),
                        (f"{int(df[cible].sum()):,}".replace(",", " "), "Churners"),
                        (fr(df[cible].mean()), "Taux d'attrition")])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), width="stretch")
        with st.expander("Dictionnaire des variables (base simulée)"):
            st.table(pd.DataFrame([(k, v) for k, v in VARS_FR.items()],
                                  columns=["Champ", "Description"]))
        st.download_button("Télécharger le jeu de données (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           "portefeuille_churn.csv", "text/csv")

# ============================================================================
# 7. PAGE EXPLORATION
# ============================================================================
elif page == "Exploration":
    render_page_header("Analyse exploratoire",
                       "Profils, distributions et facteurs associés à l'attrition")
    exiger_donnees()
    df = st.session_state["data"]
    cible = st.session_state["target"]
    feats = st.session_state["features"]
    taux = df[cible].mean()

    render_kpi_row([(f"{len(df):,}".replace(",", " "), "Clients"),
                    (f"{int(df[cible].sum()):,}".replace(",", " "), "Churners"),
                    (fr(taux), "Taux d'attrition"),
                    (f"{int((1-taux)*len(df)):,}".replace(",", " "),
                     "Clients actifs")])
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(go.Pie(labels=["Clients actifs", "Churners"],
                               values=[1 - taux, taux], hole=0.58,
                               marker=dict(colors=[JAUNE, BRUN],
                                           line=dict(color="rgba(255,255,255,0.85)",
                                                     width=2)),
                               textinfo="label+percent",
                               textfont=dict(size=11.5)))
        apply_plotly_theme(fig, title="Répartition du portefeuille",
                           showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        var = st.selectbox("Taux de churn selon…",
                           [f for f in feats if f in df.columns], format_func=lab)
        serie = df[var]
        if serie.nunique() > 8 and pd.api.types.is_numeric_dtype(serie):
            groupes = pd.qcut(serie, 4, duplicates="drop")
            g = df.groupby(groupes, observed=True)[cible].mean()
        else:
            g = df.groupby(serie, observed=True)[cible].mean()
        fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values * 100,
                               marker_color=JAUNE,
                               marker_line=dict(color=BRUN, width=1.1),
                               text=[f"{v*100:.1f}%" for v in g.values],
                               textposition="outside",
                               textfont=dict(size=11, color=BRUN)))
        apply_plotly_theme(fig, title=f"Taux de churn selon {lab(var).lower()}",
                           yaxis=dict(title=dict(text="Taux de churn (%)")))
        st.plotly_chart(fig, width="stretch")
        ecart = g.max() - g.min()
        render_insight(f"Le taux de churn varie de <b>{fr(g.min())}</b> à "
                       f"<b>{fr(g.max())}</b> selon <b>{lab(var).lower()}</b>, "
                       f"soit un écart de {fr(ecart)}.")

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        num_disp = [f for f in feats if f in df.columns
                    and pd.api.types.is_numeric_dtype(df[f])]
        var2 = st.selectbox("Distribution comparée (actifs vs churners)", num_disp,
                            format_func=lab)
        fig = go.Figure()
        fig.add_histogram(x=df.loc[df[cible] == 0, var2], name="Actifs",
                          marker_color=JAUNE, opacity=0.72, nbinsx=35)
        fig.add_histogram(x=df.loc[df[cible] == 1, var2], name="Churners",
                          marker_color=ROUGE, opacity=0.72, nbinsx=35)
        apply_plotly_theme(fig, barmode="overlay",
                           title=f"Distribution — {lab(var2)}")
        st.plotly_chart(fig, width="stretch")
    with c4:
        corr = df[num_disp + [cible]].corr()[cible].drop(cible).sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=[lab(i) for i in corr.index],
                               orientation="h",
                               marker_color=[VERT if v < 0 else ROUGE
                                             for v in corr.values]))
        apply_plotly_theme(fig, title="Corrélation de chaque variable avec le churn",
                           xaxis=dict(title=dict(text="Corrélation")))
        st.plotly_chart(fig, width="stretch")
        fuite = corr[corr.abs() > 0.9]
        if len(fuite) > 0:
            render_warning(
                f"<b>Risque de fuite de données.</b> "
                f"{', '.join(lab(i) for i in fuite.index)} présente une "
                f"corrélation supérieure à 0,9 avec la cible — c'est "
                f"anormalement élevé pour un facteur prédictif authentique. "
                f"Vérifiez qu'elle n'est pas mesurée <i>après</i> le départ "
                f"du client avant de l'inclure dans le modèle (module "
                f"Modélisation).")
    render_insight("Les barres <span style='color:#1E8449'><b>vertes</b></span> "
                   "sont des facteurs de <b>rétention</b>, les "
                   "<span style='color:#B03A2E'><b>rouges</b></span> des "
                   "facteurs de <b>risque</b>. Ce diagnostic descriptif reste "
                   "valable même pour une variable exclue du modèle prédictif.")

# ============================================================================
# 8. PAGE MODÉLISATION
# ============================================================================
elif page == "Modélisation":
    render_page_header(
        "Modélisation",
        "Pipeline scikit-learn, split train/validation/test, validation croisée")
    exiger_donnees()
    df = st.session_state["data"]
    cible = st.session_state["target"]

    c1, c2, c3 = st.columns([1.5, 1, 1])
    feats = c1.multiselect("Variables du modèle", st.session_state["features"],
                           default=st.session_state["features"], format_func=lab)
    standardiser = c2.checkbox("Standardiser les variables numériques", value=False,
                               help="Sans effet sur l'AUC du logit non pénalisé ; "
                               "modifie l'échelle des odds ratios si activé.")
    graine = c3.number_input("Graine (partition & modèles)", 0, 9999, 42)

    c4, c5 = st.columns(2)
    part_val = c4.slider("Part de l'échantillon de validation", 0.10, 0.30, 0.20, 0.05)
    part_test = c5.slider("Part de l'échantillon de test", 0.10, 0.30, 0.20, 0.05)
    st.caption(f"Répartition retenue : train {fr(1-part_val-part_test, 0)} · "
               f"validation {fr(part_val, 0)} · test {fr(part_test, 0)}. Le "
               f"seuil de classification sera choisi sur la validation ; le "
               f"test sert uniquement à l'évaluation finale, jamais aux choix "
               f"de réglage.")

    if st.button("Entraîner les modèles", width="stretch"):
        if not feats:
            st.error("Sélectionnez au moins une variable.")
            st.stop()
        if part_val + part_test >= 0.7:
            st.error("Les parts validation + test sont trop élevées.")
            st.stop()

        X = df[feats].copy()
        y = df[cible].astype(int)
        num_cols, cat_cols = types_variables(df, feats)

        # ---------------- split en 3 : train / validation / test (stratifié) --
        X_tv, X_te, y_tv, y_te = train_test_split(
            X, y, test_size=part_test, random_state=int(graine), stratify=y)
        part_val_rel = part_val / (1 - part_test)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tv, y_tv, test_size=part_val_rel, random_state=int(graine),
            stratify=y_tv)

        # ---------------- Pipelines scikit-learn (prétraitement + modèle) -----
        pre_template = construire_preprocesseur(num_cols, cat_cols, standardiser)
        pipe_logit = Pipeline([("prep", clone(pre_template)),
                               ("clf", LogisticRegression(max_iter=3000, C=1e6))])
        pipe_arbre = Pipeline([("prep", clone(pre_template)),
                               ("clf", DecisionTreeClassifier(
                                   max_depth=5, min_samples_leaf=max(20, len(X_tr)//60),
                                   random_state=42))])
        pipe_foret = Pipeline([("prep", clone(pre_template)),
                               ("clf", RandomForestClassifier(
                                   n_estimators=400, max_depth=8,
                                   min_samples_leaf=max(10, len(X_tr)//150),
                                   random_state=42))])
        pipe_logit.fit(X_tr, y_tr)
        pipe_arbre.fit(X_tr, y_tr)
        pipe_foret.fit(X_tr, y_tr)

        # ---------------- couche interprétable : statsmodels sur le MÊME
        #                  prétraitement (déjà ajusté sur le train, jamais
        #                  ré-ajusté ensuite) --------------------------------
        pre_fit = pipe_logit.named_steps["prep"]
        Xtr_enc = transformer_en_df(pre_fit, X_tr)
        Xval_enc = transformer_en_df(pre_fit, X_val)
        Xte_enc = transformer_en_df(pre_fit, X_te)
        logit_sm, note = None, ""
        try:
            logit_sm = sm.Logit(y_tr, sm.add_constant(Xtr_enc)).fit(disp=0)
        except Exception:
            note = ("Séparation quasi parfaite détectée pour statsmodels : "
                    "les probabilités du Pipeline scikit-learn sont utilisées "
                    "à la place (odds ratios/p-values indisponibles).")

        # ---------------- validation croisée (5 plis, sur le TRAIN seul) ------
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(graine))
        cv_res = {}
        for nom, pipe in [("Régression logistique", pipe_logit),
                          ("Arbre de décision", pipe_arbre),
                          ("Forêt aléatoire", pipe_foret)]:
            scores_cv = cross_val_score(clone(pipe), X_tr, y_tr, cv=skf,
                                        scoring="roc_auc")
            cv_res[nom] = (scores_cv.mean(), scores_cv.std())

        # ---------------- probabilités sur validation & test ------------------
        def proba_logit(Xenc):
            if logit_sm is not None:
                return np.asarray(logit_sm.predict(
                    sm.add_constant(Xenc, has_constant="add")))
            return None

        p_val = {"Régression logistique": (proba_logit(Xval_enc)
                                           if logit_sm is not None
                                           else pipe_logit.predict_proba(X_val)[:, 1]),
                 "Arbre de décision": pipe_arbre.predict_proba(X_val)[:, 1],
                 "Forêt aléatoire": pipe_foret.predict_proba(X_val)[:, 1]}
        p_test = {"Régression logistique": (proba_logit(Xte_enc)
                                            if logit_sm is not None
                                            else pipe_logit.predict_proba(X_te)[:, 1]),
                  "Arbre de décision": pipe_arbre.predict_proba(X_te)[:, 1],
                  "Forêt aléatoire": pipe_foret.predict_proba(X_te)[:, 1]}

        st.session_state["modeles"] = {
            "pipe_logit": pipe_logit, "pipe_arbre": pipe_arbre,
            "pipe_foret": pipe_foret, "pre_fit": pre_fit, "logit_sm": logit_sm,
            "num_cols": num_cols, "cat_cols": cat_cols, "feats": feats,
            "X_tr": X_tr, "X_val": X_val, "X_te": X_te,
            "y_tr": y_tr, "y_val": y_val, "y_te": y_te,
            "moyennes_train": Xtr_enc.mean(),
            "p_val": p_val, "p_test": p_test, "cv_res": cv_res, "note": note}
        st.session_state.pop("scores", None)
        st.success(f"Modèles entraînés — {len(X_tr)} en apprentissage, "
                   f"{len(X_val)} en validation, {len(X_te)} en test (partitions "
                   f"stratifiées et indépendantes).")
        if note:
            st.warning(note)

    if "modeles" in st.session_state:
        M = st.session_state["modeles"]
        st.markdown("---")
        st.markdown("### Validation croisée")
        st.caption("5 plis, échantillon d'apprentissage uniquement")
        cv_df = pd.DataFrame({nom: {"AUC moyen": m, "Écart-type": s}
                              for nom, (m, s) in M["cv_res"].items()}).T
        st.dataframe(cv_df.style.format("{:.3f}"), width="stretch")
        ecarts_cv = cv_df["Écart-type"].max()
        render_insight(f"L'écart-type maximal entre plis est de "
                       f"<b>{num_fr(ecarts_cv)}</b> — "
                       + ("un résultat stable : la performance ne dépend pas "
                          "d'un découpage particulier du train/test."
                          if ecarts_cv < 0.03 else
                          "une variabilité notable d'un pli à l'autre : à "
                          "surveiller si le portefeuille est de petite taille."))

        if M["logit_sm"] is not None:
            st.markdown("---")
            st.markdown("### Coefficients de la régression logistique")
            st.caption("Modèle interprétable — odds ratios et significativité")
            res = M["logit_sm"]
            tab = pd.DataFrame({"Coefficient": res.params, "Écart-type": res.bse,
                                "Odds ratio": np.exp(res.params),
                                "p-value": res.pvalues})
            tab.index = ["Constante" if i == "const"
                         else lab_colonne_encodee(i, M["cat_cols"]) for i in tab.index]
            st.dataframe(
                tab.style.format({"Coefficient": "{:.3f}", "Écart-type": "{:.3f}",
                                  "Odds ratio": "{:.3f}", "p-value": "{:.4f}"})
                .map(lambda v: f"color:{VERT};font-weight:700"
                     if isinstance(v, float) and v < 0.05 else "",
                     subset=["p-value"]), width="stretch")
            st.caption(f"Pseudo-R² de McFadden : {res.prsquared:.3f} · "
                       f"log-vraisemblance : {res.llf:.1f} · "
                       f"{int(res.nobs)} observations d'apprentissage.")
            orr = np.exp(res.params.drop("const")).sort_values()
            fig = go.Figure(go.Bar(
                x=orr.values,
                y=[lab_colonne_encodee(i, M["cat_cols"]) for i in orr.index],
                orientation="h",
                marker_color=[VERT if v < 1 else ROUGE for v in orr.values],
                text=[f"{v:.2f}" for v in orr.values], textposition="outside",
                textfont=dict(size=10.5, color=BRUN)))
            fig.add_vline(x=1, line_color=TXT, line_width=1.2)
            apply_plotly_theme(fig, title="Odds ratios — facteurs de risque (>1) "
                               "et de rétention (<1)",
                               xaxis=dict(title=dict(text="Odds ratio")))
            st.plotly_chart(fig, width="stretch")
            render_insight(
                f"Premier facteur de risque : "
                f"<b>{lab_colonne_encodee(orr.idxmax(), M['cat_cols']).lower()}</b> "
                f"(cote × {orr.max():.2f}). Premier facteur de rétention : "
                f"<b>{lab_colonne_encodee(orr.idxmin(), M['cat_cols']).lower()}</b> "
                f"(cote ÷ {1/orr.min():.1f}).")
        else:
            st.info("Coefficients interprétables indisponibles pour cet "
                    "entraînement (repli scikit-learn) — les odds ratios "
                    "nécessitent une convergence statsmodels.")

        st.markdown("---")
        st.markdown("### Importance des variables — forêt aléatoire")
        noms_enc = [c.split("__", 1)[1] if "__" in c else c
                    for c in M["pipe_foret"].named_steps["prep"].get_feature_names_out()]
        imp = pd.Series(M["pipe_foret"].named_steps["clf"].feature_importances_,
                        index=noms_enc).sort_values()
        fig = go.Figure(go.Bar(
            x=imp.values, y=[lab_colonne_encodee(i, M["cat_cols"]) for i in imp.index],
            orientation="h", marker_color=JAUNE,
            marker_line=dict(color=BRUN, width=1)))
        apply_plotly_theme(fig, title="Importance (réduction d'impureté, "
                           "forêt aléatoire)")
        st.plotly_chart(fig, width="stretch")
        render_insight("Cette importance complète les odds ratios : elle capte "
                       "aussi les effets non linéaires et les interactions "
                       "qu'un modèle linéaire ne peut pas représenter — un "
                       "classement très différent des odds ratios inviterait à "
                       "explorer des termes d'interaction dans le logit.")

# ============================================================================
# 9. PAGE ÉVALUATION
# ============================================================================
elif page == "Évaluation":
    render_page_header("Évaluation des modèles",
                       "Seuil choisi sur la validation, performance mesurée sur le test")
    exiger_modele()
    M = st.session_state["modeles"]
    y_val, y_te = M["y_val"], M["y_te"]

    # ---------------------------------------------------------- ROC (test) --
    fig = go.Figure()
    couleurs = {"Régression logistique": BLEU, "Forêt aléatoire": VERT,
                "Arbre de décision": MUT}
    lignes = []
    for nom, p_hat in M["p_test"].items():
        fpr, tpr, _ = roc_curve(y_te, p_hat)
        a = auc(fpr, tpr)
        lignes.append((nom, a, 2 * a - 1))
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{nom} (AUC={a:.3f})",
                        line=dict(color=couleurs[nom], width=2.4))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire (0,5)",
                    line=dict(color="#BBBBBB", dash="dash"))
    apply_plotly_theme(fig, title="Courbes ROC — échantillon de TEST (jamais "
                       "utilisé pour régler le seuil)",
                       xaxis=dict(title=dict(text="1 − spécificité")),
                       yaxis=dict(title=dict(text="Sensibilité")),
                       legend=dict(x=0.40, y=0.06,
                                   bgcolor="rgba(251,224,207,0.94)",
                                   bordercolor="rgba(173,122,79,0.40)",
                                   borderwidth=1, font=dict(size=11, color=TXT)))

    c1, c2 = st.columns([1.35, 1])
    c1.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Tableau comparatif multicritère (test)**")
        lignes_tab = []
        for nom, p_hat in M["p_test"].items():
            fpr, tpr, thr = roc_curve(y_val, M["p_val"][nom])
            s_y = float(thr[np.argmax(tpr - fpr)])
            yp = (p_hat >= s_y).astype(int)
            cm_ = confusion_matrix(y_te, yp)
            vn, fp, fn, vp = cm_.ravel()
            fpr_t, tpr_t, _ = roc_curve(y_te, p_hat)
            lignes_tab.append({
                "Modèle": nom, "AUC": auc(fpr_t, tpr_t),
                "Rappel": recall_score(y_te, yp, zero_division=0),
                "Précision": precision_score(y_te, yp, zero_division=0),
                "F1": f1_score(y_te, yp, zero_division=0),
                "KS": float((tpr - fpr).max()),
                "Interprétabilité": {"Régression logistique": "Élevée",
                                     "Arbre de décision": "Moyenne",
                                     "Forêt aléatoire": "Faible"}[nom]})
        comp = pd.DataFrame(lignes_tab).sort_values("AUC", ascending=False)
        st.dataframe(comp.set_index("Modèle").style.format(
            {"AUC": "{:.3f}", "Rappel": "{:.3f}", "Précision": "{:.3f}",
             "F1": "{:.3f}", "KS": "{:.3f}"}).highlight_max(
            subset=["AUC"], color=JAUNE_CLAIR), width="stretch")
        meilleur = comp.iloc[0]
        render_insight(f"<b>{meilleur['Modèle']}</b> obtient le meilleur AUC "
                       f"({num_fr(meilleur['AUC'])}) tout en offrant une "
                       f"interprétabilité <b>{meilleur['Interprétabilité'].lower()}</b> "
                       f"— c'est la combinaison performance + auditabilité qui "
                       f"justifie son choix comme modèle de production, plutôt "
                       f"que l'AUC seul.")

    # ---------------------------------------------- seuil (validation only) --
    st.markdown("---")
    st.markdown("### Choix du seuil — régression logistique")
    p_val_l = M["p_val"]["Régression logistique"]
    p_te_l = M["p_test"]["Régression logistique"]
    fpr_v, tpr_v, thr_v = roc_curve(y_val, p_val_l)
    s_youden = float(thr_v[np.argmax(tpr_v - fpr_v)])

    onglet_stat, onglet_metier = st.tabs(["Seuil statistique (Youden)",
                                          "Seuil métier (coûts asymétriques)"])
    with onglet_stat:
        st.metric("Seuil de Youden (calculé sur la VALIDATION)", num_fr(s_youden))
        seuil_choisi = s_youden
    with onglet_metier:
        cc1, cc2 = st.columns(2)
        cout_fn = cc1.number_input("Coût d'un churner non détecté (faux négatif)",
                                   10, 100000, 1000, step=50)
        cout_fp = cc2.number_input("Coût d'un contact inutile (faux positif)",
                                   1, 10000, 50, step=10)
        grille = np.linspace(0.02, 0.9, 150)
        couts = []
        for s in grille:
            yp = (p_val_l >= s).astype(int)
            vn, fp, fn, vp = confusion_matrix(y_val, yp).ravel()
            couts.append(fn * cout_fn + fp * cout_fp)
        s_metier = float(grille[np.argmin(couts)])
        fig = go.Figure()
        fig.add_scatter(x=grille, y=couts, mode="lines",
                        line=dict(color=BRUN, width=2.2))
        fig.add_vline(x=s_youden, line_dash="dash", line_color=MUT,
                      annotation_text="Youden")
        fig.add_vline(x=s_metier, line_dash="dash", line_color=ROUGE,
                      annotation_text="Coût minimal")
        apply_plotly_theme(fig, title="Coût total sur la validation selon le seuil",
                           xaxis=dict(title=dict(text="Seuil")),
                           yaxis=dict(title=dict(text="Coût total")))
        st.plotly_chart(fig, width="stretch")
        st.metric("Seuil optimal métier (calculé sur la VALIDATION)",
                  num_fr(s_metier))
        render_insight(f"Avec un ratio de coûts de {cout_fn/cout_fp:.0f}:1, le "
                       f"seuil métier ({num_fr(s_metier)}) "
                       + ("coïncide avec" if abs(s_metier - s_youden) < 0.02
                          else "diverge de") +
                       f" le seuil statistique de Youden ({num_fr(s_youden)}) — "
                       f"le seuil optimal <i>pour une banque</i> dépend de son "
                       f"appréciation réelle du coût d'un départ non anticipé.")
        seuil_choisi = st.radio("Seuil à appliquer pour l'évaluation finale",
                                [("Youden", s_youden), ("Métier", s_metier)],
                                format_func=lambda t: f"{t[0]} ({num_fr(t[1])})",
                                horizontal=True)[1]

    # ------------------------------------------- évaluation finale sur TEST --
    y_pred = (p_te_l >= seuil_choisi).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    vn, fp, fn, vp = cm.ravel()
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Prédit actif", "Prédit churner"],
            y=["Réel actif", "Réel churner"],
            colorscale=[[0, "#FFF6DC"], [1, BRUN]], showscale=False,
            text=[[f"VN<br>{vn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"VP<br>{vp}"]],
            texttemplate="%{text}", textfont=dict(size=14)))
        apply_plotly_theme(fig, title=f"Matrice de confusion — TEST "
                           f"(seuil = {seuil_choisi:.3f})",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = go.Figure(go.Heatmap(
            z=cm_norm, x=["Prédit actif", "Prédit churner"],
            y=["Réel actif", "Réel churner"],
            colorscale=[[0, "#FFF6DC"], [1, BRUN]], showscale=False,
            text=[[f"{cm_norm[0,0]*100:.1f}%", f"{cm_norm[0,1]*100:.1f}%"],
                  [f"{cm_norm[1,0]*100:.1f}%", f"{cm_norm[1,1]*100:.1f}%"]],
            texttemplate="%{text}", textfont=dict(size=14)))
        apply_plotly_theme(fig, title="Matrice de confusion normalisée "
                           "(% par ligne)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    render_insight(f"Sur l'échantillon de <b>test</b>, totalement indépendant "
                   f"du choix de seuil, le modèle détecte <b>{vp} des "
                   f"{vp+fn} churners</b> (rappel {fr(vp/(vp+fn))}), au prix de "
                   f"{fp} clients fidèles sollicités à tort.")

    # -------------------------------------------------------- calibration --
    st.markdown("---")
    st.markdown("### Calibration des probabilités")
    frac_pos, moy_pred = calibration_curve(y_te, p_te_l, n_bins=10,
                                           strategy="quantile")
    brier = brier_score_loss(y_te, p_te_l)
    c5, c6 = st.columns([1.3, 1])
    with c5:
        fig = go.Figure()
        fig.add_scatter(x=moy_pred, y=frac_pos, mode="lines+markers",
                        name="Modèle", line=dict(color=BRUN, width=2.4),
                        marker=dict(size=7, color=JAUNE,
                                    line=dict(color=BRUN, width=1.2)))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                        name="Calibration parfaite",
                        line=dict(color="#BBBBBB", dash="dash"))
        apply_plotly_theme(fig, title="Courbe de calibration (test, 10 quantiles)",
                           xaxis=dict(title=dict(text="Probabilité moyenne prédite")),
                           yaxis=dict(title=dict(text="Fréquence observée de churn")))
        st.plotly_chart(fig, width="stretch")
    with c6:
        st.metric("Score de Brier (plus bas = meilleur)", num_fr(brier, 4))
        render_insight("La transformation en score de fidélité (300-900) "
                       "utilise la probabilité <i>elle-même</i>, pas seulement "
                       "son rang : une calibration correcte est donc "
                       "indispensable pour que le score soit interprétable en "
                       "valeur absolue, pas seulement pour classer les "
                       "clients entre eux.")

# ============================================================================
# 10. PAGE SCORING
# ============================================================================
elif page == "Scoring":
    render_page_header("Scoring du portefeuille",
                       "Score de fidélité 300-900 et segmentation en classes de risque")
    exiger_modele()
    df = st.session_state["data"]
    cible = st.session_state["target"]
    M = st.session_state["modeles"]

    X_all = df[M["feats"]]
    Xall_enc = transformer_en_df(M["pre_fit"], X_all)   # transform seul, jamais fit
    if M["logit_sm"] is not None:
        p_all = np.asarray(M["logit_sm"].predict(
            sm.add_constant(Xall_enc, has_constant="add")))
    else:
        p_all = M["pipe_logit"].predict_proba(X_all)[:, 1]
    eps = 1e-9
    score = np.clip(600 + 50 * np.log2((1 - p_all + eps) / (p_all + eps)), 300, 900)

    def classe(s):
        return ("Faible" if s >= 700 else "Modéré" if s >= 600
                else "Élevé" if s >= 500 else "Critique")

    id_col = colonne_id(df)
    scored = df.copy()
    if id_col is None:
        scored.insert(0, "id_client", np.arange(1, len(scored) + 1))
    scored["p_churn"] = p_all
    scored["score"] = score.round(0).astype(int)
    scored["classe_risque"] = scored["score"].map(classe)
    st.session_state["scores"] = scored

    render_glass_card(
        "Score = 600 + 50 × log₂((1 − p̂) / p̂) — 600 points = cote 1 contre 1 ; "
        "chaque tranche de <b>50 points double la cote de fidélité</b>.",
        titre="Formule de score")

    fig = go.Figure()
    for a, b, c in [(300, 500, FONDS_CLASSES["Critique"]),
                    (500, 600, FONDS_CLASSES["Élevé"]),
                    (600, 700, FONDS_CLASSES["Modéré"]),
                    (700, 900, FONDS_CLASSES["Faible"])]:
        fig.add_vrect(x0=a, x1=b, fillcolor=c, opacity=0.42, line_width=0)
    fig.add_histogram(x=scored.loc[scored[cible] == 0, "score"],
                      name="Clients actifs", marker_color=BLEU, opacity=0.78,
                      nbinsx=45)
    fig.add_histogram(x=scored.loc[scored[cible] == 1, "score"],
                      name="Churners observés", marker_color=ROUGE, opacity=0.78,
                      nbinsx=45)
    for lim in (500, 600, 700):
        fig.add_vline(x=lim, line_dash="dash", line_color=MUT, line_width=1.1)
    apply_plotly_theme(fig, barmode="overlay",
                       title="Distribution des scores et zones de risque",
                       xaxis=dict(title=dict(text="Score de fidélité (300 – 900)")),
                       yaxis=dict(title=dict(text="Nombre de clients")))
    st.plotly_chart(fig, width="stretch")

    grille = (scored.groupby("classe_risque")
              .agg(Effectif=(cible, "size"), Churn_observe=(cible, "mean"),
                   Score_moyen=("score", "mean"))
              .reindex(["Faible", "Modéré", "Élevé", "Critique"]).fillna(0))
    grille["Part"] = grille["Effectif"] / len(scored)
    grille = grille[["Effectif", "Part", "Churn_observe", "Score_moyen"]]
    grille.columns = ["Effectif", "Part du portefeuille", "Taux de churn observé",
                      "Score moyen"]

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("**Grille de score et validation**")
        st.dataframe(grille.style.format(
            {"Effectif": "{:,.0f}", "Part du portefeuille": lambda v: fr(v),
             "Taux de churn observé": lambda v: fr(v), "Score moyen": "{:.0f}"})
            .apply(lambda r: [f"background-color:{FONDS_CLASSES.get(r.name,'')}"]
                   * len(r), axis=1), width="stretch")
    with c2:
        mono = grille["Taux de churn observé"].dropna().is_monotonic_increasing
        render_insight("Le taux de churn observé " +
                       ("<b>croît strictement</b> d'une classe à l'autre : le "
                        "score discrimine réellement le risque réel."
                        if mono else
                        "n'est pas parfaitement monotone : vérifiez la taille "
                        "des classes extrêmes."))
    st.download_button("Télécharger le portefeuille scoré (CSV)",
                       scored.to_csv(index=False).encode("utf-8"),
                       "portefeuille_score.csv", "text/csv")

# ============================================================================
# 11. PAGE PORTEFEUILLE CLIENTS
# ============================================================================
elif page == "Portefeuille clients":
    render_page_header("Portefeuille clients",
                       "Filtrer et explorer le portefeuille scoré")
    exiger_scores()
    scored = st.session_state["scores"]
    id_col = "id_client" if "id_client" in scored.columns else colonne_id(scored)

    c1, c2 = st.columns([1, 1])
    classes_sel = c1.multiselect("Classe de risque",
                                 ["Faible", "Modéré", "Élevé", "Critique"],
                                 default=["Élevé", "Critique"])
    seuil_score = c2.slider("Score maximal affiché", 300, 900, 900, 10)

    filtre = scored[scored["classe_risque"].isin(classes_sel)
                    & (scored["score"] <= seuil_score)]
    filtre = filtre.sort_values("score")
    st.caption(f"{len(filtre)} client(s) affiché(s) sur {len(scored)}.")

    colonnes_aff = [c for c in [id_col, "score", "classe_risque", "p_churn"]
                    if c and c in filtre.columns] + \
                   [c for c in st.session_state["modeles"]["feats"]
                    if c in filtre.columns]
    st.dataframe(
        filtre[colonnes_aff].style.format({"p_churn": "{:.1%}"})
        .apply(lambda r: [f"background-color:{FONDS_CLASSES.get(r['classe_risque'],'')}"
                          if col == "classe_risque" else "" for col in colonnes_aff],
               axis=1),
        width="stretch", height=420)
    st.download_button(f"Exporter cette sélection ({len(filtre)} clients)",
                       filtre.to_csv(index=False).encode("utf-8"),
                       "portefeuille_filtre.csv", "text/csv")

# ============================================================================
# 12. PAGE SIMULATEUR CLIENT
# ============================================================================
elif page == "Simulateur client":
    render_page_header("Simulateur client",
                       "Probabilité, score et explication individuelle du risque")
    exiger_modele()
    df = st.session_state["data"]
    M = st.session_state["modeles"]

    render_glass_card(
        "Renseignez le profil du client : les <b>mêmes transformations "
        "apprises sur le train</b> (imputation, encodage) sont appliquées à ce "
        "nouveau client — jamais recalculées sur lui.", titre="Profil client")

    valeurs = {}
    cols = st.columns(3)
    for i, f in enumerate(M["feats"]):
        c = cols[i % 3]
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            uniq = sorted(s.dropna().unique())
            if set(uniq) <= {0, 1}:
                valeurs[f] = 1 if c.selectbox(lab(f), ["Non", "Oui"],
                                              index=int(round(s.mean()))) == "Oui" else 0
            else:
                lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                entier = pd.api.types.is_integer_dtype(s)
                valeurs[f] = c.number_input(lab(f), lo, hi, med,
                                            step=1.0 if entier else 0.1)
        else:
            valeurs[f] = c.selectbox(lab(f), sorted(s.dropna().astype(str).unique()))
    if "solde_moyen" in valeurs and "ln_solde" in M["feats"]:
        valeurs["ln_solde"] = float(np.log1p(valeurs["solde_moyen"]))

    if st.button("Évaluer ce client", width="stretch"):
        x = pd.DataFrame([valeurs])[M["feats"]]
        x_enc = transformer_en_df(M["pre_fit"], x)         # transform seul
        if M["logit_sm"] is not None:
            p = float(np.asarray(M["logit_sm"].predict(
                sm.add_constant(x_enc, has_constant="add"))).ravel()[0])
        else:
            p = float(M["pipe_logit"].predict_proba(x)[0, 1])
        sc = float(np.clip(600 + 50 * np.log2((1 - p + 1e-9) / (p + 1e-9)), 300, 900))
        cl = ("Faible" if sc >= 700 else "Modéré" if sc >= 600
              else "Élevé" if sc >= 500 else "Critique")

        st.markdown("### Analyse")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sc,
                number={"font": {"color": BRUN, "size": 42}},
                title={"text": "Score de fidélité",
                       "font": {"color": MUT, "size": 13}},
                gauge={"axis": {"range": [300, 900],
                                "tickfont": {"size": 10, "color": MUT}},
                       "bar": {"color": BRUN, "thickness": 0.7},
                       "bgcolor": "rgba(0,0,0,0)",
                       "borderwidth": 0,
                       "steps": [{"range": [300, 500], "color": FONDS_CLASSES["Critique"]},
                                 {"range": [500, 600], "color": FONDS_CLASSES["Élevé"]},
                                 {"range": [600, 700], "color": FONDS_CLASSES["Modéré"]},
                                 {"range": [700, 900], "color": FONDS_CLASSES["Faible"]}]}))
            apply_plotly_theme(fig, height=300)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(
                f"<div class='glass-card solid' style='text-align:center;"
                f"padding:26px 22px;'>"
                f"<div style='font-size:0.79rem;color:{MUT};letter-spacing:.05em;"
                f"text-transform:uppercase;'>Probabilité de churn (6 mois)</div>"
                f"<div style='font-size:2.5rem;font-weight:700;color:{BRUN};"
                f"line-height:1.2;margin:8px 0 12px 0;'>{fr(p)}</div>"
                f"{render_badge(cl, 'Risque ')}</div>", unsafe_allow_html=True)
            actions = {"Faible": "Entretenir la relation via BBM et le parrainage.",
                       "Modéré": "Proposer un 2ᵉ produit, activer BBM, domicilier le salaire.",
                       "Élevé": "Contact proactif du conseiller sous 15 jours, geste commercial.",
                       "Critique": "Traitement prioritaire par le directeur sous 7 jours."}
            render_glass_card(actions[cl], titre="Action recommandée")

        if M["logit_sm"] is not None:
            st.markdown("---")
            st.markdown("### Explication individuelle du score")
            facteurs = contributions_client(x_enc.iloc[0], M["logit_sm"],
                                            M["moyennes_train"], M["cat_cols"])
            hausse = [f for f in facteurs if f[1] > 0]
            baisse = [f for f in facteurs if f[1] < 0]
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f"<div style='font-weight:650;color:{ROUGE};"
                            f"font-size:0.92rem;margin-bottom:6px;'>"
                            f"Facteurs qui augmentent le risque</div>",
                            unsafe_allow_html=True)
                if hausse:
                    for nom, val in hausse:
                        st.markdown(f"- {nom} (+{val:.2f} sur le logit) — "
                                    f"{suggestion_pour(nom)}")
                else:
                    st.caption("Aucun facteur dominant en ce sens pour ce client.")
            with cc2:
                st.markdown(f"<div style='font-weight:650;color:{VERT};"
                            f"font-size:0.92rem;margin-bottom:6px;'>"
                            f"Facteurs qui réduisent le risque</div>",
                            unsafe_allow_html=True)
                if baisse:
                    for nom, val in baisse:
                        st.markdown(f"- {nom} ({val:.2f} sur le logit)")
                else:
                    st.caption("Aucun facteur dominant en ce sens pour ce client.")
            render_insight(f"Un score de <b>{sc:.0f}</b> place ce client en "
                           f"risque <b>{cl.lower()}</b>, principalement "
                           f"expliqué par ses facteurs ci-dessus — décomposition "
                           f"de l'écart de son logit par rapport au client "
                           f"moyen de l'échantillon d'apprentissage.")

            if SHAP_OK:
                with st.expander("Explication avancée (SHAP — forêt aléatoire)"):
                    try:
                        x_enc_foret = transformer_en_df(
                            M["pipe_foret"].named_steps["prep"], x)
                        expl = shap.TreeExplainer(M["pipe_foret"].named_steps["clf"])
                        sv = expl.shap_values(x_enc_foret)
                        sv1 = sv[1][0] if isinstance(sv, list) else sv[0]
                        ordre = np.argsort(-np.abs(sv1))[:8]
                        fig = go.Figure(go.Bar(
                            x=sv1[ordre][::-1],
                            y=[lab_colonne_encodee(x_enc_foret.columns[i], M["cat_cols"])
                               for i in ordre][::-1],
                            orientation="h",
                            marker_color=[ROUGE if v > 0 else VERT
                                          for v in sv1[ordre][::-1]]))
                        apply_plotly_theme(fig, title="Contribution SHAP à la "
                                           "probabilité (forêt aléatoire)")
                        st.plotly_chart(fig, width="stretch")
                    except Exception as e:
                        st.caption(f"Explication SHAP indisponible pour ce cas ({e}).")
            else:
                st.caption("Installez le paquet `shap` pour activer l'explication "
                           "avancée de la forêt aléatoire.")

# ============================================================================
# 13. PAGE RECOMMANDATIONS
# ============================================================================
elif page == "Recommandations":
    render_page_header("Recommandations opérationnelles",
                       "Plan de rétention par classe et facteurs individuels par client")
    exiger_scores()
    scored = st.session_state["scores"]
    M = st.session_state["modeles"]
    cible = st.session_state["target"]
    id_col = "id_client" if "id_client" in scored.columns else colonne_id(scored)

    comptes = scored["classe_risque"].value_counts()
    render_kpi_row([(f"{int(comptes.get(c, 0)):,}".replace(",", " "),
                     f"Clients — risque {c.lower()}")
                    for c in ["Faible", "Modéré", "Élevé", "Critique"]])
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    plans = {
        "Critique": ("Traitement prioritaire — direction d'agence",
                     ["Appel du directeur sous 7 jours",
                      "Résolution accélérée des réclamations ouvertes",
                      "Offre de rétention sur mesure ; à défaut, entretien de sortie"]),
        "Élevé": ("Contact proactif — chargé de clientèle",
                  ["Entretien de découverte sous 15 jours", "Geste commercial ciblé",
                   "Suivi personnalisé inscrit au CRM"]),
        "Modéré": ("Campagnes d'équipement ciblées",
                   ["Proposer un 2ᵉ produit adapté", "Activation de Barid Bank Mobile",
                    "Domiciliation du salaire"]),
        "Faible": ("Entretenir la relation à moindre coût",
                   ["Communication digitale via BBM", "Information sur les nouveautés",
                    "Programme de parrainage"]),
    }
    for cl, (titre, acts) in plans.items():
        n = int(comptes.get(cl, 0))
        with st.expander(f"Risque {cl.lower()} — {n} client(s) · {titre}",
                         expanded=(cl in ("Critique", "Élevé"))):
            st.markdown(
                f"<div style='margin-bottom:10px'>{render_badge(cl, 'Niveau ')}"
                f"<span style='color:{MUT};font-size:0.84rem;margin-left:10px;'>"
                f"Action prioritaire : {titre.split('—')[0].strip().lower()}</span>"
                f"</div>", unsafe_allow_html=True)
            for a in acts:
                st.markdown(f"- {a}")

    st.markdown("---")
    st.markdown("### Classement individuel des clients à risque")
    a_contacter = scored[scored["classe_risque"].isin(["Élevé", "Critique"])] \
        .sort_values("score")

    if M["logit_sm"] is not None and len(a_contacter) > 0:
        n_aff = st.slider("Nombre de clients détaillés", 3,
                          min(30, len(a_contacter)), min(10, len(a_contacter)))
        lignes = []
        Xc = a_contacter[M["feats"]].head(n_aff)
        Xc_enc = transformer_en_df(M["pre_fit"], Xc)
        for idx in range(len(Xc)):
            facteurs = contributions_client(Xc_enc.iloc[idx], M["logit_sm"],
                                            M["moyennes_train"], M["cat_cols"])
            hausse = [f[0] for f in facteurs if f[1] > 0][:2]
            texte = ("; ".join(f"{h} → {suggestion_pour(h)}" for h in hausse)
                     if hausse else
                     "profil globalement défavorable, sans facteur isolé dominant")
            row = a_contacter.iloc[idx]
            lignes.append({
                "ID client": row[id_col] if id_col else row.name,
                "Score": int(row["score"]), "Classe": row["classe_risque"],
                "Probabilité": f"{row['p_churn']:.1%}",
                "Facteurs principaux & action": texte})
        st.dataframe(pd.DataFrame(lignes).style.apply(
            lambda r: [f"background-color:{FONDS_CLASSES.get(r['Classe'],'')}"
                       if c == "Classe" else "" for c in r.index], axis=1),
            width="stretch", height=400)
        render_insight("Contrairement à une recommandation uniforme par classe, "
                       "chaque ligne explique <i>pourquoi ce client précis</i> "
                       "est à risque — ce qui permet au conseiller d'adapter "
                       "concrètement son discours plutôt que d'appliquer un "
                       "script générique.")
    else:
        st.dataframe(a_contacter[[c for c in [id_col, "score", "classe_risque",
                                              "p_churn"] if c]], width="stretch")

    st.download_button(f"Exporter les {len(a_contacter)} clients à contacter",
                       a_contacter.to_csv(index=False).encode("utf-8"),
                       "clients_a_contacter.csv", "text/csv")

# ============================================================================
# 14. PAGE À PROPOS
# ============================================================================
else:
    render_page_header("À propos",
                       "Cadre académique, méthodologie et limites du modèle")
    c1, c2 = st.columns(2)
    with c1:
        render_glass_card(
            "PFA — « Analyse et prédiction de l'attrition des clients "
            "(Customer Churn) », stage effectué au sein d'Al Barid Bank, "
            "agence Drissia (Tanger).<br><br>"
            "<b>Auteur :</b> Walid BEN ABID — Filière Ingénierie Financière et "
            "Actuarielle, FST Errachidia (UMI).<br>"
            "<b>Encadrant pédagogique :</b> M. Sidi Ammi.<br>"
            "<b>Encadrante professionnelle :</b> Mme Nejwa Rachiq.",
            titre="Projet")
        render_glass_card(
            "1. Définition du churn (fenêtres 12 + 6 mois)<br>"
            "2. Split <b>train / validation / test</b> stratifié<br>"
            "3. <b>Pipelines scikit-learn</b> (imputation, encodage, échelle) "
            "ajustés sur le train uniquement<br>"
            "4. Logit (interprétable) vs arbre vs forêt, "
            "<b>validation croisée</b><br>"
            "5. Seuil choisi sur la <b>validation</b> (Youden ou métier), "
            "évalué sur le <b>test</b><br>"
            "6. Calibration (Brier, courbe de fiabilité)<br>"
            "7. Score 300-900, classes de risque, explications individuelles",
            titre="Chaîne méthodologique")
    with c2:
        render_warning(
            "• Sur données simulées, les coefficients reflètent le générateur "
            "choisi, pas une mesure empirique réelle.<br>"
            "• Les odds ratios sont des <b>associations</b>, pas des effets "
            "causaux (pas de test A/B réalisé).<br>"
            "• Aucune validation <b>hors échantillon temporel</b> : la dérive "
            "du modèle dans le temps n'est pas mesurée ici.<br>"
            "• Sur un CSV réel adapté, certaines variables peuvent être des "
            "<b>proxys approximatifs</b> ou présenter une <b>fuite de "
            "données</b> (ex. une réclamation quasi confondue avec le churn) — "
            "à vérifier systématiquement via le contrôle de corrélation du "
            "module Exploration.<br>"
            "• Le modèle doit être <b>ré-estimé périodiquement</b> et son "
            "AUC/KS suivi dans le temps avant tout usage en production.",
            label="Limites du modèle")
        render_glass_card(
            "Hosmer &amp; Lemeshow, <i>Applied Logistic Regression</i> (2013) · "
            "James et al., <i>An Introduction to Statistical Learning</i> "
            "(2021) · Thomas et al., <i>Credit Scoring and Its Applications</i> "
            "(2017) · documentation scikit-learn, statsmodels &amp; SHAP.",
            titre="Références")
