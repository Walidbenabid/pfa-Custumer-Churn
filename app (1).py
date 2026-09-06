# -*- coding: utf-8 -*-
# ============================================================================
#  CHURN ANALYTICS — AL BARID BANK (PFA) — v2 (édition professionnelle)
#  Analyse et prédiction de l'attrition des clients (Customer Churn)
#  ---------------------------------------------------------------------------
#  Chaîne complète : données (simulées ou CSV réel) → exploration →
#  modélisation (Pipeline scikit-learn, split train/validation/test,
#  validation croisée) → évaluation (ROC, calibration, seuil métier) →
#  scoring 300-900 → tableau de bord → portefeuille filtrable →
#  simulateur client avec explication individuelle → recommandations
#  personnalisées → limites du modèle.
#  Auteur : WALID [Nom] — FST Errachidia (IFA) · Encadrant : Pr. L. BEN HSSAIN
# ============================================================================

import base64
import io

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
# 0. CONFIGURATION GÉNÉRALE & CHARTE GRAPHIQUE AL BARID BANK
# ============================================================================
st.set_page_config(
    page_title="Churn Analytics — Al Barid Bank (PFA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

JAUNE = "#F5B800"
JAUNE_CLAIR = "#FFF6DC"
BRUN = "#4D3F37"
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

LOGO_B64 = "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"

st.markdown(f"""
<style>
    .stApp {{ background-color: #FFFFFF; }}
    h1, h2, h3 {{ color: {BRUN}; font-family: Georgia, 'Times New Roman', serif; }}
    [data-testid="stSidebar"] {{ background-color: {BRUN}; }}
    [data-testid="stSidebar"] * {{ color: #F2ECE6 !important; }}
    [data-testid="stSidebar"] .stRadio label:hover {{ color: {JAUNE} !important; }}
    .bandeau {{
        background: linear-gradient(90deg, {BRUN} 0%, {BRUN_FONCE} 100%);
        border-radius: 12px; padding: 20px 28px; margin-bottom: 18px;
        border-left: 8px solid {JAUNE};
    }}
    .bandeau h1 {{ color: #FFFFFF !important; margin: 0; font-size: 1.65rem; }}
    .bandeau p {{ color: #E8DFD8; margin: 6px 0 0 0; font-size: 0.95rem; }}
    .carte {{ background: {GRIS_CARTE}; border-radius: 12px; padding: 16px 20px;
              margin-bottom: 12px; }}
    .carte-jaune {{ background: {JAUNE_CLAIR}; border-radius: 12px;
                    padding: 16px 20px; border-left: 6px solid {JAUNE};
                    margin-bottom: 12px; }}
    .carte-rouge {{ background: #FBEAE8; border-radius: 12px;
                    padding: 16px 20px; border-left: 6px solid {ROUGE};
                    margin-bottom: 12px; }}
    .interpretation {{ background: {JAUNE_CLAIR}; border-left: 6px solid {JAUNE};
                       border-radius: 8px; padding: 14px 18px;
                       margin: 10px 0 18px 0; color: {TXT}; font-size: 0.95rem; }}
    .alerte {{ background: #FBEAE8; border-left: 6px solid {ROUGE};
              border-radius: 8px; padding: 14px 18px; margin: 10px 0 18px 0;
              color: {TXT}; font-size: 0.95rem; }}
    .kpi {{ background: {GRIS_CARTE}; border-radius: 12px; padding: 14px 10px;
            text-align: center; }}
    .kpi .val {{ font-size: 1.9rem; font-weight: 700; color: {BRUN};
                 font-family: Georgia, serif; }}
    .kpi .lab {{ font-size: 0.82rem; color: {MUT}; }}
    .badge {{ display: inline-block; padding: 4px 14px; border-radius: 14px;
              font-weight: 700; font-size: 0.9rem; }}
    div[data-testid="stMetricValue"] {{ color: {BRUN}; }}
    .stButton > button, .stDownloadButton > button {{
        background-color: {JAUNE}; color: {BRUN}; font-weight: 700;
        border: none; border-radius: 8px; }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {BRUN}; color: #FFFFFF; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    font=dict(family="Segoe UI, Arial", color=TXT),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
)


def fr(x, dec=1):
    return f"{x*100:.{dec}f}".replace(".", ",") + " %"


def num_fr(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def bandeau(titre, sous_titre=""):
    st.markdown(f"<div class='bandeau'><h1>{titre}</h1><p>{sous_titre}</p></div>",
               unsafe_allow_html=True)


def interpretation(html):
    st.markdown(f"<div class='interpretation'>🖊️ <b>Interprétation.</b> {html}</div>",
               unsafe_allow_html=True)


def alerte(html):
    st.markdown(f"<div class='alerte'>⚠️ {html}</div>", unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (val, lab_) in zip(cols, items):
        c.markdown(f"<div class='kpi'><div class='val'>{val}</div>"
                  f"<div class='lab'>{lab_}</div></div>", unsafe_allow_html=True)


# ============================================================================
# 1. MOTEUR DE DONNÉES : SIMULATION & IMPORT CSV (cible jamais imputée)
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
# 3. BARRE LATÉRALE : NAVIGATION + ÉTAT DE LA CHAÎNE
# ============================================================================
with st.sidebar:
    st.markdown(f"<div style='background:#FFFFFF;border-radius:10px;padding:10px;"
               f"text-align:center;margin-bottom:8px;'>"
               f"<img src='data:image/png;base64,{LOGO_B64}' width='150'></div>",
               unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-weight:700;font-size:1.02rem;'>"
               f"Churn Analytics</div><div style='text-align:center;font-size:0.8rem;'>"
               f"Attrition des clients — PFA</div>", unsafe_allow_html=True)
    st.markdown("---")
    PAGES = ["🏠 Accueil", "📊 Dashboard", "🗄️ Données", "🔍 Exploration",
             "⚙️ Modélisation", "✅ Évaluation", "⭐ Scoring",
             "📋 Portefeuille clients", "👤 Simulateur client",
             "💡 Recommandations", "ℹ️ À propos"]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    ok_d = "data" in st.session_state
    ok_m = "modeles" in st.session_state
    ok_s = "scores" in st.session_state
    st.markdown(
        f"{'🟢' if ok_d else '⚪'} Données&nbsp;: {'chargées' if ok_d else 'à charger'}<br>"
        f"{'🟢' if ok_m else '⚪'} Modèles&nbsp;: {'entraînés' if ok_m else 'à entraîner'}<br>"
        f"{'🟢' if ok_s else '⚪'} Scoring&nbsp;: {'calculé' if ok_s else 'à calculer'}",
        unsafe_allow_html=True)
    st.markdown("---")
    st.caption("PFA — Ingénierie Financière et Actuarielle · FST Errachidia\n\n"
               "Stage : Al Barid Bank, agence Drissia (Tanger)")


def exiger_donnees():
    if "data" not in st.session_state:
        st.info("⬅️ Commencez par charger ou simuler des données dans le "
                "module **🗄️ Données**.")
        st.stop()


def exiger_modele():
    exiger_donnees()
    if "modeles" not in st.session_state:
        st.info("⬅️ Entraînez d'abord les modèles dans le module "
                "**⚙️ Modélisation**.")
        st.stop()


def exiger_scores():
    exiger_modele()
    if "scores" not in st.session_state:
        st.info("⬅️ Calculez d'abord le scoring du portefeuille dans le "
                "module **⭐ Scoring**.")
        st.stop()


# ============================================================================
# 4. PAGE ACCUEIL
# ============================================================================
if page == "🏠 Accueil":
    bandeau("Analyse et prédiction de l'attrition des clients (Customer Churn)",
            "Al Barid Bank — Agence Drissia, Tanger · Projet de Fin d'Année")

    st.markdown("<div class='carte-jaune'><b>Problématique.</b> Comment identifier, "
               "de manière précoce et objective, les clients susceptibles de quitter "
               "Al Barid Bank, afin de mettre en place des actions préventives de "
               "rétention ciblées et efficaces&nbsp;?</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    etapes = [
        ("🗄️", "1. Données", "Simulez un portefeuille ou importez votre CSV réel."),
        ("⚙️", "2. Modèle", "Pipeline scikit-learn, split train/validation/test, CV."),
        ("✅", "3. Évaluation", "ROC, calibration, seuil métier vs seuil statistique."),
        ("⭐", "4. Scoring", "Score de fidélité 300-900 et 4 classes de risque."),
        ("💡", "5. Actions", "Recommandations personnalisées par client."),
    ]
    for c, (ic, t, d) in zip(cols, etapes):
        c.markdown(f"<div class='carte' style='min-height:165px'>"
                  f"<div style='font-size:1.5rem'>{ic}</div><b>{t}</b><br>"
                  f"<span style='color:{MUT};font-size:0.84rem'>{d}</span></div>",
                  unsafe_allow_html=True)

    st.subheader("Repères issus du mémoire (base de référence : 4 000 clients)")
    kpi_row([("13,9 %", "taux d'attrition observé"),
             ("0,805", "AUC — régression logistique"),
             ("85,5 %", "churners détectés (rappel)"),
             ("× 2,04", "cote de churn par réclamation")])
    st.markdown("")
    interpretation("Cette édition applique une discipline de validation "
                  "<b>train / validation / test</b> stricte, des "
                  "<b>Pipelines scikit-learn</b> sans fuite de données, une "
                  "<b>vérification de calibration</b> des probabilités, et "
                  "des explications <b>individuelles</b> par client — au-delà "
                  "de la simple classe de risque.")

# ============================================================================
# 5. PAGE DASHBOARD
# ============================================================================
elif page == "📊 Dashboard":
    bandeau("📊 Tableau de bord", "Vue d'ensemble du portefeuille scoré")
    exiger_scores()
    scored = st.session_state["scores"]
    cible = st.session_state["target"]
    comptes = scored["classe_risque"].value_counts()
    n = len(scored)

    kpi_row([
        (f"{n:,}".replace(",", " "), "clients au portefeuille"),
        (fr(scored[cible].mean()), "taux d'attrition global"),
        (f"{int(comptes.get('Élevé', 0)):,}".replace(",", " "),
         f"risque élevé ({fr(comptes.get('Élevé', 0)/n)})"),
        (f"{int(comptes.get('Critique', 0)):,}".replace(",", " "),
         f"risque critique ({fr(comptes.get('Critique', 0)/n)})"),
    ])
    st.markdown("")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        ordre = ["Faible", "Modéré", "Élevé", "Critique"]
        vals = [int(comptes.get(c, 0)) for c in ordre]
        fig = go.Figure(go.Pie(labels=ordre, values=vals, hole=0.5,
                               marker=dict(colors=[COULEURS_CLASSES[c] for c in ordre]),
                               textinfo="label+percent"))
        fig.update_layout(title="Portefeuille par classe de risque",
                          showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        g = scored.groupby("classe_risque", observed=True)[cible].mean().reindex(
            ["Faible", "Modéré", "Élevé", "Critique"])
        fig = go.Figure(go.Bar(x=g.index, y=g.values * 100,
                               marker_color=[COULEURS_CLASSES[c] for c in g.index],
                               text=[f"{v*100:.1f}%" for v in g.values],
                               textposition="outside"))
        fig.update_layout(title="Taux de churn observé par classe",
                          yaxis_title="Taux de churn (%)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    interpretation("Le tableau de bord se met à jour automatiquement à chaque "
                  "nouveau calcul du scoring — c'est la vue destinée à un "
                  "comité de pilotage ou à un directeur d'agence, sans détail "
                  "technique.")

# ============================================================================
# 6. PAGE DONNÉES
# ============================================================================
elif page == "🗄️ Données":
    bandeau("🗄️ Données", "Portefeuille simulé paramétrable ou import d'un CSV réel")
    mode = st.radio("Source des données", ["Simulation (paramétrable)",
                    "Import d'un fichier CSV réel"], horizontal=True)

    if mode == "Simulation (paramétrable)":
        c1, c2, c3 = st.columns([2, 1, 1])
        n = c1.slider("Nombre de clients simulés", 500, 20000, 4000, step=500)
        graine = c2.number_input("Graine aléatoire", 0, 9999, 42)
        c3.markdown("<br>", unsafe_allow_html=True)
        if c3.button("🎲 Générer le portefeuille", width="stretch"):
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
        st.markdown("<div class='carte'>Le fichier doit contenir une colonne "
                   "cible binaire (0/1) et des variables explicatives. "
                   "<b>Les lignes dont la cible est manquante ou non binaire "
                   "sont supprimées</b>, jamais imputées à 0 — imputer une "
                   "cible inconnue biaiserait l'apprentissage.</div>",
                   unsafe_allow_html=True)
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
                if st.button("✅ Valider ce jeu de données"):
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
        st.subheader("Aperçu du jeu de données actif")
        st.caption(st.session_state["source"])
        kpi_row([(f"{len(df):,}".replace(",", " "), "clients"),
                 (str(len(st.session_state['features'])), "variables explicatives"),
                 (f"{int(df[cible].sum()):,}".replace(",", " "), "churners"),
                 (fr(df[cible].mean()), "taux d'attrition")])
        st.markdown("")
        st.dataframe(df.head(10), width="stretch")
        with st.expander("📖 Dictionnaire des variables (base simulée)"):
            st.table(pd.DataFrame([(k, v) for k, v in VARS_FR.items()],
                                  columns=["Champ", "Description"]))
        st.download_button("⬇️ Télécharger le jeu de données (CSV)",
                          df.to_csv(index=False).encode("utf-8"),
                          "portefeuille_churn.csv", "text/csv")

# ============================================================================
# 7. PAGE EXPLORATION
# ============================================================================
elif page == "🔍 Exploration":
    bandeau("🔍 Analyse exploratoire",
            "Taux d'attrition, profils à risque et liaisons bivariées")
    exiger_donnees()
    df = st.session_state["data"]
    cible = st.session_state["target"]
    feats = st.session_state["features"]
    taux = df[cible].mean()

    kpi_row([(f"{len(df):,}".replace(",", " "), "clients"),
             (f"{int(df[cible].sum()):,}".replace(",", " "), "churners"),
             (fr(taux), "taux d'attrition"),
             (f"{int((1-taux)*len(df)):,}".replace(",", " "), "clients actifs")])
    st.markdown("")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(go.Pie(labels=["Clients actifs", "Churners"],
                               values=[1 - taux, taux], hole=0.55,
                               marker=dict(colors=[JAUNE, BRUN]),
                               textinfo="label+percent"))
        fig.update_layout(title="Répartition du portefeuille", showlegend=False,
                          **PLOTLY_LAYOUT)
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
                               marker_line=dict(color=BRUN, width=1.2),
                               text=[f"{v*100:.1f}%" for v in g.values],
                               textposition="outside"))
        fig.update_layout(title=f"Taux de churn selon {lab(var).lower()}",
                          yaxis_title="Taux de churn (%)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        ecart = g.max() - g.min()
        interpretation(f"Le taux de churn varie de <b>{fr(g.min())}</b> à "
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
                          marker_color=JAUNE, opacity=0.75, nbinsx=35)
        fig.add_histogram(x=df.loc[df[cible] == 1, var2], name="Churners",
                          marker_color=ROUGE, opacity=0.75, nbinsx=35)
        fig.update_layout(barmode="overlay", title=f"Distribution — {lab(var2)}",
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c4:
        corr = df[num_disp + [cible]].corr()[cible].drop(cible).sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=[lab(i) for i in corr.index],
                               orientation="h",
                               marker_color=[VERT if v < 0 else ROUGE
                                            for v in corr.values]))
        fig.update_layout(title="Corrélation de chaque variable avec le churn",
                          xaxis_title="Corrélation", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        fuite = corr[corr.abs() > 0.9]
        if len(fuite) > 0:
            alerte(f"<b>Risque de fuite de données.</b> "
                  f"{', '.join(lab(i) for i in fuite.index)} présente une "
                  f"corrélation supérieure à 0,9 avec la cible — c'est "
                  f"anormalement élevé pour un facteur prédictif authentique. "
                  f"Vérifiez qu'elle n'est pas mesurée <i>après</i> le départ "
                  f"du client avant de l'inclure dans le modèle (module "
                  f"Modélisation).")
    interpretation("Les barres <span style='color:#1E8449'><b>vertes</b></span> "
                  "sont des facteurs de <b>rétention</b>, les "
                  "<span style='color:#B03A2E'><b>rouges</b></span> des "
                  "facteurs de <b>risque</b>. Ce diagnostic descriptif reste "
                  "valable même pour une variable exclue du modèle prédictif.")

# ============================================================================
# 8. PAGE MODÉLISATION
# ============================================================================
elif page == "⚙️ Modélisation":
    bandeau("⚙️ Modélisation",
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

    if st.button("🚀 Entraîner les modèles", width="stretch"):
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
            note = ("⚠️ Séparation quasi parfaite détectée pour statsmodels : "
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
        st.subheader("Validation croisée (5 plis, échantillon d'apprentissage)")
        cv_df = pd.DataFrame({nom: {"AUC moyen": m, "Écart-type": s}
                              for nom, (m, s) in M["cv_res"].items()}).T
        st.dataframe(cv_df.style.format("{:.3f}"), width="stretch")
        ecarts_cv = cv_df["Écart-type"].max()
        interpretation(f"L'écart-type maximal entre plis est de "
                      f"<b>{num_fr(ecarts_cv)}</b> — "
                      + ("un résultat stable : la performance ne dépend pas "
                         "d'un découpage particulier du train/test."
                         if ecarts_cv < 0.03 else
                         "une variabilité notable d'un pli à l'autre : à "
                         "surveiller si le portefeuille est de petite taille."))

        if M["logit_sm"] is not None:
            st.markdown("---")
            st.subheader("Coefficients de la régression logistique (interprétable)")
            res = M["logit_sm"]
            tab = pd.DataFrame({"Coefficient": res.params, "Écart-type": res.bse,
                               "Odds ratio": np.exp(res.params), "p-value": res.pvalues})
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
                x=orr.values, y=[lab_colonne_encodee(i, M["cat_cols"]) for i in orr.index],
                orientation="h",
                marker_color=[VERT if v < 1 else ROUGE for v in orr.values],
                text=[f"{v:.2f}" for v in orr.values], textposition="outside"))
            fig.add_vline(x=1, line_color=TXT)
            fig.update_layout(title="Odds ratios — facteurs de risque (>1) et "
                              "de rétention (<1)", xaxis_title="Odds ratio",
                              **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            interpretation(f"Premier facteur de risque : "
                          f"<b>{lab_colonne_encodee(orr.idxmax(), M['cat_cols']).lower()}</b> "
                          f"(cote × {orr.max():.2f}). Premier facteur de rétention : "
                          f"<b>{lab_colonne_encodee(orr.idxmin(), M['cat_cols']).lower()}</b> "
                          f"(cote ÷ {1/orr.min():.1f}).")
        else:
            st.info("Coefficients interprétables indisponibles pour cet "
                   "entraînement (repli scikit-learn) — les odds ratios "
                   "nécessitent une convergence statsmodels.")

        st.markdown("---")
        st.subheader("Importance des variables — forêt aléatoire")
        noms_enc = [c.split("__", 1)[1] if "__" in c else c
                   for c in M["pipe_foret"].named_steps["prep"].get_feature_names_out()]
        imp = pd.Series(M["pipe_foret"].named_steps["clf"].feature_importances_,
                        index=noms_enc).sort_values()
        fig = go.Figure(go.Bar(
            x=imp.values, y=[lab_colonne_encodee(i, M["cat_cols"]) for i in imp.index],
            orientation="h", marker_color=JAUNE,
            marker_line=dict(color=BRUN, width=1)))
        fig.update_layout(title="Importance (réduction d'impureté, forêt aléatoire)",
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        interpretation("Cette importance complète les odds ratios : elle capte "
                      "aussi les effets non linéaires et les interactions "
                      "qu'un modèle linéaire ne peut pas représenter — un "
                      "classement très différent des odds ratios inviterait à "
                      "explorer des termes d'interaction dans le logit.")

# ============================================================================
# 9. PAGE ÉVALUATION
# ============================================================================
elif page == "✅ Évaluation":
    bandeau("✅ Évaluation des modèles",
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
    fig.update_layout(title="Courbes ROC — échantillon de TEST (jamais utilisé "
                      "pour régler le seuil)", xaxis_title="1 − spécificité",
                      yaxis_title="Sensibilité", legend=dict(x=0.42, y=0.08),
                      **PLOTLY_LAYOUT)

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
        interpretation(f"<b>{meilleur['Modèle']}</b> obtient le meilleur AUC "
                      f"({num_fr(meilleur['AUC'])}) tout en offrant une "
                      f"interprétabilité <b>{meilleur['Interprétabilité'].lower()}</b> "
                      f"— c'est la combinaison performance + auditabilité qui "
                      f"justifie son choix comme modèle de production, plutôt "
                      f"que l'AUC seul.")

    # ---------------------------------------------- seuil (validation only) --
    st.markdown("---")
    st.subheader("Choix du seuil — régression logistique")
    p_val_l, p_te_l = M["p_val"]["Régression logistique"], M["p_test"]["Régression logistique"]
    fpr_v, tpr_v, thr_v = roc_curve(y_val, p_val_l)
    s_youden = float(thr_v[np.argmax(tpr_v - fpr_v)])

    onglet_stat, onglet_metier = st.tabs(["📐 Seuil statistique (Youden)",
                                          "💰 Seuil métier (coûts asymétriques)"])
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
        fig.add_scatter(x=grille, y=couts, mode="lines", line=dict(color=BRUN, width=2.2))
        fig.add_vline(x=s_youden, line_dash="dash", line_color=MUT,
                     annotation_text="Youden")
        fig.add_vline(x=s_metier, line_dash="dash", line_color=ROUGE,
                     annotation_text="Coût minimal")
        fig.update_layout(title="Coût total sur la validation selon le seuil",
                          xaxis_title="Seuil", yaxis_title="Coût total",
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.metric("Seuil optimal métier (calculé sur la VALIDATION)", num_fr(s_metier))
        interpretation(f"Avec un ratio de coûts de {cout_fn/cout_fp:.0f}:1, le "
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
        fig.update_layout(title=f"Matrice de confusion — TEST (seuil = "
                          f"{seuil_choisi:.3f})", yaxis_autorange="reversed",
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = go.Figure(go.Heatmap(
            z=cm_norm, x=["Prédit actif", "Prédit churner"],
            y=["Réel actif", "Réel churner"],
            colorscale=[[0, "#FFF6DC"], [1, BRUN]], showscale=False,
            text=[[f"{cm_norm[0,0]*100:.1f}%", f"{cm_norm[0,1]*100:.1f}%"],
                  [f"{cm_norm[1,0]*100:.1f}%", f"{cm_norm[1,1]*100:.1f}%"]],
            texttemplate="%{text}", textfont=dict(size=14)))
        fig.update_layout(title="Matrice de confusion normalisée (% par ligne)",
                          yaxis_autorange="reversed", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    interpretation(f"Sur l'échantillon de <b>test</b>, totalement indépendant "
                  f"du choix de seuil, le modèle détecte <b>{vp} des "
                  f"{vp+fn} churners</b> (rappel {fr(vp/(vp+fn))}), au prix de "
                  f"{fp} clients fidèles sollicités à tort.")

    # -------------------------------------------------------- calibration --
    st.markdown("---")
    st.subheader("Calibration des probabilités")
    frac_pos, moy_pred = calibration_curve(y_te, p_te_l, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_te, p_te_l)
    c5, c6 = st.columns([1.3, 1])
    with c5:
        fig = go.Figure()
        fig.add_scatter(x=moy_pred, y=frac_pos, mode="lines+markers",
                        name="Modèle", line=dict(color=BRUN, width=2.4))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                        line=dict(color="#BBBBBB", dash="dash"))
        fig.update_layout(title="Courbe de calibration (test, 10 quantiles)",
                          xaxis_title="Probabilité moyenne prédite",
                          yaxis_title="Fréquence observée de churn", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c6:
        st.metric("Score de Brier (plus bas = meilleur)", num_fr(brier, 4))
        interpretation("La transformation en score de fidélité (300-900) "
                      "utilise la probabilité <i>elle-même</i>, pas seulement "
                      "son rang : une calibration correcte est donc "
                      "indispensable pour que le score soit interprétable en "
                      "valeur absolue, pas seulement pour classer les "
                      "clients entre eux.")

# ============================================================================
# 10. PAGE SCORING
# ============================================================================
elif page == "⭐ Scoring":
    bandeau("⭐ Scoring du portefeuille",
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

    st.markdown("<div class='carte-jaune'><b>Formule.</b> Score = 600 + 50 × "
               "log₂((1 − p̂) / p̂) — 600 points = cote 1 contre 1 ; chaque "
               "tranche de <b>50 points double la cote de fidélité</b>.</div>",
               unsafe_allow_html=True)

    fig = go.Figure()
    for a, b, c in [(300, 500, FONDS_CLASSES["Critique"]), (500, 600, FONDS_CLASSES["Élevé"]),
                    (600, 700, FONDS_CLASSES["Modéré"]), (700, 900, FONDS_CLASSES["Faible"])]:
        fig.add_vrect(x0=a, x1=b, fillcolor=c, opacity=0.55, line_width=0)
    fig.add_histogram(x=scored.loc[scored[cible] == 0, "score"], name="Clients actifs",
                      marker_color=BLEU, opacity=0.8, nbinsx=45)
    fig.add_histogram(x=scored.loc[scored[cible] == 1, "score"], name="Churners observés",
                      marker_color=ROUGE, opacity=0.8, nbinsx=45)
    for lim in (500, 600, 700):
        fig.add_vline(x=lim, line_dash="dash", line_color=MUT)
    fig.update_layout(barmode="overlay", title="Distribution des scores et zones de risque",
                      xaxis_title="Score de fidélité (300 – 900)",
                      yaxis_title="Nombre de clients", **PLOTLY_LAYOUT)
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
        interpretation("Le taux de churn observé " +
                      ("<b>croît strictement</b> d'une classe à l'autre : le "
                       "score discrimine réellement le risque réel."
                       if mono else
                       "n'est pas parfaitement monotone : vérifiez la taille "
                       "des classes extrêmes."))
    st.download_button("⬇️ Télécharger le portefeuille scoré (CSV)",
                      scored.to_csv(index=False).encode("utf-8"),
                      "portefeuille_score.csv", "text/csv")

# ============================================================================
# 11. PAGE PORTEFEUILLE CLIENTS
# ============================================================================
elif page == "📋 Portefeuille clients":
    bandeau("📋 Portefeuille clients", "Filtrer et explorer le portefeuille scoré")
    exiger_scores()
    scored = st.session_state["scores"]
    id_col = "id_client" if "id_client" in scored.columns else colonne_id(scored)

    c1, c2 = st.columns([1, 1])
    classes_sel = c1.multiselect("Classe de risque", ["Faible", "Modéré", "Élevé", "Critique"],
                                 default=["Élevé", "Critique"])
    seuil_score = c2.slider("Score maximal affiché", 300, 900, 900, 10)

    filtre = scored[scored["classe_risque"].isin(classes_sel) & (scored["score"] <= seuil_score)]
    filtre = filtre.sort_values("score")
    st.caption(f"{len(filtre)} client(s) affiché(s) sur {len(scored)}.")

    colonnes_aff = [c for c in [id_col, "score", "classe_risque", "p_churn"]
                    if c and c in filtre.columns] + \
                   [c for c in st.session_state["modeles"]["feats"] if c in filtre.columns]
    st.dataframe(
        filtre[colonnes_aff].style.format({"p_churn": "{:.1%}"})
        .apply(lambda r: [f"background-color:{FONDS_CLASSES.get(r['classe_risque'],'')}"
                          if col == "classe_risque" else "" for col in colonnes_aff], axis=1),
        width="stretch", height=420)
    st.download_button(f"⬇️ Exporter cette sélection ({len(filtre)} clients)",
                      filtre.to_csv(index=False).encode("utf-8"),
                      "portefeuille_filtre.csv", "text/csv")

# ============================================================================
# 12. PAGE SIMULATEUR CLIENT
# ============================================================================
elif page == "👤 Simulateur client":
    bandeau("👤 Simulateur client",
            "Probabilité, score et explication individuelle du risque")
    exiger_modele()
    df = st.session_state["data"]
    M = st.session_state["modeles"]

    st.markdown("<div class='carte'>Renseignez le profil du client : les "
               "<b>mêmes transformations apprises sur le train</b> (imputation, "
               "encodage) sont appliquées à ce nouveau client — jamais "
               "recalculées sur lui.</div>", unsafe_allow_html=True)

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

    if st.button("🎯 Évaluer ce client", width="stretch"):
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

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sc, number={"font": {"color": BRUN}},
                title={"text": "Score de fidélité", "font": {"color": BRUN}},
                gauge={"axis": {"range": [300, 900]}, "bar": {"color": BRUN},
                      "steps": [{"range": [300, 500], "color": FONDS_CLASSES["Critique"]},
                                {"range": [500, 600], "color": FONDS_CLASSES["Élevé"]},
                                {"range": [600, 700], "color": FONDS_CLASSES["Modéré"]},
                                {"range": [700, 900], "color": FONDS_CLASSES["Faible"]}]}))
            fig.update_layout(height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(f"<div class='carte' style='text-align:center'>"
                       f"<div style='color:{MUT}'>Probabilité de churn (6 mois)</div>"
                       f"<div style='font-size:2.3rem;font-weight:700;color:{BRUN}'>"
                       f"{fr(p)}</div><span class='badge' style='background:"
                       f"{FONDS_CLASSES[cl]};color:{COULEURS_CLASSES[cl]}'>"
                       f"Risque {cl.lower()}</span></div>", unsafe_allow_html=True)
            actions = {"Faible": "Entretenir la relation via BBM et le parrainage.",
                      "Modéré": "Proposer un 2ᵉ produit, activer BBM, domicilier le salaire.",
                      "Élevé": "Contact proactif du conseiller sous 15 jours, geste commercial.",
                      "Critique": "Traitement prioritaire par le directeur sous 7 jours."}
            st.markdown(f"<div class='carte-jaune'><b>Action recommandée.</b> "
                       f"{actions[cl]}</div>", unsafe_allow_html=True)

        if M["logit_sm"] is not None:
            st.markdown("---")
            st.subheader("Explication individuelle du score")
            facteurs = contributions_client(x_enc.iloc[0], M["logit_sm"],
                                           M["moyennes_train"], M["cat_cols"])
            hausse = [f for f in facteurs if f[1] > 0]
            baisse = [f for f in facteurs if f[1] < 0]
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**🔺 Facteurs qui augmentent le risque**")
                if hausse:
                    for nom, val in hausse:
                        st.markdown(f"- {nom} (+{val:.2f} sur le logit) — "
                                   f"{suggestion_pour(nom)}")
                else:
                    st.caption("Aucun facteur dominant en ce sens pour ce client.")
            with cc2:
                st.markdown("**🔻 Facteurs qui réduisent le risque**")
                if baisse:
                    for nom, val in baisse:
                        st.markdown(f"- {nom} ({val:.2f} sur le logit)")
                else:
                    st.caption("Aucun facteur dominant en ce sens pour ce client.")
            interpretation(f"Un score de <b>{sc:.0f}</b> place ce client en "
                          f"risque <b>{cl.lower()}</b>, principalement "
                          f"expliqué par ses facteurs ci-dessus — décomposition "
                          f"de l'écart de son logit par rapport au client "
                          f"moyen de l'échantillon d'apprentissage.")

            if SHAP_OK:
                with st.expander("🔬 Explication avancée (SHAP — forêt aléatoire)"):
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
                            marker_color=[ROUGE if v > 0 else VERT for v in sv1[ordre][::-1]]))
                        fig.update_layout(title="Contribution SHAP à la probabilité "
                                          "(forêt aléatoire)", **PLOTLY_LAYOUT)
                        st.plotly_chart(fig, width="stretch")
                    except Exception as e:
                        st.caption(f"Explication SHAP indisponible pour ce cas ({e}).")
            else:
                st.caption("Installez le paquet `shap` pour activer l'explication "
                          "avancée de la forêt aléatoire.")

# ============================================================================
# 13. PAGE RECOMMANDATIONS
# ============================================================================
elif page == "💡 Recommandations":
    bandeau("💡 Recommandations opérationnelles",
            "Plan de rétention par classe et facteurs individuels par client")
    exiger_scores()
    scored = st.session_state["scores"]
    M = st.session_state["modeles"]
    cible = st.session_state["target"]
    id_col = "id_client" if "id_client" in scored.columns else colonne_id(scored)

    comptes = scored["classe_risque"].value_counts()
    kpi_row([(f"{int(comptes.get(c, 0)):,}".replace(",", " "),
              f"clients — risque {c.lower()}")
             for c in ["Faible", "Modéré", "Élevé", "Critique"]])
    st.markdown("")

    plans = {
        "Critique": ("🔴", "Traitement prioritaire — direction d'agence",
                    ["Appel du directeur sous 7 jours",
                     "Résolution accélérée des réclamations ouvertes",
                     "Offre de rétention sur mesure ; à défaut, entretien de sortie"]),
        "Élevé": ("🟠", "Contact proactif — chargé de clientèle",
                 ["Entretien de découverte sous 15 jours", "Geste commercial ciblé",
                  "Suivi personnalisé inscrit au CRM"]),
        "Modéré": ("🟡", "Campagnes d'équipement ciblées",
                  ["Proposer un 2ᵉ produit adapté", "Activation de Barid Bank Mobile",
                   "Domiciliation du salaire"]),
        "Faible": ("🟢", "Entretenir la relation à moindre coût",
                  ["Communication digitale via BBM", "Information sur les nouveautés",
                   "Programme de parrainage"]),
    }
    for cl, (ic, titre, acts) in plans.items():
        n = int(comptes.get(cl, 0))
        with st.expander(f"{ic} Risque {cl.lower()} — {n} client(s) · {titre}",
                        expanded=(cl in ("Critique", "Élevé"))):
            for a in acts:
                st.markdown(f"- {a}")

    st.markdown("---")
    st.subheader("Classement individuel des clients à risque")
    a_contacter = scored[scored["classe_risque"].isin(["Élevé", "Critique"])].sort_values("score")

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
                    if hausse else "profil globalement défavorable, sans facteur isolé dominant")
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
        interpretation("Contrairement à une recommandation uniforme par classe, "
                      "chaque ligne explique <i>pourquoi ce client précis</i> "
                      "est à risque — ce qui permet au conseiller d'adapter "
                      "concrètement son discours plutôt que d'appliquer un "
                      "script générique.")
    else:
        st.dataframe(a_contacter[[c for c in [id_col, "score", "classe_risque",
                                              "p_churn"] if c]], width="stretch")

    st.download_button(f"⬇️ Exporter les {len(a_contacter)} clients à contacter",
                      a_contacter.to_csv(index=False).encode("utf-8"),
                      "clients_a_contacter.csv", "text/csv")

# ============================================================================
# 14. PAGE À PROPOS
# ============================================================================
else:
    bandeau("ℹ️ À propos", "Cadre académique, méthodologie et limites du modèle")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='carte'><b>Projet.</b> PFA — « Analyse et "
                   "prédiction de l'attrition des clients (Customer Churn) », "
                   "stage effectué au sein d'Al Barid Bank, agence Drissia "
                   "(Tanger).<br><br><b>Auteur :</b> WALID [Nom] — Filière "
                   "Ingénierie Financière et Actuarielle, FST Errachidia "
                   "(UMI).<br><b>Encadrant pédagogique :</b> Pr. Lhoucine "
                   "BEN HSSAIN.<br><b>Encadrant professionnel :</b> M. "
                   "[Encadrant agence].</div>", unsafe_allow_html=True)
        st.markdown("<div class='carte'><b>Chaîne méthodologique (v2).</b><br>"
                   "1. Définition du churn (fenêtres 12 + 6 mois)<br>"
                   "2. Split <b>train / validation / test</b> stratifié<br>"
                   "3. <b>Pipelines scikit-learn</b> (imputation, encodage, "
                   "échelle) ajustés sur le train uniquement<br>"
                   "4. Logit (interprétable) vs arbre vs forêt, "
                   "<b>validation croisée</b><br>"
                   "5. Seuil choisi sur la <b>validation</b> (Youden ou métier), "
                   "évalué sur le <b>test</b><br>"
                   "6. Calibration (Brier, courbe de fiabilité)<br>"
                   "7. Score 300-900, classes de risque, explications "
                   "individuelles</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='carte-rouge'><b>⚠️ Limites du modèle.</b><br>"
                   "• Sur données simulées, les coefficients reflètent le "
                   "générateur choisi, pas une mesure empirique réelle.<br>"
                   "• Les odds ratios sont des <b>associations</b>, pas des "
                   "effets causaux (pas de test A/B réalisé).<br>"
                   "• Aucune validation <b>hors échantillon temporel</b> : la "
                   "dérive du modèle dans le temps n'est pas mesurée ici.<br>"
                   "• Sur un CSV réel adapté, certaines variables peuvent être "
                   "des <b>proxys approximatifs</b> ou présenter une "
                   "<b>fuite de données</b> (ex. une réclamation quasi "
                   "confondue avec le churn) — à vérifier systématiquement "
                   "via le contrôle de corrélation du module Exploration.<br>"
                   "• Le modèle doit être <b>ré-estimé périodiquement</b> et "
                   "son AUC/KS suivi dans le temps avant tout usage en "
                   "production.</div>", unsafe_allow_html=True)
        st.markdown("<div class='carte'><b>Références.</b> Hosmer &amp; "
                   "Lemeshow, <i>Applied Logistic Regression</i> (2013) · "
                   "James et al., <i>An Introduction to Statistical "
                   "Learning</i> (2021) · Thomas et al., <i>Credit Scoring "
                   "and Its Applications</i> (2017) · documentation "
                   "scikit-learn, statsmodels &amp; SHAP.</div>",
                   unsafe_allow_html=True)
