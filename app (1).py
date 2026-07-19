# -*- coding: utf-8 -*-
# ============================================================================
#  CHURN ANALYTICS — AL BARID BANK (PFA)
#  Analyse et prédiction de l'attrition des clients (Customer Churn)
#  ---------------------------------------------------------------------------
#  Application Streamlit automatisant la chaîne complète du rapport :
#  données (simulées ou CSV réel) → exploration → modélisation → évaluation
#  → scoring 300-900 → simulateur client → recommandations.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    [data-testid="stSidebar"] {{
        background-color: {BRUN};
    }}
    [data-testid="stSidebar"] * {{ color: #F2ECE6 !important; }}
    [data-testid="stSidebar"] .stRadio label:hover {{ color: {JAUNE} !important; }}
    .bandeau {{
        background: linear-gradient(90deg, {BRUN} 0%, {BRUN_FONCE} 100%);
        border-radius: 12px; padding: 20px 28px; margin-bottom: 18px;
        border-left: 8px solid {JAUNE};
    }}
    .bandeau h1 {{ color: #FFFFFF !important; margin: 0; font-size: 1.65rem; }}
    .bandeau p {{ color: #E8DFD8; margin: 6px 0 0 0; font-size: 0.95rem; }}
    .carte {{
        background: {GRIS_CARTE}; border-radius: 12px; padding: 16px 20px;
        margin-bottom: 12px;
    }}
    .carte-jaune {{
        background: {JAUNE_CLAIR}; border-radius: 12px; padding: 16px 20px;
        border-left: 6px solid {JAUNE}; margin-bottom: 12px;
    }}
    .interpretation {{
        background: {JAUNE_CLAIR}; border-left: 6px solid {JAUNE};
        border-radius: 8px; padding: 14px 18px; margin: 10px 0 18px 0;
        color: {TXT}; font-size: 0.95rem;
    }}
    .kpi {{
        background: {GRIS_CARTE}; border-radius: 12px; padding: 14px 10px;
        text-align: center;
    }}
    .kpi .val {{ font-size: 1.9rem; font-weight: 700; color: {BRUN};
                 font-family: Georgia, serif; }}
    .kpi .lab {{ font-size: 0.82rem; color: {MUT}; }}
    .badge {{
        display: inline-block; padding: 4px 14px; border-radius: 14px;
        font-weight: 700; font-size: 0.9rem;
    }}
    div[data-testid="stMetricValue"] {{ color: {BRUN}; }}
    .stButton > button, .stDownloadButton > button {{
        background-color: {JAUNE}; color: {BRUN}; font-weight: 700;
        border: none; border-radius: 8px;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {BRUN}; color: #FFFFFF;
    }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    font=dict(family="Segoe UI, Arial", color=TXT),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
)


def fr(x, dec=1):
    """Format pourcentage à la française."""
    return f"{x*100:.{dec}f}".replace(".", ",") + " %"


def num_fr(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def bandeau(titre, sous_titre=""):
    st.markdown(
        f"<div class='bandeau'><h1>{titre}</h1><p>{sous_titre}</p></div>",
        unsafe_allow_html=True)


def interpretation(html):
    st.markdown(
        f"<div class='interpretation'>🖊️ <b>Interprétation.</b> {html}</div>",
        unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (val, lab) in zip(cols, items):
        c.markdown(
            f"<div class='kpi'><div class='val'>{val}</div>"
            f"<div class='lab'>{lab}</div></div>", unsafe_allow_html=True)


# ============================================================================
# 1. MOTEUR DE DONNÉES : SIMULATION (même DGP que le rapport) & IMPORT CSV
# ============================================================================
VARS_FR = {
    "age": "Âge",
    "anciennete": "Ancienneté (mois)",
    "nb_produits": "Nombre de produits",
    "solde_moyen": "Solde moyen (DH)",
    "ln_solde": "ln(Solde moyen)",
    "nb_transactions": "Transactions (6 mois)",
    "bbm": "Usage Barid Bank Mobile",
    "salaire_dom": "Salaire domicilié",
    "credit": "Crédit en cours",
    "reclamations": "Réclamations (12 mois)",
    "churn": "Churn (cible)",
}
FEATURES_DEFAUT = ["age", "anciennete", "nb_produits", "ln_solde",
                   "nb_transactions", "bbm", "salaire_dom", "credit",
                   "reclamations"]


@st.cache_data(show_spinner=False)
def simuler_donnees(n: int, graine: int) -> pd.DataFrame:
    """Génère un portefeuille simulé (générateur identique à celui du rapport ;
    n=4000 et graine=42 reproduisent exactement les chiffres du mémoire)."""
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
    """Lecture robuste d'un CSV réel (séparateur auto, encodage de secours)."""
    contenu = fichier.getvalue()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contenu), sep=None,
                               engine="python", encoding=enc)
        except Exception:
            continue
    raise ValueError("Format de fichier non reconnu.")


def preparer_X(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Encodage one-hot des variables qualitatives + imputation médiane/mode."""
    X = df[features].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median())
        else:  # texte / catégorielle (couvre object, category et str Arrow)
            X[c] = X[c].astype("object")
            mode = X[c].mode()
            X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "Inconnu")
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    return X.fillna(0.0)


def lab(col):
    return VARS_FR.get(col, col.replace("_", " ").capitalize())


# ============================================================================
# 2. BARRE LATÉRALE : IDENTITÉ + NAVIGATION + ÉTAT
# ============================================================================
with st.sidebar:
    st.markdown(
        f"<div style='background:#FFFFFF;border-radius:10px;padding:10px;"
        f"text-align:center;margin-bottom:8px;'>"
        f"<img src='data:image/png;base64,{LOGO_B64}' width='150'></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;font-weight:700;font-size:1.02rem;'>"
        f"Churn Analytics</div>"
        f"<div style='text-align:center;font-size:0.8rem;'>Attrition des clients — PFA</div>",
        unsafe_allow_html=True)
    st.markdown("---")
    PAGES = ["🏠 Accueil", "🗄️ Données", "🔍 Exploration",
             "⚙️ Modélisation", "✅ Évaluation", "⭐ Scoring",
             "👤 Simulateur client", "💡 Recommandations", "ℹ️ À propos"]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    ok_d = "data" in st.session_state
    ok_m = "modeles" in st.session_state
    st.markdown(
        f"{'🟢' if ok_d else '⚪'} Données&nbsp;: "
        f"{'chargées' if ok_d else 'à charger'}<br>"
        f"{'🟢' if ok_m else '⚪'} Modèle&nbsp;: "
        f"{'entraîné' if ok_m else 'à entraîner'}",
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


# ============================================================================
# 3. PAGE ACCUEIL
# ============================================================================
if page == "🏠 Accueil":
    bandeau("Analyse et prédiction de l'attrition des clients (Customer Churn)",
            "Al Barid Bank — Agence Drissia, Tanger · Projet de Fin d'Année")

    st.markdown(
        "<div class='carte-jaune'><b>Problématique.</b> Comment identifier, "
        "de manière précoce et objective, les clients susceptibles de quitter "
        "Al Barid Bank, afin de mettre en place des actions préventives de "
        "rétention ciblées et efficaces&nbsp;?</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    etapes = [
        ("🗄️", "1. Données", "Simulez un portefeuille (taille au choix) ou importez votre CSV réel anonymisé."),
        ("⚙️", "2. Modèle", "Régression logistique interprétable, comparée à un arbre et à une forêt aléatoire."),
        ("⭐", "3. Scoring", "Probabilités converties en score de fidélité 300-900 et en 4 classes de risque."),
        ("💡", "4. Actions", "Plan de rétention différencié par classe, listes de clients à contacter exportables."),
    ]
    for c, (ic, t, d) in zip([c1, c2, c3, c4], etapes):
        c.markdown(f"<div class='carte' style='min-height:150px'>"
                   f"<div style='font-size:1.6rem'>{ic}</div>"
                   f"<b>{t}</b><br><span style='color:{MUT};font-size:0.88rem'>"
                   f"{d}</span></div>", unsafe_allow_html=True)

    st.subheader("Repères issus du mémoire (base de référence : 4 000 clients)")
    kpi_row([("13,9 %", "taux d'attrition observé"),
             ("0,805", "AUC — régression logistique"),
             ("85,5 %", "churners détectés (rappel)"),
             ("× 2,04", "cote de churn par réclamation")])
    st.markdown("")
    interpretation(
        "Cette application automatise <b>toute la chaîne du rapport</b> : "
        "vous pouvez reproduire à l'identique les résultats du mémoire "
        "(simulation de 4 000 clients, graine 42), étudier la sensibilité des "
        "résultats à la taille de l'échantillon, ou appliquer la même "
        "méthodologie à un <b>fichier réel anonymisé</b> de l'agence.")

# ============================================================================
# 4. PAGE DONNÉES
# ============================================================================
elif page == "🗄️ Données":
    bandeau("🗄️ Données", "Portefeuille simulé paramétrable ou import d'un CSV réel")

    mode = st.radio("Source des données", ["Simulation (paramétrable)",
                    "Import d'un fichier CSV réel"], horizontal=True)

    # ------------------------------------------------------------ simulation
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
            for k in ("modeles", "eval", "scores"):
                st.session_state.pop(k, None)
            st.success("Portefeuille généré. Le générateur est identique à "
                       "celui du rapport : n = 4 000 et graine = 42 "
                       "reproduisent exactement les chiffres du mémoire.")

    # ---------------------------------------------------------------- import
    else:
        st.markdown(
            "<div class='carte'>Le fichier doit contenir <b>une colonne cible "
            "binaire (0/1)</b> — le churn — et des variables explicatives "
            "numériques ou qualitatives. Les qualitatives sont encodées "
            "automatiquement (one-hot) et les valeurs manquantes imputées "
            "(médiane / mode).</div>", unsafe_allow_html=True)
        up = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
        if up is not None:
            try:
                brut = lire_csv(up)
                st.dataframe(brut.head(8), width="stretch")
                colonnes = list(brut.columns)
                cible_defaut = ("churn" if "churn" in colonnes
                                else colonnes[-1])
                cible = st.selectbox("Colonne cible (churn : 0 = actif, "
                                     "1 = churner)", colonnes,
                                     index=colonnes.index(cible_defaut))
                candidates = [c for c in colonnes if c != cible
                              and not c.lower().startswith("id")]
                feats = st.multiselect("Variables explicatives", candidates,
                                       default=candidates)
                if st.button("✅ Valider ce jeu de données"):
                    y_chk = pd.to_numeric(brut[cible], errors="coerce")
                    if set(y_chk.dropna().unique()) - {0, 1}:
                        st.error("La colonne cible doit être binaire (0/1).")
                    elif len(feats) == 0:
                        st.error("Sélectionnez au moins une variable explicative.")
                    else:
                        df = brut.copy()
                        df[cible] = y_chk.fillna(0).astype(int)
                        st.session_state["data"] = df
                        st.session_state["source"] = f"CSV importé — {up.name}"
                        st.session_state["target"] = cible
                        st.session_state["features"] = feats
                        for k in ("modeles", "eval", "scores"):
                            st.session_state.pop(k, None)
                        st.success("Jeu de données validé.")
            except Exception as e:
                st.error(f"Lecture impossible : {e}")

    # ------------------------------------------------------------- aperçu
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
            dic = pd.DataFrame(
                [(k, v) for k, v in VARS_FR.items()],
                columns=["Champ", "Description"])
            st.table(dic)
        st.download_button(
            "⬇️ Télécharger le jeu de données (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "portefeuille_churn.csv", "text/csv")

# ============================================================================
# 5. PAGE EXPLORATION
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
        fig = go.Figure(go.Pie(
            labels=["Clients actifs", "Churners"],
            values=[1 - taux, taux], hole=0.55,
            marker=dict(colors=[JAUNE, BRUN]),
            textinfo="label+percent"))
        fig.update_layout(title="Répartition du portefeuille",
                          showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        var = st.selectbox(
            "Taux de churn selon…",
            [f for f in feats if f in df.columns],
            format_func=lab)
        serie = df[var]
        if serie.nunique() > 8 and pd.api.types.is_numeric_dtype(serie):
            groupes = pd.qcut(serie, 4, duplicates="drop")
            g = df.groupby(groupes, observed=True)[cible].mean()
            xlabs = [str(i) for i in g.index]
        else:
            g = df.groupby(serie, observed=True)[cible].mean()
            xlabs = [str(i) for i in g.index]
        fig = go.Figure(go.Bar(x=xlabs, y=g.values * 100,
                               marker_color=JAUNE,
                               marker_line=dict(color=BRUN, width=1.2),
                               text=[f"{v*100:.1f}%" for v in g.values],
                               textposition="outside"))
        fig.update_layout(title=f"Taux de churn selon {lab(var).lower()}",
                          yaxis_title="Taux de churn (%)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        ecart = g.max() - g.min()
        interpretation(
            f"Le taux de churn varie de <b>{fr(g.min())}</b> à "
            f"<b>{fr(g.max())}</b> selon <b>{lab(var).lower()}</b>, soit un "
            f"écart de {fr(ecart)} — "
            + ("un pouvoir discriminant important pour le modèle."
               if ecart > 0.05 else
               "une liaison modérée avec l'attrition."))

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        var2 = st.selectbox("Distribution comparée (actifs vs churners)",
                            [f for f in feats if f in df.columns and
                             pd.api.types.is_numeric_dtype(df[f])],
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
        num = [f for f in feats if f in df.columns
               and pd.api.types.is_numeric_dtype(df[f])]
        corr = df[num + [cible]].corr()[cible].drop(cible).sort_values()
        fig = go.Figure(go.Bar(
            x=corr.values, y=[lab(i) for i in corr.index], orientation="h",
            marker_color=[VERT if v < 0 else ROUGE for v in corr.values]))
        fig.update_layout(title="Corrélation de chaque variable avec le churn",
                          xaxis_title="Corrélation", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    interpretation(
        "Les barres <span style='color:#1E8449'><b>vertes</b></span> sont des "
        "facteurs de <b>rétention</b> (corrélation négative avec le churn), "
        "les <span style='color:#B03A2E'><b>rouges</b></span> des facteurs de "
        "<b>risque</b>. Ce diagnostic descriptif sera confirmé — toutes "
        "choses égales par ailleurs — par la régression logistique.")

# ============================================================================
# 6. PAGE MODÉLISATION
# ============================================================================
elif page == "⚙️ Modélisation":
    bandeau("⚙️ Modélisation",
            "Régression logistique interprétable + modèles challengers")
    exiger_donnees()
    df = st.session_state["data"]
    cible = st.session_state["target"]

    c1, c2, c3 = st.columns([1.4, 1, 1])
    feats = c1.multiselect("Variables du modèle",
                           st.session_state["features"],
                           default=st.session_state["features"],
                           format_func=lab)
    part_test = c2.slider("Part de l'échantillon test", 0.1, 0.5, 0.30, 0.05)
    graine = c3.number_input("Graine (partition)", 0, 9999, 42)

    if st.button("🚀 Entraîner les modèles", width="stretch"):
        if not feats:
            st.error("Sélectionnez au moins une variable.")
            st.stop()
        X = preparer_X(df, feats)
        y = df[cible].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=part_test, random_state=int(graine), stratify=y)

        # --- régression logistique (statsmodels : Wald, p-values) ----------
        logit_sm, note = None, ""
        try:
            logit_sm = sm.Logit(y_tr, sm.add_constant(X_tr)).fit(disp=0)
            p_logit = logit_sm.predict(sm.add_constant(X_te))
        except Exception:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
            p_logit = pd.Series(lr.predict_proba(X_te)[:, 1], index=X_te.index)
            logit_sm = lr
            note = ("⚠️ Séparation quasi parfaite détectée : repli sur "
                    "scikit-learn (sans p-values).")

        arbre = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                       random_state=42).fit(X_tr, y_tr)
        foret = RandomForestClassifier(n_estimators=400, max_depth=8,
                                       min_samples_leaf=20,
                                       random_state=42).fit(X_tr, y_tr)
        st.session_state["modeles"] = {
            "logit": logit_sm, "arbre": arbre, "foret": foret,
            "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
            "p": {"Régression logistique": np.asarray(p_logit),
                  "Arbre de décision": arbre.predict_proba(X_te)[:, 1],
                  "Forêt aléatoire": foret.predict_proba(X_te)[:, 1]},
            "note": note, "feats": feats}
        st.session_state.pop("scores", None)
        st.success(f"Modèles entraînés — {len(X_tr)} clients en apprentissage, "
                   f"{len(X_te)} en test (stratifié).")
        if note:
            st.warning(note)

    if "modeles" in st.session_state:
        M = st.session_state["modeles"]
        st.markdown("---")
        st.subheader("Coefficients de la régression logistique")
        if hasattr(M["logit"], "params"):
            res = M["logit"]
            tab = pd.DataFrame({
                "Coefficient": res.params, "Écart-type": res.bse,
                "Odds ratio": np.exp(res.params), "p-value": res.pvalues})
            tab.index = [("Constante" if i == "const" else lab(i))
                         for i in tab.index]
            st.dataframe(
                tab.style.format({"Coefficient": "{:.3f}",
                                  "Écart-type": "{:.3f}",
                                  "Odds ratio": "{:.3f}",
                                  "p-value": "{:.4f}"})
                .map(lambda v: f"color:{VERT};font-weight:700"
                     if isinstance(v, float) and v < 0.05 else "",
                     subset=["p-value"]),
                width="stretch")
            st.caption(f"Pseudo-R² de McFadden : {res.prsquared:.3f} · "
                       f"log-vraisemblance : {res.llf:.1f} · "
                       f"{int(res.nobs)} observations d'apprentissage.")

            orr = np.exp(res.params.drop("const")).sort_values()
            fig = go.Figure(go.Bar(
                x=orr.values, y=[lab(i) for i in orr.index], orientation="h",
                marker_color=[VERT if v < 1 else ROUGE for v in orr.values],
                text=[f"{v:.2f}" for v in orr.values], textposition="outside"))
            fig.add_vline(x=1, line_color=TXT)
            fig.update_layout(title="Odds ratios — facteurs de risque (>1) "
                              "et de rétention (<1)",
                              xaxis_title="Odds ratio", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")

            risque, retention = orr.idxmax(), orr.idxmin()
            interpretation(
                f"Le premier facteur de <b>risque</b> est "
                f"<b>{lab(risque).lower()}</b> : chaque unité supplémentaire "
                f"multiplie la cote de churn par <b>{orr.max():.2f}</b>. À "
                f"l'inverse, <b>{lab(retention).lower()}</b> est le facteur de "
                f"<b>rétention</b> le plus puissant (cote divisée par "
                f"<b>{1/orr.min():.1f}</b>). Les p-values en vert sont "
                f"significatives au seuil de 5&nbsp;%.")
        else:
            st.info("Modèle scikit-learn de repli : coefficients sans "
                    "p-values.")

        with st.expander("🌲 Importance des variables — forêt aléatoire"):
            imp = pd.Series(M["foret"].feature_importances_,
                            index=M["X_tr"].columns).sort_values()
            fig = go.Figure(go.Bar(x=imp.values,
                                   y=[lab(i) for i in imp.index],
                                   orientation="h", marker_color=JAUNE,
                                   marker_line=dict(color=BRUN, width=1)))
            fig.update_layout(title="Importance (réduction d'impureté)",
                              **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")

# ============================================================================
# 7. PAGE ÉVALUATION
# ============================================================================
elif page == "✅ Évaluation":
    bandeau("✅ Évaluation des modèles",
            "Courbes ROC, choix du seuil et matrice de confusion (échantillon test)")
    exiger_modele()
    M = st.session_state["modeles"]
    y_te = M["y_te"]

    # ------------------------------------------------------------ ROC + AUC
    fig = go.Figure()
    couleurs = {"Régression logistique": "#003366",
                "Forêt aléatoire": VERT, "Arbre de décision": MUT}
    lignes = []
    for nom, p_hat in M["p"].items():
        fpr, tpr, _ = roc_curve(y_te, p_hat)
        a = auc(fpr, tpr)
        lignes.append((nom, a, 2 * a - 1))
        fig.add_scatter(x=fpr, y=tpr, mode="lines",
                        name=f"{nom} (AUC = {a:.3f})",
                        line=dict(color=couleurs[nom], width=2.4))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire (0,5)",
                    line=dict(color="#BBBBBB", dash="dash"))
    fig.update_layout(title="Courbes ROC sur l'échantillon test",
                      xaxis_title="Taux de faux positifs (1 − spécificité)",
                      yaxis_title="Taux de vrais positifs (sensibilité)",
                      legend=dict(x=0.45, y=0.08), **PLOTLY_LAYOUT)

    c1, c2 = st.columns([1.35, 1])
    c1.plotly_chart(fig, width="stretch")
    with c2:
        comp = pd.DataFrame(lignes, columns=["Modèle", "AUC", "Gini"]) \
                 .sort_values("AUC", ascending=False).reset_index(drop=True)
        st.markdown("**Pouvoir discriminant**")
        st.dataframe(comp.style.format({"AUC": "{:.3f}", "Gini": "{:.3f}"})
                     .highlight_max(subset=["AUC"], color=JAUNE_CLAIR),
                     width="stretch", hide_index=True)
        meilleur = comp.iloc[0]
        interpretation(
            f"<b>{meilleur['Modèle']}</b> domine avec un AUC de "
            f"<b>{num_fr(meilleur['AUC'])}</b> (Gini "
            f"{num_fr(meilleur['Gini'])}) : le modèle classe correctement un "
            f"couple (churner, actif) tiré au hasard dans "
            f"{fr(meilleur['AUC'])} des cas. La régression logistique reste "
            f"en outre <b>parfaitement interprétable</b> — critère décisif en "
            f"banque.")

    # ------------------------------------------------- seuil + matrice
    st.markdown("---")
    st.subheader("Choix du seuil de classification — régression logistique")
    p_logit = M["p"]["Régression logistique"]
    fpr, tpr, thr = roc_curve(y_te, p_logit)
    s_youden = float(thr[np.argmax(tpr - fpr)])
    c1, c2 = st.columns([1, 2])
    auto = c1.toggle("Seuil optimal de Youden", value=True)
    if auto:
        seuil = s_youden
        c1.metric("Seuil retenu", num_fr(seuil))
    else:
        seuil = c2.slider("Seuil manuel", 0.02, 0.90, round(s_youden, 3), 0.005)
    y_pred = (p_logit >= seuil).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    vn, fp, fn, vp = cm.ravel()

    c3, c4 = st.columns([1.15, 1])
    with c3:
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Prédit actif", "Prédit churner"],
            y=["Réel actif", "Réel churner"],
            colorscale=[[0, "#FFF6DC"], [1, BRUN]], showscale=False,
            text=[[f"Vrais négatifs<br>{vn}", f"Faux positifs<br>{fp}"],
                  [f"Faux négatifs<br>{fn}", f"Vrais positifs<br>{vp}"]],
            texttemplate="%{text}",
            textfont=dict(size=14)))
        fig.update_layout(title=f"Matrice de confusion (seuil = {seuil:.3f})",
                          yaxis_autorange="reversed", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c4:
        met = {
            "Exactitude": accuracy_score(y_te, y_pred),
            "Rappel (churners détectés)": recall_score(y_te, y_pred),
            "Spécificité": vn / (vn + fp) if vn + fp else 0.0,
            "Précision": precision_score(y_te, y_pred, zero_division=0),
            "Score F1": f1_score(y_te, y_pred, zero_division=0),
            "Statistique KS": float((tpr - fpr).max()),
        }
        st.markdown("**Métriques au seuil retenu**")
        st.dataframe(pd.DataFrame(met, index=["Valeur"]).T
                     .style.format(lambda v: num_fr(v)),
                     width="stretch")
    interpretation(
        f"Au seuil de <b>{num_fr(seuil)}</b>, le modèle détecte "
        f"<b>{vp} des {vp + fn} churners</b> du test (rappel "
        f"{fr(met['Rappel (churners détectés)'])}), au prix de {fp} clients "
        f"fidèles sollicités à tort. Ce compromis est assumé : rater un "
        f"churner coûte la valeur vie du client, alors qu'un faux positif ne "
        f"coûte qu'une action commerciale peu onéreuse. C'est pourquoi le "
        f"seuil naïf de 0,5 est écarté au profit du seuil de Youden.")

# ============================================================================
# 8. PAGE SCORING
# ============================================================================
elif page == "⭐ Scoring":
    bandeau("⭐ Scoring du portefeuille",
            "Score de fidélité 300-900 et segmentation en classes de risque")
    exiger_modele()
    df = st.session_state["data"]
    cible = st.session_state["target"]
    M = st.session_state["modeles"]

    X_all = preparer_X(df, M["feats"]).reindex(columns=M["X_tr"].columns,
                                              fill_value=0.0)
    if hasattr(M["logit"], "params"):
        p_all = np.asarray(M["logit"].predict(sm.add_constant(X_all)))
    else:
        p_all = M["logit"].predict_proba(X_all)[:, 1]
    eps = 1e-9
    score = np.clip(600 + 50 * np.log2((1 - p_all + eps) / (p_all + eps)),
                    300, 900)

    def classe(s):
        return ("Faible" if s >= 700 else "Modéré" if s >= 600
                else "Élevé" if s >= 500 else "Critique")

    scored = df.copy()
    scored["p_churn"] = p_all
    scored["score"] = score.round(0).astype(int)
    scored["classe_risque"] = scored["score"].map(classe)
    st.session_state["scores"] = scored

    st.markdown(
        "<div class='carte-jaune'><b>Formule.</b> Score = 600 + 50 × "
        "log₂((1 − p̂) / p̂) — 600 points = cote 1 contre 1 ; chaque tranche "
        "de <b>50 points double la cote de fidélité</b>. Échelle bornée "
        "300-900, lisible par tout conseiller.</div>",
        unsafe_allow_html=True)

    # ------------------------------------------------------- distribution
    fig = go.Figure()
    for a, b, c in [(300, 500, FONDS_CLASSES["Critique"]),
                    (500, 600, FONDS_CLASSES["Élevé"]),
                    (600, 700, FONDS_CLASSES["Modéré"]),
                    (700, 900, FONDS_CLASSES["Faible"])]:
        fig.add_vrect(x0=a, x1=b, fillcolor=c, opacity=0.55, line_width=0)
    fig.add_histogram(x=scored.loc[scored[cible] == 0, "score"],
                      name="Clients actifs", marker_color="#003366",
                      opacity=0.8, nbinsx=45)
    fig.add_histogram(x=scored.loc[scored[cible] == 1, "score"],
                      name="Churners observés", marker_color=ROUGE,
                      opacity=0.8, nbinsx=45)
    for lim in (500, 600, 700):
        fig.add_vline(x=lim, line_dash="dash", line_color=MUT)
    fig.update_layout(barmode="overlay",
                      title="Distribution des scores et zones de risque",
                      xaxis_title="Score de fidélité (300 – 900)",
                      yaxis_title="Nombre de clients", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    # -------------------------------------------------------------- grille
    grille = (scored.groupby("classe_risque")
              .agg(Effectif=(cible, "size"), Churn_observe=(cible, "mean"),
                   Score_moyen=("score", "mean"))
              .reindex(["Faible", "Modéré", "Élevé", "Critique"]).fillna(0))
    grille["Part"] = grille["Effectif"] / len(scored)
    grille = grille[["Effectif", "Part", "Churn_observe", "Score_moyen"]]
    grille.columns = ["Effectif", "Part du portefeuille",
                      "Taux de churn observé", "Score moyen"]

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("**Grille de score et validation**")
        st.dataframe(
            grille.style.format({"Effectif": "{:,.0f}", "Part du portefeuille":
                                 lambda v: fr(v), "Taux de churn observé":
                                 lambda v: fr(v), "Score moyen": "{:.0f}"})
            .apply(lambda r: [f"background-color:{FONDS_CLASSES.get(r.name,'')}"]
                   * len(r), axis=1),
            width="stretch")
    with c2:
        mono = grille["Taux de churn observé"].dropna().is_monotonic_increasing
        interpretation(
            "Le taux de churn observé "
            + ("<b>croît strictement</b> d'une classe à l'autre : le score "
               "ordonne correctement les clients selon leur risque réel — "
               "c'est la validation opérationnelle de la grille."
               if mono else
               "n'est pas parfaitement monotone sur cet échantillon : "
               "vérifiez la taille des classes extrêmes (effectifs faibles) "
               "ou ré-entraînez sur davantage de données."))
    st.download_button(
        "⬇️ Télécharger le portefeuille scoré (CSV)",
        scored.to_csv(index=False).encode("utf-8"),
        "portefeuille_score.csv", "text/csv")

# ============================================================================
# 9. PAGE SIMULATEUR CLIENT
# ============================================================================
elif page == "👤 Simulateur client":
    bandeau("👤 Simulateur client",
            "Estimez en direct la probabilité d'attrition d'un client donné")
    exiger_modele()
    df = st.session_state["data"]
    M = st.session_state["modeles"]

    st.markdown("<div class='carte'>Renseignez le profil du client : le "
                "modèle calcule sa probabilité de churn, son score et sa "
                "classe de risque, puis propose l'action adaptée.</div>",
                unsafe_allow_html=True)

    valeurs = {}
    cols = st.columns(3)
    for i, f in enumerate(M["feats"]):
        c = cols[i % 3]
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            uniq = sorted(s.dropna().unique())
            if set(uniq) <= {0, 1}:
                valeurs[f] = 1 if c.selectbox(
                    lab(f), ["Non", "Oui"],
                    index=int(round(s.mean()))) == "Oui" else 0
            else:
                lo, hi = float(s.min()), float(s.max())
                med = float(s.median())
                entier = pd.api.types.is_integer_dtype(s)
                valeurs[f] = c.number_input(
                    lab(f), lo, hi, med, step=1.0 if entier else 0.1)
        else:
            opts = sorted(s.dropna().astype(str).unique())
            valeurs[f] = c.selectbox(lab(f), opts)

    if "solde_moyen" in valeurs and "ln_solde" in M["feats"]:
        valeurs["ln_solde"] = float(np.log1p(valeurs["solde_moyen"]))

    if st.button("🎯 Évaluer ce client", width="stretch"):
        x = pd.DataFrame([valeurs])
        x = preparer_X(x, M["feats"]).reindex(columns=M["X_tr"].columns,
                                              fill_value=0.0)
        if hasattr(M["logit"], "params"):
            p = float(np.asarray(M["logit"].predict(sm.add_constant(x, has_constant="add"))).ravel()[0])
        else:
            p = float(M["logit"].predict_proba(x)[0, 1])
        sc = float(np.clip(600 + 50 * np.log2((1 - p + 1e-9) / (p + 1e-9)),
                           300, 900))
        cl = ("Faible" if sc >= 700 else "Modéré" if sc >= 600
              else "Élevé" if sc >= 500 else "Critique")

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sc,
                number={"font": {"color": BRUN}},
                title={"text": "Score de fidélité", "font": {"color": BRUN}},
                gauge={
                    "axis": {"range": [300, 900]},
                    "bar": {"color": BRUN},
                    "steps": [
                        {"range": [300, 500], "color": FONDS_CLASSES["Critique"]},
                        {"range": [500, 600], "color": FONDS_CLASSES["Élevé"]},
                        {"range": [600, 700], "color": FONDS_CLASSES["Modéré"]},
                        {"range": [700, 900], "color": FONDS_CLASSES["Faible"]}],
                }))
            fig.update_layout(height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(
                f"<div class='carte' style='text-align:center'>"
                f"<div style='color:{MUT}'>Probabilité de churn (6 mois)</div>"
                f"<div style='font-size:2.3rem;font-weight:700;color:{BRUN}'>"
                f"{fr(p)}</div>"
                f"<span class='badge' style='background:"
                f"{FONDS_CLASSES[cl]};color:{COULEURS_CLASSES[cl]}'>"
                f"Risque {cl.lower()}</span></div>", unsafe_allow_html=True)
            actions = {
                "Faible": "Entretenir la relation : communication via BBM, "
                          "programme de parrainage.",
                "Modéré": "Campagne d'équipement : proposer un 2ᵉ produit, "
                          "activer BBM, domicilier le salaire.",
                "Élevé": "Contact proactif du conseiller sous 15 jours, geste "
                         "commercial, suivi CRM.",
                "Critique": "Traitement prioritaire par le directeur : appel "
                            "sous 7 jours, réclamations accélérées, offre de "
                            "rétention sur mesure.",
            }
            st.markdown(f"<div class='carte-jaune'><b>Action recommandée.</b> "
                        f"{actions[cl]}</div>", unsafe_allow_html=True)
        interpretation(
            f"Un score de <b>{sc:.0f}</b> points place ce client en risque "
            f"<b>{cl.lower()}</b>. Rappel de lecture : 600 points = cote 1 "
            f"contre 1 ; chaque tranche de 50 points double la cote de "
            f"fidélité.")

# ============================================================================
# 10. PAGE RECOMMANDATIONS
# ============================================================================
elif page == "💡 Recommandations":
    bandeau("💡 Recommandations opérationnelles",
            "Du score à l'action : plan de rétention pour l'agence")
    exiger_modele()
    if "scores" not in st.session_state:
        st.info("⬅️ Passez d'abord par le module **⭐ Scoring** pour scorer "
                "le portefeuille.")
        st.stop()
    scored = st.session_state["scores"]
    cible = st.session_state["target"]

    comptes = scored["classe_risque"].value_counts()
    kpi_row([(f"{int(comptes.get(c, 0)):,}".replace(",", " "),
              f"clients — risque {c.lower()}")
             for c in ["Faible", "Modéré", "Élevé", "Critique"]])
    st.markdown("")

    plans = {
        "Critique": ("🔴", "Traitement prioritaire — direction d'agence",
                     ["Appel du directeur sous 7 jours",
                      "Résolution accélérée des réclamations ouvertes",
                      "Offre de rétention sur mesure ; à défaut, entretien de "
                      "sortie documenté"]),
        "Élevé": ("🟠", "Contact proactif — chargé de clientèle",
                  ["Entretien de découverte des motifs d'insatisfaction sous "
                   "15 jours", "Geste commercial ciblé (exonération, "
                   "revalorisation du package)", "Suivi personnalisé inscrit "
                   "au CRM"]),
        "Modéré": ("🟡", "Campagnes d'équipement ciblées",
                   ["Proposer un 2ᵉ produit adapté (CEN, carte, "
                    "bancassurance)", "Activation de Barid Bank Mobile", 
                    "Domiciliation du salaire — les 3 leviers les plus forts "
                    "du modèle"]),
        "Faible": ("🟢", "Entretenir la relation à moindre coût",
                   ["Communication digitale via BBM", "Information sur les "
                    "nouveautés", "Programme de parrainage (client "
                    "prescripteur)"]),
    }
    for cl, (ic, titre, acts) in plans.items():
        n = int(comptes.get(cl, 0))
        with st.expander(f"{ic} Risque {cl.lower()} — {n} client(s) · {titre}",
                         expanded=(cl in ("Critique", "Élevé"))):
            for a in acts:
                st.markdown(f"- {a}")

    st.markdown("---")
    st.subheader("Recommandations transversales issues du modèle")
    st.markdown(
        "<div class='carte'>"
        "1️⃣ <b>Soigner la première année</b> — parcours d'onboarding "
        "(appels à 1, 3 et 6 mois) : le churn est maximal chez les clients "
        "récents.<br>"
        "2️⃣ <b>BBM dès l'ouverture</b> — l'activation assistée divise la "
        "cote de churn par près de trois.<br>"
        "3️⃣ <b>Traiter chaque réclamation comme une opportunité</b> — une "
        "réclamation double la cote de départ.<br>"
        "4️⃣ <b>Surveiller les signaux faibles</b> — cartes non retirées, "
        "chute des transactions → alertes CRM.<br>"
        "5️⃣ <b>Industrialiser le score</b> — recalcul mensuel, suivi des "
        "migrations de classes, groupe témoin.</div>",
        unsafe_allow_html=True)

    a_contacter = scored[scored["classe_risque"].isin(["Élevé", "Critique"])] \
        .sort_values("score")
    st.download_button(
        f"⬇️ Exporter la liste des {len(a_contacter)} clients à contacter "
        f"(risque élevé + critique)",
        a_contacter.to_csv(index=False).encode("utf-8"),
        "clients_a_contacter.csv", "text/csv")
    interpretation(
        f"Sur ce portefeuille, la campagne prioritaire cible "
        f"<b>{len(a_contacter)} clients</b> (classes élevé et critique), soit "
        f"{fr(len(a_contacter)/len(scored))} du portefeuille — un effort "
        f"commercial concentré là où le risque d'attrition est maximal.")

# ============================================================================
# 11. PAGE À PROPOS
# ============================================================================
else:
    bandeau("ℹ️ À propos", "Cadre académique, méthodologie et avertissements")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div class='carte'><b>Projet.</b> Application développée dans le "
            "cadre du Projet de Fin d'Année (PFA) — « Analyse et prédiction "
            "de l'attrition des clients (Customer Churn) », stage effectué au "
            "sein d'Al Barid Bank, agence Drissia (Tanger).<br><br>"
            "<b>Auteur :</b> WALID [Nom] — Filière Ingénierie Financière et "
            "Actuarielle, FST Errachidia (Université Moulay Ismaïl).<br>"
            "<b>Encadrant pédagogique :</b> Pr. Lhoucine BEN HSSAIN.<br>"
            "<b>Encadrant professionnel :</b> M. [Encadrant agence].</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='carte-jaune'><b>Confidentialité.</b> Aucune donnée "
            "réelle de la clientèle d'Al Barid Bank n'est embarquée dans "
            "cette application. Le générateur reproduit des comportements "
            "réalistes ; l'import CSV est prévu pour des données "
            "<b>anonymisées</b> uniquement.</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div class='carte'><b>Chaîne méthodologique.</b><br>"
            "1. Définition opérationnelle du churn (fenêtres 12 + 6 mois)<br>"
            "2. Variables explicatives (profil, équipement, comportement, "
            "qualité de service)<br>"
            "3. Régression logistique (MLE, Wald) vs arbre vs forêt<br>"
            "4. Évaluation : ROC / AUC / Gini, Youden, matrice de confusion<br>"
            "5. Score 300-900 (50 points = cote doublée) et 4 classes<br>"
            "6. Plan de rétention différencié par classe</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='carte'><b>Références.</b> Hosmer &amp; Lemeshow, "
            "<i>Applied Logistic Regression</i> (2013) · James et al., "
            "<i>An Introduction to Statistical Learning</i> (2021) · Thomas "
            "et al., <i>Credit Scoring and Its Applications</i> (2017) · "
            "documentation scikit-learn &amp; statsmodels.</div>",
            unsafe_allow_html=True)
