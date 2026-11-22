import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor


df = pd.read_csv('vgsales.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

print(df[df["Year"].isnull()])
print(df[df["Publisher"].isnull()])

# 1. Describe de las filas SIN Publisher
print("=== Filas SIN Publisher ===")
print(df[df["Publisher"].isnull()]["Global_Sales"].describe())

# 2. Describe GLOBAL (todas las filas)
print("\n=== Global (todas las filas) ===")
print(df["Global_Sales"].describe())

# Drop de las 58 filas sin Publisher
df = df.dropna(subset=["Publisher"]).copy()  # .copy() para evitar SettingWithCopyWarning
print(df.shape)                    # debería bajar ~58 filas
print(df["Publisher"].isnull().sum())  # debe ser 0

print(df.groupby("Platform")["Year"].agg(["min", "max", "median", "count"]).sort_values("median"))


print(df[(df["Platform"] == "DS") & (df["Year"] < 2004)] )


# Año de lanzamiento real de cada plataforma (conocimiento de dominio)
launch_years = {
    "2600": 1977, "NES": 1983, "SNES": 1990, "GEN": 1988, "GB": 1989,
    "PS": 1994, "N64": 1996, "SAT": 1994, "DC": 1998, "PS2": 2000,
    "GC": 2001, "GBA": 2001, "XB": 2001, "DS": 2004, "PSP": 2004,
    "Wii": 2006, "X360": 2005, "PS3": 2006, "3DS": 2011, "PSV": 2011,
    "WiiU": 2012, "PS4": 2013, "XOne": 2013,
    # ... añade las que falten
}

# Marcar filas con año anterior al lanzamiento de su plataforma
mask_error = df.apply(
    lambda r: r["Year"] < launch_years.get(r["Platform"], 0), axis=1
)
print("Filas con año imposible:", mask_error.sum())
print(df[mask_error][["Name", "Platform", "Year", "Global_Sales"]])


df = df[~mask_error].copy() 

print(df.dtypes)

mediana_plataforma = df.groupby("Platform")["Year"].median()
print(mediana_plataforma)

df["Year"] = df["Year"].fillna(df["Platform"].map(mediana_plataforma))

print(df.info()) #its workinnnnnnggggg yeii


print(df["Publisher"].unique())

# 1. Normalización mínima (mayúsculas/espacios) — rápido
df["Publisher"] = df["Publisher"].str.strip()

mapa_unificacion = {
    # Variantes de escritura
    "Valve Software": "Valve",
    "Interplay Productions": "Interplay",
    "Rebellion Developments": "Rebellion",
    "Daedalic Entertainment": "Daedalic",
    "Paon Corporation": "Paon",
    "Avanquest Software": "Avanquest",
    "System 3 Arcade Software": "System 3",
    "Falcom Corporation": "Nihon Falcom Corporation",
    "Milestone": "Milestone S.r.l.",
    "Milestone S.r.l": "Milestone S.r.l.",
    "Big Ben Interactive": "Bigben Interactive",
    "imageepoch Inc.": "Image Epoch",
    # Grupos con varias etiquetas
    "Marvelous Entertainment": "Marvelous Games",
    "Marvelous Interactive": "Marvelous Games",
    "Idea Factory International": "Idea Factory",
    "Codemasters Online": "Codemasters",
    # Eras históricas (opcional — yo lo haría)
    "SquareSoft": "Square Enix",
    "Square": "Square Enix",
    "Enix Corporation": "Square Enix",
    "SNK Playmore": "SNK",
    # Basura evidente
    "7G//AMES": "7GAMES",
}

antes = df["Publisher"].nunique()
df["Publisher"] = df["Publisher"].str.strip().replace(mapa_unificacion)
despues = df["Publisher"].nunique()
print(f"Categorías únicas: {antes} → {despues}")
print("Nulos en Publisher:", df["Publisher"].isnull().sum())


y = df["Global_Sales"]

X = df[["Platform", "Year", "Genre", "Publisher"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



print(f"Train: {X_train.shape} - Test: {X_test.shape}")

preprocesador = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Platform", "Genre", "Publisher"]),
    ("num", "passthrough", ["Year"]),
])


modelo_rf = Pipeline([
    ("prep", preprocesador),
    ("reg", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
])

modelo_rf.fit(X_train, y_train)
y_pred = modelo_rf.predict(X_test)

# 5. Métricas
print("\n=== Random Forest ===")
print("MAE :", mean_absolute_error(y_test, y_pred).__round__(4))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)).__round__(4))
print("R²  :", r2_score(y_test, y_pred).__round__(4))


dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print("=== Baseline tonto (predecir la media) ===")
print("MAE :", mean_absolute_error(y_test, y_dummy).__round__(4))
print("R²  :", r2_score(y_test, y_dummy).__round__(4))


# Target transformado
y_log = np.log1p(df["Global_Sales"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

y_train_log = y_train
y_test_log  = y_test
print("Train:", X_train.shape, "| Test:", X_test.shape)  # debe ser 80/20

modelo_rf.fit(X_train, y_train)
y_pred_log = modelo_rf.predict(X_test)
print("-- con log--")
# R² en escala log (más representativo)
print("R² (log):", r2_score(y_test, y_pred_log).__round__(4))

# MAE en escala original (volviendo con expm1)
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

print("MAE (millones):", mean_absolute_error(y_test_orig, y_pred_orig).__round__(4))
print("RMSE (millones):", np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)).__round__(4))

# ------------------------------------------------------------
# EXPERIMENTO A — Dummy en escala log (referencia correcta)
# ------------------------------------------------------------
dummy_log = DummyRegressor(strategy="mean").fit(X_train, y_train_log)
y_pred_dummy_log = dummy_log.predict(X_test)

r2_dummy_log   = r2_score(y_test_log, y_pred_dummy_log)
mae_dummy_orig = mean_absolute_error(np.expm1(y_test_log), np.expm1(y_pred_dummy_log))

print("=" * 50)
print("=== A) Dummy (media) en log ===")
print(f"R²  (log)      : {r2_dummy_log:.4f}")
print(f"MAE (millones) : {mae_dummy_orig:.4f}")

# ------------------------------------------------------------
# EXPERIMENTO B — Regresión Lineal en log
# ------------------------------------------------------------
modelo_lr = Pipeline([
    ("prep", preprocesador),
    ("reg",  LinearRegression()),
])
modelo_lr.fit(X_train, y_train_log)
y_pred_lr_log = modelo_lr.predict(X_test)

r2_lr_log   = r2_score(y_test_log, y_pred_lr_log)
mae_lr_orig = mean_absolute_error(np.expm1(y_test_log), np.expm1(y_pred_lr_log))
rmse_lr_orig = np.sqrt(np.mean((np.expm1(y_test_log) - np.expm1(y_pred_lr_log)) ** 2))

print("=" * 50)
print("=== B) Linear Regression (log) ===")
print(f"R²  (log)      : {r2_lr_log:.4f}")
print(f"MAE (millones) : {mae_lr_orig:.4f}")
print(f"RMSE(millones) : {rmse_lr_orig:.4f}")

# ------------------------------------------------------------
# EXPERIMENTO C — Validación cruzada 5-fold del RF (log)
# ------------------------------------------------------------
scores_rf_cv = cross_val_score(
    modelo_rf, X, y_log, cv=5, scoring="r2", n_jobs=-1
)

print("=" * 50)
print("=== C) Random Forest — CV 5-fold (log) ===")
print("R² por fold    :", np.round(scores_rf_cv, 4))
print(f"Media          : {scores_rf_cv.mean():.4f} ± {scores_rf_cv.std():.4f}")

# ------------------------------------------------------------
# RESUMEN COMPARATIVO
# ------------------------------------------------------------
print("=" * 50)
print("=== RESUMEN (target log) ===")
print(f"{'Modelo':<22}{'R²(log)':>10}{'MAE(M)':>10}")
print("-" * 42)
print(f"{'Dummy (media)':<22}{r2_dummy_log:>10.4f}{mae_dummy_orig:>10.4f}")
print(f"{'Linear Regression':<22}{r2_lr_log:>10.4f}{mae_lr_orig:>10.4f}")
print(f"{'Random Forest (test)':<22}{0.2035:>10.4f}{0.4852:>10.4f}")  # tus valores previos
print(f"{'Random Forest (CV)':<22}{scores_rf_cv.mean():>10.4f}{'—':>10}")
print("=" * 50)