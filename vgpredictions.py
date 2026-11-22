import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


