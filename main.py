# ---------------------------------------------------------
# Aufgabe: Daten einlesen
# Laden der Datei mit korrektem Trennzeichen (;) und Dezimalzeichen (,)
# ---------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

file_path = "AirQualityUCI.csv"
df = pd.read_csv(file_path, delimiter=';', decimal=",", low_memory=False)

# ---------------------------------------------------------
# Aufgabe: Leere Spalten entfernen (z. B. 'Unnamed: 15', 'Unnamed: 16')
# ---------------------------------------------------------
df = df.dropna(axis=1, how='all')

# ---------------------------------------------------------
# Aufgabe: Datensatz erkunden
# Erste und letzte 5 Zeilen anzeigen
# ---------------------------------------------------------
print("\nErste 5 Zeilen des DataFrames:")
print(df.head())

print("\nLetzte 5 Zeilen des DataFrames:")
print(df.tail())

# ---------------------------------------------------------
# Aufgabe: Zusammenfassung
# Spaltennamen, Datentypen und Anzahl Nicht-Null-Werte anzeigen
# ---------------------------------------------------------
print("\nAllgemeine Informationen zum DataFrame:")
print(df.info())

# ---------------------------------------------------------
# Aufgabe: Statistische Kennwerte anzeigen
# ---------------------------------------------------------
print("\nStatistische Übersicht des DataFrames:")
print(df.describe())

# ---------------------------------------------------------
# Aufgabe: Spalten auswählen
# Nur CO(GT) und NO2(GT) anzeigen, erste 10 Zeilen
# ---------------------------------------------------------
selected_columns = df[['CO(GT)', 'NO2(GT)']]
print("\nAusgewählte Spalten (CO(GT) & NO2(GT)):")
print(selected_columns.head(10))

# ---------------------------------------------------------
# Aufgabe: Bedingte Auswahl
# Zeilen mit NO2(GT) > 200
# ---------------------------------------------------------
filtered_df_8 = df[df['NO2(GT)'] > 200]
print("\nZeilen mit NO2(GT) > 200:")
print(filtered_df_8)

# ---------------------------------------------------------
# Aufgabe: Mehrere Bedingungen
# CO(GT) < 1.0 und T < 10°C
# ---------------------------------------------------------
filtered_df = df[(df['CO(GT)'] < 1.0) & (df['T'] < 10)]
print("\nZeilen mit CO(GT) < 1.0 und T < 10:")
print(filtered_df)

# ---------------------------------------------------------
# Aufgabe: Neue Spalte hinzufügen
# CO_per_NO2 = CO(GT) / NO2(GT)
# ---------------------------------------------------------
df['CO_per_NO2'] = df['CO(GT)'] / df['NO2(GT)']
print("\nNeue Spalte CO(GT) / NO2(GT):")
print(df[['CO(GT)', 'NO2(GT)', 'CO_per_NO2']].head())

# ---------------------------------------------------------
# Aufgabe: Temperatur kategorisieren
# Temp_Stufe: unter 5°C = kalt, 5–20°C = mäßig, über 20°C = warm
# ---------------------------------------------------------
def temperatur_label(t):
    if t < 5:
        return "kalt"
    elif 5 <= t <= 20:
        return "mäßig"
    else:
        return "warm"

df['Temp_Stufe'] = df['T'].apply(temperatur_label)
print("\nTemperaturstufen:")
print(df[['T', 'Temp_Stufe']].head())

# ---------------------------------------------------------
# Aufgabe: Zeilen löschen bei fehlendem Benzolwert (C6H6(GT))
# ---------------------------------------------------------
df = df.dropna(subset=['C6H6(GT)'])
print("\nDaten nach Entfernen von Zeilen mit fehlendem C6H6(GT):")
print(df.head())

# ---------------------------------------------------------
# Aufgabe: Spaltenüberschriften anzeigen
# ---------------------------------------------------------
print("\nSpaltenüberschriften des DataFrames:")
print(df.columns)

# ---------------------------------------------------------
# Aufgabe: Spaltenüberschriften ersetzen durch deutsche Bezeichnungen
# ---------------------------------------------------------
deutsch_columns = {
    'CO(GT)': 'fester Säuregehalt',
    'PT08.S1(CO)': 'flüchtiger Säuregehalt',
    'NMHC(GT)': 'Zitronensäure',
    'C6H6(GT)': 'Restzucker',
    'PT08.S2(NMHC)': 'Chloride',
    'NO2(GT)': 'freies Schwefeldioxid',
    'PT08.S4(NO2)': 'Gesamtschwefeldioxid',
    'PT08.S5(O3)': 'Dichte',
    'T': 'Temperatur',
    'RH': 'Sulfate',
    'AH': 'Alkohol',
}

df.rename(columns=deutsch_columns, inplace=True)
print("\nSpaltennamen nach Umbenennung:")
print(df.columns)

# ---------------------------------------------------------
# Aufgabe: Scatterplot mit Seaborn
# CO(GT) vs. NO2(GT) visualisieren
# ---------------------------------------------------------
print("\nErstelle Scatterplot zwischen CO und NO2:")
sns.scatterplot(x=df['fester Säuregehalt'], y=df['freies Schwefeldioxid'])
plt.xlabel("Fester Säuregehalt")
plt.ylabel("Freies Schwefeldioxid")
plt.title("CO vs. NO2")
plt.show()

# ---------------------------------------------------------
# Aufgabe: KMeans-Clustering
# - Nicht-numerische Spalten entfernen
# - Clustering durchführen
# - Ergebnis als neue Spalte 'cluster' hinzufügen
# ---------------------------------------------------------
print("\nStarte KMeans Clustering...")

# Nicht-numerische Spalten entfernen
data_unknown = df.drop(['Date', 'Time'], axis=1, errors='ignore')
data_unknown = data_unknown.select_dtypes(include='number').dropna()

# KMeans-Modell erstellen
model = KMeans()

# Elbow-Methode zur Bestimmung optimaler Cluster-Anzahl
visualizer = KElbowVisualizer(model, k=(2, 9))
visualizer.fit(data_unknown)
visualizer.show()

# KMeans mit fester Clusteranzahl (z.B. 3)
kmeans = KMeans(n_clusters=3)
pred = kmeans.fit_predict(data_unknown)

# Cluster-Spalte hinzufügen
df['cluster'] = pred
print("\nDataFrame mit Cluster-Zugehörigkeit:")
print(df[['fester Säuregehalt', 'freies Schwefeldioxid', 'Temperatur', 'cluster']].head())

# Optional: Ergebnis als CSV speichern
df.to_csv("data_mit_clustering.csv", index=False)
