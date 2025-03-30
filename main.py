# Notwendige Bibliotheken importieren
import pandas as pd               # Für die Arbeit mit Tabellen (DataFrames)
import tensorflow as tf           # Wird hier zwar importiert, aber nicht verwendet (kann ignoriert werden)
import seaborn as sns             # Für schöne Diagramme
from sklearn.model_selection import train_test_split  # Wird hier nicht verwendet (kann ignoriert werden)
import matplotlib.pyplot as plt   # Zum Erstellen von Diagrammen
from sklearn.cluster import KMeans  # Für maschinelles Lernen (Clustering)
from yellowbrick.cluster import KElbowVisualizer  # Zum Visualisieren der besten Clusteranzahl

# 1. CSV-Datei mit Weindaten einlesen
file_path = "winequality-red.csv"
df = pd.read_csv(file_path, delimiter=';')  # Daten mit Semikolon als Trennzeichen einlesen

# 2. Ein Blick in die Daten
print("\nErste 5 Zeilen des DataFrames:")
print(df.head())  # Zeigt die ersten 5 Zeilen

print("\nLetzte 5 Zeilen des DataFrames:")
print(df.tail())  # Zeigt die letzten 5 Zeilen

# 3. Informationen über den gesamten Datensatz
print("\nAllgemeine Informationen zum DataFrame:")
print(df.info())  # Zeigt Datentypen, Anzahl der Einträge usw.

print("\nStatistische Übersicht des DataFrames:")
print(df.describe())  # Zeigt z. B. Mittelwert, Min, Max für jede Spalte

# 4. Nur bestimmte Spalten auswählen: Alkohol und pH-Wert
selected_columns = df[['alcohol', 'pH']]
print("\nErste 10 Zeilen der ausgewählten Spalten (Alkohol & pH):")
print(selected_columns.head(10))

# 5. Zeilen filtern: nur Weine mit Qualität = 8
filtered_df_8 = df[df['quality'] == 8]
print("\nZeilen mit Qualität genau 8:")
print(filtered_df_8)

# 6. Mehrere Bedingungen: Alkohol > 12.5 UND Qualität ≥ 7
filtered_df = df[(df['alcohol'] > 12.5) & (df['quality'] >= 7)]
print("\nZeilen mit Alkoholgehalt > 12.5 und Qualität >= 7:")
print(filtered_df)

# 7. Neue Spalte berechnen: Verhältnis Dichte zu Alkohol
df['density_alcohol_ratio'] = df['density'] / df['alcohol']
print("\nErste Zeilen mit neuer Spalte (Dichte/Alkoholgehalt):")
print(df.head(8))

# 8. Qualität als Text beschreiben statt nur als Zahl
def quality_label(q):
    if q == 3:
        return "sehr schlecht"
    elif q == 4:
        return "schlecht"
    elif q == 5:
        return "okay"
    elif q == 6:
        return "gut"
    else:
        return "sehr gut"

# Neue Spalte mit den Qualitäts-Beschreibungen hinzufügen
df['quality_label'] = df['quality'].apply(quality_label)
print("\nZuordnung der Qualitätswerte:")
print(df[['quality', 'quality_label']].head())

# 9. Zeilen entfernen, wenn pH-Wert kleiner als 3.0
df = df[df['pH'] >= 3.0]
print("\nErste Zeilen nach Entfernen von pH-Werten < 3.0:")
print(df.head())

# 10. Spaltennamen anzeigen
print("\nSpaltenüberschriften des DataFrames:")
print(df.columns)

# 11. Spaltennamen ins Deutsche übersetzen
deutsch_columns = {
    'fixed acidity': 'fester Säuregehalt',
    'volatile acidity': 'flüchtiger Säuregehalt',
    'citric acid': 'Zitronensäure',
    'residual sugar': 'Restzucker',
    'chlorides': 'Chloride',
    'free sulfur dioxide': 'freies Schwefeldioxid',
    'total sulfur dioxide': 'Gesamtschwefeldioxid',
    'density': 'Dichte',
    'pH': 'pH-Wert',
    'sulphates': 'Sulfate',
    'alcohol': 'Alkohol',
    'quality': 'Qualität'
}

df.rename(columns=deutsch_columns, inplace=True)  # Spaltennamen ändern
print("\nErste Zeilen nach Umbenennung der Spalten:")
print(df.head())

# 12. Diagramm: Alkoholgehalt im Vergleich zur Qualität
print("\nErstelle Scatterplot für Alkohol vs. Qualität...")
sns.scatterplot(x=df['Alkohol'], y=df['Qualität'])
plt.xlabel("Alkoholgehalt")
plt.ylabel("Qualität")
plt.title("Alkohol vs. Qualität")
plt.show()  # Zeigt das Diagramm

# 13. KMeans Clustering: Weine automatisch in Gruppen einteilen
print("\n#13 KMeans Qualität")

# Zielspalten entfernen, damit der Algorithmus unvoreingenommen ist
data_unknown = df.drop(['Qualität', 'quality_label'], axis=1)
print(data_unknown.dtypes)  # Zeigt Datentypen der verbleibenden Spalten

# Modell erstellen
model = KMeans()

# Mit dem "Elbow"-Verfahren die beste Anzahl an Gruppen finden
visualizer = KElbowVisualizer(model, k=(2, 9))
visualizer.fit(data_unknown)
visualizer.show()

# Danach Anzahl der Gruppen festlegen (z. B. 4)
kmeans = KMeans(n_clusters=4)

# Daten gruppieren (Clustern)
pred = kmeans.fit_predict(data_unknown)

# Die erkannten Gruppen (Cluster) dem ursprünglichen DataFrame hinzufügen
data_new = pd.concat([df, pd.DataFrame(pred, columns=['label'])], axis=1)
print(data_new)

# Neue Datei mit den zusätzlichen Gruppierungen speichern
data_new.to_csv("./data_new.csv")
