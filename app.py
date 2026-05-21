import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

import os

# 1. Cargar los datos
archivo = "Como influye escuchar música al estudiar.csv"
if not os.path.exists(archivo):
    archivo = "/content/Como influye escuchar música al estudiar.csv"
df = pd.read_csv(archivo, encoding='latin1')

# Mostrar preguntas descriptivas simples
print("--- ANÁLISIS DESCRIPTIVO ---")

# 1. ¿Qué porcentaje de estudiantes escucha música?
col_musica = '¿Con qué frecuencia escuchas música mientras estudias?'
freq_musica = df[col_musica].value_counts(normalize=True) * 100
print(f"\n1. Frecuencia de escuchar música:\n{freq_musica}")

# 2. ¿Qué nivel educativo escucha música con mayor frecuencia?
col_educacion = '¿Cuál es tu nivel de estudios?'
# Filtramos solo a los que escuchan frecuentemente/siempre
escuchan_musica = df[df[col_musica].isin(['Siempre', 'Casi siempre', 'Frecuentemente'])]
nivel_freq = escuchan_musica[col_educacion].value_counts()
print(f"\n2. Nivel educativo que más escucha música:\n{nivel_freq}")

# 3. Actividades académicas más comunes
col_actividades = '¿En qué tipo de actividades te ayuda más la música?  '
print(f"\n3. Actividades que más se realizan con música:\n{df[col_actividades].value_counts().head(3)}")

print("\n--- ANÁLISIS DE CORRELACIÓN ---")

# 4. Relación entre escuchar música y concentración (Prueba Chi-Cuadrado)
col_concentracion = 'En una escala del 1 al 5, ¿cuánto crees que la música mejora tu concentración?'
contingencia_conc = pd.crosstab(df[col_musica], df[col_concentracion])
chi2, p, dof, ex = stats.chi2_contingency(contingencia_conc)
print(f"\n4. Relación Música-Concentración (Chi-cuadrado): p-valor = {p:.4f}")
if p < 0.05:
    print("   -> SÍ existe una relación estadísticamente significativa.")
else:
    print("   -> NO hay una relación estadísticamente significativa.")

# 5. Relación entre tipo de música y rendimiento académico
col_genero = '¿Qué género escuchas principalmente al estudiar?'
col_rendimiento = 'Notas que al estudiar con música cambia tu rendimiento académico'
contingencia_rend = pd.crosstab(df[col_genero], df[col_rendimiento])
chi2_r, p_r, dof_r, ex_r = stats.chi2_contingency(contingencia_rend)
print(f"\n5. Relación Género Musical-Rendimiento (Chi-cuadrado): p-valor = {p_r:.4f}")

print("\n--- MACHINE LEARNING: PERFILES (K-MEANS) ---")
# Codifica variables de texto a números para que K-Means pueda leerlas
encoder = LabelEncoder()
columnas_perfil = [col_musica, col_concentracion, '¿Has dejado de estudiar por distraerte con la música? ']

# Llenar vacíos si los hay
df_limpio = df[columnas_perfil].fillna('No responde')

df_encoded = pd.DataFrame()
for col in columnas_perfil:
    df_encoded[col] = encoder.fit_transform(df_limpio[col].astype(str))

# Aplicar K-Means para encontrar 4 perfiles
kmeans = KMeans(n_clusters=4, random_state=42)
df_limpio['Perfil_Asignado'] = kmeans.fit_predict(df_encoded)

print("Se han agrupado a los estudiantes en 4 perfiles basados en sus respuestas.")
print("Distribución de los perfiles encontrados:")
print(df_limpio['Perfil_Asignado'].value_counts())