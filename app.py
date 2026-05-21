import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

import os
import sys

# Forzar salida UTF-8 en Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Colores ANSI ---
class Color:
    RESET   = '\033[0m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    # Texto
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'
    # Fondos
    BG_BLUE    = '\033[44m'
    BG_GREEN   = '\033[42m'
    BG_RED     = '\033[41m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN    = '\033[46m'

def titulo_seccion(texto, color=Color.CYAN):
    """Imprime un título de sección con formato destacado."""
    ancho = 60
    linea = "═" * ancho
    print(f"\n{color}{Color.BOLD}{linea}")
    print(f"  {texto.upper()}")
    print(f"{linea}{Color.RESET}")

def subtitulo(numero, texto):
    """Imprime un subtítulo numerado."""
    print(f"\n{Color.YELLOW}{Color.BOLD}  {numero}. {texto}{Color.RESET}")

def separador():
    print(f"{Color.DIM}{'─' * 60}{Color.RESET}")

def resultado_sig(p_valor, umbral=0.05):
    """Devuelve texto coloreado según significancia estadística."""
    if p_valor < umbral:
        return f"{Color.GREEN}{Color.BOLD}SÍ{Color.RESET}{Color.GREEN} existe una relación estadísticamente significativa.{Color.RESET}"
    else:
        return f"{Color.RED}{Color.BOLD}NO{Color.RESET}{Color.RED} hay una relación estadísticamente significativa.{Color.RESET}"

def p_valor_color(p):
    """Colorea el p-valor según su significancia."""
    color = Color.GREEN if p < 0.05 else Color.RED
    return f"{color}{Color.BOLD}{p:.4f}{Color.RESET}"

def barra_horizontal(valor, maximo, ancho=30, color=Color.CYAN):
    """Genera una barra horizontal proporcional al valor."""
    longitud = int((valor / maximo) * ancho) if maximo > 0 else 0
    barra = "█" * longitud + "░" * (ancho - longitud)
    return f"{color}{barra}{Color.RESET}"

# Habilitar colores en Windows
os.system('')

# 1. Cargar los datos
archivo = "Como influye escuchar música al estudiar.csv"
if not os.path.exists(archivo):
    archivo = "/content/Como influye escuchar música al estudiar.csv"
df = pd.read_csv(archivo, encoding='latin1')

# ============================================================
# ANÁLISIS DESCRIPTIVO
# ============================================================
titulo_seccion("Análisis Descriptivo", Color.CYAN)

# 1. ¿Qué porcentaje de estudiantes escucha música?
col_musica = '¿Con qué frecuencia escuchas música mientras estudias?'
freq_musica = df[col_musica].value_counts(normalize=True) * 100

subtitulo(1, "Frecuencia de escuchar música")
max_val = freq_musica.max()
for categoria, porcentaje in freq_musica.items():
    barra = barra_horizontal(porcentaje, max_val, ancho=25)
    # Color según nivel
    if porcentaje > 30:
        c = Color.GREEN
    elif porcentaje > 15:
        c = Color.YELLOW
    else:
        c = Color.RED
    print(f"    {barra} {c}{porcentaje:5.1f}%{Color.RESET}  {categoria}")

# 2. ¿Qué nivel educativo escucha música con mayor frecuencia?
col_educacion = '¿Cuál es tu nivel de estudios?'
escuchan_musica = df[df[col_musica].isin(['Siempre', 'Casi siempre', 'Frecuentemente'])]
nivel_freq = escuchan_musica[col_educacion].value_counts()

subtitulo(2, "Nivel educativo que más escucha música")
max_nivel = nivel_freq.max()
for nivel, conteo in nivel_freq.items():
    barra = barra_horizontal(conteo, max_nivel, ancho=25, color=Color.MAGENTA)
    print(f"    {barra} {Color.WHITE}{Color.BOLD}{conteo:3d}{Color.RESET}  {nivel}")

# 3. Actividades académicas más comunes
col_actividades = '¿En qué tipo de actividades te ayuda más la música?  '

subtitulo(3, "Actividades que más se realizan con música")
top_actividades = df[col_actividades].value_counts().head(3)
max_act = top_actividades.max()
colores_act = [Color.GREEN, Color.CYAN, Color.YELLOW]
for i, (actividad, conteo) in enumerate(top_actividades.items()):
    color = colores_act[i] if i < len(colores_act) else Color.WHITE
    barra = barra_horizontal(conteo, max_act, ancho=25, color=color)
    medalla = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
    print(f"    {medalla} {barra} {color}{Color.BOLD}{conteo:3d}{Color.RESET}  {actividad}")

# ============================================================
# ANÁLISIS DE CORRELACIÓN
# ============================================================
titulo_seccion("Análisis de Correlación", Color.YELLOW)

# 4. Relación entre escuchar música y concentración
col_concentracion = 'En una escala del 1 al 5, ¿cuánto crees que la música mejora tu concentración?'
contingencia_conc = pd.crosstab(df[col_musica], df[col_concentracion])
chi2, p, dof, ex = stats.chi2_contingency(contingencia_conc)

subtitulo(4, "Relación Música → Concentración (Chi-cuadrado)")
print(f"    p-valor = {p_valor_color(p)}")
print(f"    → {resultado_sig(p)}")

# 5. Relación entre tipo de música y rendimiento académico
col_genero = '¿Qué género escuchas principalmente al estudiar?'
col_rendimiento = 'Notas que al estudiar con música cambia tu rendimiento académico'
contingencia_rend = pd.crosstab(df[col_genero], df[col_rendimiento])
chi2_r, p_r, dof_r, ex_r = stats.chi2_contingency(contingencia_rend)

subtitulo(5, "Relación Género Musical → Rendimiento (Chi-cuadrado)")
print(f"    p-valor = {p_valor_color(p_r)}")
print(f"    → {resultado_sig(p_r)}")

# ============================================================
# MACHINE LEARNING
# ============================================================
titulo_seccion("Machine Learning: Perfiles (K-Means)", Color.MAGENTA)

# Codifica variables de texto a números
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

print(f"\n  Se agruparon a los estudiantes en {Color.BOLD}{Color.WHITE}4 perfiles{Color.RESET} basados en sus respuestas.")
separador()

perfiles = df_limpio['Perfil_Asignado'].value_counts().sort_index()
max_perfil = perfiles.max()
colores_perfil = [Color.CYAN, Color.GREEN, Color.YELLOW, Color.MAGENTA]

print(f"  {Color.BOLD}Distribución de los perfiles:{Color.RESET}\n")
for perfil, conteo in perfiles.items():
    color = colores_perfil[perfil % len(colores_perfil)]
    barra = barra_horizontal(conteo, max_perfil, ancho=25, color=color)
    porcentaje = (conteo / perfiles.sum()) * 100
    print(f"    Perfil {color}{Color.BOLD}{perfil}{Color.RESET}  {barra} {Color.WHITE}{Color.BOLD}{conteo:3d}{Color.RESET} estudiantes ({color}{porcentaje:.1f}%{Color.RESET})")

separador()
total = perfiles.sum()
print(f"\n  {Color.DIM}Total de estudiantes analizados: {Color.WHITE}{Color.BOLD}{total}{Color.RESET}")
print()