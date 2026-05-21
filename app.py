import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Forzar salida UTF-8 en Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Colores ANSI ---
class Color:
    RESET   = '\033[0m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'
    BG_BLUE    = '\033[44m'
    BG_GREEN   = '\033[42m'
    BG_RED     = '\033[41m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN    = '\033[46m'

def titulo_seccion(texto, color=Color.CYAN):
    """Imprime un título de sección con formato destacado."""
    ancho = 70
    linea = "═" * ancho
    print(f"\n{color}{Color.BOLD}{linea}")
    print(f"  {texto.upper()}")
    print(f"{linea}{Color.RESET}")

def subtitulo(num, texto):
    """Imprime un subtítulo numerado."""
    print(f"\n{Color.YELLOW}{Color.BOLD}  {num}. {texto}{Color.RESET}")

def separador():
    print(f"{Color.DIM}{'─' * 70}{Color.RESET}")

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

def conclusion(texto):
    """Imprime una conclusión destacada."""
    print(f"\n    {Color.GREEN}{Color.BOLD}→ Conclusión:{Color.RESET} {Color.WHITE}{texto}{Color.RESET}")

# Habilitar colores en Windows
os.system('')

# ══════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════
archivo = "Como influye escuchar música al estudiar.csv"
if not os.path.exists(archivo):
    archivo = "/content/Como influye escuchar música al estudiar.csv"

df = pd.read_csv(archivo, encoding='latin1')
df.columns = df.columns.str.strip()  # Limpiar espacios en nombres de columnas

# --- Alias de columnas para código más legible ---
COL_FREQ        = '¿Con qué frecuencia escuchas música mientras estudias?'
COL_EDU         = '¿Cuál es tu nivel de estudios?'
COL_ACT         = '¿En qué tipo de actividades te ayuda más la música?'
COL_CONC        = 'En una escala del 1 al 5, ¿cuánto crees que la música mejora tu concentración?'
COL_GENERO      = '¿Qué género escuchas principalmente al estudiar?'
COL_REND        = 'Notas que al estudiar con música cambia tu rendimiento académico'
COL_DISTR       = 'En una escala del 1 al 5. Cuando estudias con música desde tu teléfono, ¿qué tan frecuente es que termines distrayéndote revisando notificaciones, redes sociales o mensajes?'
COL_RITUAL      = '¿Utilizas la música como una señal para "entrar en modo estudio" (es decir, como un ritual para empezar)?'
COL_VOL         = '¿Qué nivel de volumen usas al estudiar con música?'
COL_AUD         = 'Cuando estudias, ¿escuchas música con o sin audífonos?'
COL_LETRA       = '¿La música que escuchas tiene letra (voz)?'
COL_DEJO        = '¿Has dejado de estudiar por distraerte con la música?'
COL_TIEMPO_ELIGE = '¿Cuánto tiempo dedicas a elegir la música antes de comenzar realmente a estudiar?'
COL_ANSIEDAD    = 'Sientes menos ansiedad cuando escuchas música'
COL_BUSCA       = '¿Qué buscas al poner música para estudiar?'
COL_PERCEP_T    = '¿Cómo influye la música en tu percepción del paso del tiempo mientras estudias?'
COL_AYUDA_CONC  = 'La música. ¿te ayuda a concentrarte más en tus tareas?'
COL_AISLAMIENTO = 'En una escala del 1 al 5. ¿Estudiar con música usando audífonos te ayuda a aislarte del ruido del entorno?'
COL_CANTAR      = '¿Sueles cantar mientras estudias con música?'
COL_REACCION    = 'Si una canción que te gusta mucho empieza a sonar, ¿Cuál es tu reacción inmediata?'
COL_ESTRATEGIA  = 'Cuando escuchas música desde tu teléfono para estudiar, ¿usas alguna estrategia para evitar distracciones digitales?'
COL_MEJORA_COM  = 'Consideras que la música mejora tu rendimiento o te hace sentir más cómodo'
COL_EDAD        = '¿Cuál es tu edad?'
COL_TIEMPO_EST  = 'Regularmente, ¿Cuánto tiempo al día, pasas estudiando?'
COL_DIFICULTAD  = 'La música te ayuda más cuando el tema o materia es'

# --- Encabezado del reporte ---
print(f"\n{Color.BG_BLUE}{Color.WHITE}{Color.BOLD} PROYECTO FINAL: MINERÍA DE DATOS — MÚSICA Y ESTUDIO {Color.RESET}")
print(f"  {Color.CYAN}Objetivo: Determinar si escuchar música mientras se estudia")
print(f"  mejora la concentración y el rendimiento académico.{Color.RESET}")
print(f"  {Color.DIM}Total de respuestas: {Color.WHITE}{Color.BOLD}{len(df)}{Color.RESET}{Color.DIM} estudiantes{Color.RESET}")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 0: DEFINICIÓN DE VARIABLES INDEPENDIENTES Y DEPENDIENTES
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("Definición de Variables Independientes y Dependientes", Color.BLUE)

print(f"""
  {Color.CYAN}{Color.BOLD}▐ VARIABLES INDEPENDIENTES{Color.RESET} {Color.DIM}(factores que el estudiante controla o trae consigo){Color.RESET}

    {Color.WHITE}{Color.BOLD}1. Frecuencia de escuchar música al estudiar{Color.RESET}
       {Color.DIM}Columna: "{COL_FREQ}"
       Justificación: Es un hábito personal previo al resultado académico;
       el estudiante decide con qué frecuencia escucha música.{Color.RESET}

    {Color.WHITE}{Color.BOLD}2. Género musical que escucha al estudiar{Color.RESET}
       {Color.DIM}Columna: "{COL_GENERO}"
       Justificación: Elección personal que podría influir de forma
       diferenciada en la concentración o distracción.{Color.RESET}

    {Color.WHITE}{Color.BOLD}3. Nivel de volumen{Color.RESET}
       {Color.DIM}Columna: "{COL_VOL}"
       Justificación: Factor directamente controlable por el estudiante
       que puede afectar la capacidad de concentración.{Color.RESET}

    {Color.WHITE}{Color.BOLD}4. Uso de audífonos vs bocinas{Color.RESET}
       {Color.DIM}Columna: "{COL_AUD}"
       Justificación: El método de escucha determina el nivel de
       aislamiento acústico del entorno.{Color.RESET}

    {Color.WHITE}{Color.BOLD}5. Presencia de letra (voz) en la música{Color.RESET}
       {Color.DIM}Columna: "{COL_LETRA}"
       Justificación: La letra puede competir con los procesos cognitivos
       de lectura y comprensión del estudiante.{Color.RESET}
""")

print(f"""  {Color.YELLOW}{Color.BOLD}▐ VARIABLES DEPENDIENTES{Color.RESET} {Color.DIM}(resultados o efectos medidos){Color.RESET}

    {Color.WHITE}{Color.BOLD}1. Mejora de concentración (escala 1-5){Color.RESET}
       {Color.DIM}Columna: "{COL_CONC}"
       Justificación: Es el resultado percibido que depende de las
       condiciones en que se escucha la música.{Color.RESET}

    {Color.WHITE}{Color.BOLD}2. Cambio en rendimiento académico{Color.RESET}
       {Color.DIM}Columna: "{COL_REND}"
       Justificación: Efecto académico percibido; es la consecuencia
       directa del hábito musical al estudiar.{Color.RESET}

    {Color.WHITE}{Color.BOLD}3. Distracción por notificaciones del teléfono (escala 1-5){Color.RESET}
       {Color.DIM}Columna: "{COL_DISTR}"
       Justificación: Efecto colateral de usar el teléfono como
       reproductor de música durante el estudio.{Color.RESET}

    {Color.WHITE}{Color.BOLD}4. Ha dejado de estudiar por distraerse con la música{Color.RESET}
       {Color.DIM}Columna: "{COL_DEJO}"
       Justificación: Resultado conductual negativo; consecuencia
       directa de la exposición a la música.{Color.RESET}

    {Color.WHITE}{Color.BOLD}5. Percepción del paso del tiempo{Color.RESET}
       {Color.DIM}Columna: "{COL_PERCEP_T}"
       Justificación: Efecto psicológico que resulta de la experiencia
       de estudiar acompañado de música.{Color.RESET}
""")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: ANÁLISIS DESCRIPTIVO
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("1. Análisis Descriptivo", Color.CYAN)

# --- 1.1 Porcentaje que escucha música ---
subtitulo("1.1", "¿Qué porcentaje de estudiantes escucha música mientras estudia?")
freq_musica = df[COL_FREQ].value_counts(normalize=True) * 100
max_val = freq_musica.max()
for cat, pct in freq_musica.items():
    barra = barra_horizontal(pct, max_val, ancho=25)
    c = Color.GREEN if pct > 30 else (Color.YELLOW if pct > 15 else Color.RED)
    print(f"    {barra} {c}{pct:5.1f}%{Color.RESET}  {cat}")

pct_escucha = 100 - freq_musica.get('Nunca', 0)
print(f"\n    {Color.BG_GREEN}{Color.WHITE}{Color.BOLD}  ▶ {pct_escucha:.1f}% de los estudiantes escucha música al estudiar  {Color.RESET}")
separador()

# --- 1.2 Actividades académicas con música ---
subtitulo("1.2", "¿Qué tipo de actividades académicas se realizan más con música?")
top_act = df[COL_ACT].value_counts().head(5)
max_act = top_act.max()
colores_act = [Color.GREEN, Color.CYAN, Color.YELLOW, Color.MAGENTA, Color.BLUE]
medallas = ["🥇", "🥈", "🥉", "  ", "  "]
for i, (act, cnt) in enumerate(top_act.items()):
    color = colores_act[i % len(colores_act)]
    barra = barra_horizontal(cnt, max_act, ancho=25, color=color)
    med = medallas[i] if i < len(medallas) else "  "
    print(f"    {med} {barra} {color}{Color.BOLD}{cnt:3d}{Color.RESET}  {act}")
separador()

# --- 1.3 Nivel educativo ---
subtitulo("1.3", "¿Qué nivel educativo escucha música con mayor frecuencia?")
escuchan = df[df[COL_FREQ].isin(['Siempre', 'Casi siempre'])]
nivel_freq = escuchan[COL_EDU].value_counts()
max_niv = nivel_freq.max()
for niv, cnt in nivel_freq.items():
    barra = barra_horizontal(cnt, max_niv, ancho=25, color=Color.MAGENTA)
    print(f"    {barra} {Color.WHITE}{Color.BOLD}{cnt:3d}{Color.RESET}  {niv}")

if len(nivel_freq) > 0:
    conclusion(f'El nivel "{nivel_freq.index[0]}" es el que más escucha música con alta frecuencia.')
separador()

# --- 1.4 Distracción por notificaciones del teléfono ---
subtitulo("1.4", "¿Qué tan frecuente es la distracción por notificaciones al usar música desde el teléfono?")
dist_vals = df[COL_DISTR].value_counts().sort_index()
max_d = dist_vals.max()
escala_desc = {1: "Nunca", 2: "Casi nunca", 3: "A veces", 4: "Frecuentemente", 5: "Siempre"}
for nivel, cnt in dist_vals.items():
    desc = escala_desc.get(nivel, "")
    c = Color.GREEN if nivel <= 2 else (Color.YELLOW if nivel == 3 else Color.RED)
    barra = barra_horizontal(cnt, max_d, ancho=25, color=c)
    print(f"    {barra} {c}{Color.BOLD}{cnt:3d}{Color.RESET}  Nivel {nivel}/5 ({desc})")

media_distr = df[COL_DISTR].mean()
pct_alta_distr = (df[COL_DISTR] >= 4).mean() * 100
print(f"\n    {Color.DIM}Media de distracción: {Color.WHITE}{Color.BOLD}{media_distr:.2f}/5{Color.RESET}")
print(f"    {Color.DIM}Estudiantes con alta distracción (≥4): {Color.RED}{Color.BOLD}{pct_alta_distr:.1f}%{Color.RESET}")
conclusion(f"La distracción por notificaciones es {'frecuente' if media_distr >= 3 else 'moderada'} (media {media_distr:.2f}/5). El {pct_alta_distr:.1f}% reporta alta distracción.")
separador()

# --- 1.5 Música como ritual de estudio ---
subtitulo("1.5", '¿Qué tan común es utilizar música como "ritual de estudio"?')
ritual_vals = df[COL_RITUAL].value_counts(normalize=True) * 100
max_r = ritual_vals.max()
ritual_short = {
    'Sí, ponerme los audífonos o la música es lo que le indica a mi cerebro que es hora de trabajar': 'Sí, es mi señal para trabajar',
    'No, la pongo simplemente para que me acompañe una vez que ya empecé': 'No, solo acompaña',
    'A veces, solo cuando me cuesta mucho trabajo iniciar la tarea': 'A veces, cuando cuesta iniciar'
}
colores_ritual = [Color.GREEN, Color.YELLOW, Color.CYAN]
for i, (resp, pct) in enumerate(ritual_vals.items()):
    short = ritual_short.get(resp, resp[:50])
    c = colores_ritual[i % len(colores_ritual)]
    barra = barra_horizontal(pct, max_r, ancho=25, color=c)
    print(f"    {barra} {c}{pct:5.1f}%{Color.RESET}  {short}")

pct_ritual_si = ritual_vals.get(
    'Sí, ponerme los audífonos o la música es lo que le indica a mi cerebro que es hora de trabajar', 0)
conclusion(f"{pct_ritual_si:.1f}% de los estudiantes usa la música como ritual para entrar en modo estudio.")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: ANÁLISIS DE CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("2. Análisis de Correlación", Color.YELLOW)

# --- 2.1 Relación música → concentración ---
subtitulo("2.1", "¿Existe relación entre escuchar música y mejorar la concentración?")
contingencia_conc = pd.crosstab(df[COL_FREQ], df[COL_CONC])
chi2, p, dof, ex = stats.chi2_contingency(contingencia_conc)
print(f"    Prueba: Chi-cuadrado de independencia")
print(f"    Chi² = {Color.WHITE}{Color.BOLD}{chi2:.4f}{Color.RESET}  |  gl = {dof}  |  p-valor = {p_valor_color(p)}")
print(f"    → {resultado_sig(p)}")
separador()

# --- 2.2 Relación tipo de música → rendimiento ---
subtitulo("2.2", "¿Existe relación entre el tipo de música y el rendimiento académico percibido?")
contingencia_rend = pd.crosstab(df[COL_GENERO], df[COL_REND])
chi2_r, p_r, dof_r, ex_r = stats.chi2_contingency(contingencia_rend)
print(f"    Prueba: Chi-cuadrado de independencia")
print(f"    Chi² = {Color.WHITE}{Color.BOLD}{chi2_r:.4f}{Color.RESET}  |  gl = {dof_r}  |  p-valor = {p_valor_color(p_r)}")
print(f"    → {resultado_sig(p_r)}")
separador()

# --- 2.3 Relación distracción teléfono ↔ tiempo eligiendo música ---
subtitulo("2.3", "Si se distrae con el teléfono, ¿también dedica más tiempo a elegir música?")
tiempo_map = {
    'Nada, uso siempre la misma lista predeterminada': 0,
    'Menos de 5 minutos': 1,
    'Entre 5 y 15 minutos': 2,
    'Más de 15 minutos (a veces procrastino eligiendo la música)': 3
}
df['_tiempo_num'] = df[COL_TIEMPO_ELIGE].map(tiempo_map)
valid = df['_tiempo_num'].notna() & df[COL_DISTR].notna()
rho, p_sp = stats.spearmanr(df.loc[valid, COL_DISTR], df.loc[valid, '_tiempo_num'])

print(f"    Prueba: Correlación de Spearman (variables ordinales)")
print(f"    ρ (rho) = {Color.WHITE}{Color.BOLD}{rho:.4f}{Color.RESET}  |  p-valor = {p_valor_color(p_sp)}")
print(f"    → {resultado_sig(p_sp)}")
if rho > 0:
    print(f"    {Color.DIM}Dirección: Correlación positiva — a mayor distracción, más tiempo eligiendo música.{Color.RESET}")
else:
    print(f"    {Color.DIM}Dirección: Correlación negativa — a mayor distracción, menos tiempo eligiendo música.{Color.RESET}")
separador()

# --- 2.4 Variables que explican mejor el rendimiento ---
subtitulo("2.4", "¿Qué variables explican mejor la mejora del rendimiento académico?")

feature_cols_rend = {
    'Frecuencia música': COL_FREQ,
    'Volumen': COL_VOL,
    'Audífonos': COL_AUD,
    'Letra/voz': COL_LETRA,
    'Busca al estudiar': COL_BUSCA,
    'Reduce ansiedad': COL_ANSIEDAD,
    'Canta al estudiar': COL_CANTAR,
    'Dificultad materia': COL_DIFICULTAD,
    'Tiempo estudio/día': COL_TIEMPO_EST,
    'Concentración (1-5)': COL_CONC,
    'Distracción tel (1-5)': COL_DISTR,
    'Aislamiento (1-5)': COL_AISLAMIENTO,
}

df_rf = df[list(feature_cols_rend.values()) + [COL_REND]].dropna().copy()
y_rf = (df_rf[COL_REND] == 'Si, mejora').astype(int)

for col in feature_cols_rend.values():
    le = LabelEncoder()
    df_rf[col] = le.fit_transform(df_rf[col].astype(str))

X_rf = df_rf[list(feature_cols_rend.values())]
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)

importances = pd.Series(
    rf.feature_importances_,
    index=list(feature_cols_rend.keys())
).sort_values(ascending=False)

print(f"\n    {Color.WHITE}{Color.BOLD}Importancia de las variables (Random Forest):{Color.RESET}")
print(f"    {Color.DIM}Modelo entrenado para predecir: 'rendimiento mejora'{Color.RESET}\n")
max_imp = importances.max()
for nombre, imp in importances.items():
    barra = barra_horizontal(imp, max_imp, ancho=25, color=Color.CYAN)
    c = Color.GREEN if imp > importances.mean() else Color.DIM
    print(f"    {barra} {c}{Color.BOLD}{imp:.4f}{Color.RESET}  {nombre}")

top3 = importances.head(3).index.tolist()
conclusion(f"Las 3 variables más influyentes en el rendimiento son: {', '.join(top3)}.")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CLASIFICACIÓN (Machine Learning Supervisado)
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("3. Clasificación — Machine Learning Supervisado", Color.MAGENTA)

# --- 3.1 Predecir distracción con el teléfono ---
subtitulo("3.1", "¿Es posible predecir si un estudiante se distraerá con el teléfono mientras escucha música?")

clf_features_distr = {
    'Frecuencia música': COL_FREQ,
    'Volumen': COL_VOL,
    'Audífonos': COL_AUD,
    'Letra/voz': COL_LETRA,
    'Estrategia anti-distracción': COL_ESTRATEGIA,
    'Canta al estudiar': COL_CANTAR,
    'Tiempo eligiendo música': COL_TIEMPO_ELIGE,
}

df_clf1 = df[list(clf_features_distr.values()) + [COL_DISTR]].dropna().copy()
y_distr = (df_clf1[COL_DISTR] >= 4).astype(int)  # 1 = se distrae frecuentemente

for col in clf_features_distr.values():
    le = LabelEncoder()
    df_clf1[col] = le.fit_transform(df_clf1[col].astype(str))

X_distr = df_clf1[list(clf_features_distr.values())]
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_distr, y_distr, test_size=0.3, random_state=42)

lr_distr = LogisticRegression(max_iter=1000, random_state=42)
lr_distr.fit(X_train1, y_train1)
y_pred1 = lr_distr.predict(X_test1)
acc1 = accuracy_score(y_test1, y_pred1)

print(f"    {Color.DIM}Modelo: Regresión Logística{Color.RESET}")
print(f"    {Color.DIM}Variable objetivo: Distracción ≥ 4/5 → 'Se distrae'{Color.RESET}")
print(f"    {Color.DIM}Datos: {len(X_train1)} entrenamiento | {len(X_test1)} prueba{Color.RESET}")
print(f"\n    Accuracy: {Color.WHITE}{Color.BOLD}{acc1:.2%}{Color.RESET}")

coefs1 = pd.Series(
    np.abs(lr_distr.coef_[0]),
    index=list(clf_features_distr.keys())
).sort_values(ascending=False)
print(f"\n    {Color.WHITE}{Color.BOLD}Variables más influyentes (|coeficientes|):{Color.RESET}\n")
max_c1 = coefs1.max()
for nombre, coef in coefs1.items():
    barra = barra_horizontal(coef, max_c1, ancho=20, color=Color.MAGENTA)
    print(f"    {barra} {Color.DIM}{coef:.4f}{Color.RESET}  {nombre}")

conclusion(f"El modelo predice si un estudiante se distraerá con el teléfono con {acc1:.0%} de precisión.")
separador()

# --- 3.2 Predecir si cree que la música mejora su rendimiento ---
subtitulo("3.2", "¿Se puede identificar qué estudiantes consideran que la música mejora su rendimiento?")

clf_features_rend = {
    'Frecuencia música': COL_FREQ,
    'Volumen': COL_VOL,
    'Audífonos': COL_AUD,
    'Letra/voz': COL_LETRA,
    'Reduce ansiedad': COL_ANSIEDAD,
    'Busca al estudiar': COL_BUSCA,
    'Dificultad materia': COL_DIFICULTAD,
    'Tiempo estudio/día': COL_TIEMPO_EST,
}

df_clf2 = df[list(clf_features_rend.values()) + [COL_REND]].dropna().copy()
y_rend = (df_clf2[COL_REND] == 'Si, mejora').astype(int)

for col in clf_features_rend.values():
    le = LabelEncoder()
    df_clf2[col] = le.fit_transform(df_clf2[col].astype(str))

X_rend = df_clf2[list(clf_features_rend.values())]
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_rend, y_rend, test_size=0.3, random_state=42)

lr_rend = LogisticRegression(max_iter=1000, random_state=42)
lr_rend.fit(X_train2, y_train2)
y_pred2 = lr_rend.predict(X_test2)
acc2 = accuracy_score(y_test2, y_pred2)

print(f"    {Color.DIM}Modelo: Regresión Logística{Color.RESET}")
print(f"    {Color.DIM}Variable objetivo: Rendimiento = 'Si, mejora' → 1{Color.RESET}")
print(f"    {Color.DIM}Datos: {len(X_train2)} entrenamiento | {len(X_test2)} prueba{Color.RESET}")
print(f"\n    Accuracy: {Color.WHITE}{Color.BOLD}{acc2:.2%}{Color.RESET}")

coefs2 = pd.Series(
    np.abs(lr_rend.coef_[0]),
    index=list(clf_features_rend.keys())
).sort_values(ascending=False)
print(f"\n    {Color.WHITE}{Color.BOLD}Variables más influyentes (|coeficientes|):{Color.RESET}\n")
max_c2 = coefs2.max()
for nombre, coef in coefs2.items():
    barra = barra_horizontal(coef, max_c2, ancho=20, color=Color.CYAN)
    print(f"    {barra} {Color.DIM}{coef:.4f}{Color.RESET}  {nombre}")

conclusion(f"El modelo identifica estudiantes que perciben mejora en su rendimiento con {acc2:.0%} de precisión.")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: CLUSTERING — IDENTIFICACIÓN DE PERFILES
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("4. Clustering — Identificación de Perfiles (K-Means)", Color.MAGENTA)

# Columnas para el clustering (set amplio de variables)
cluster_cols = {
    'Frecuencia música': COL_FREQ,
    'Concentración (1-5)': COL_CONC,
    'Distracción tel (1-5)': COL_DISTR,
    'Dejó de estudiar': COL_DEJO,
    'Busca al estudiar': COL_BUSCA,
    'Volumen': COL_VOL,
    'Canta al estudiar': COL_CANTAR,
    'Reacción canción favorita': COL_REACCION,
    'Ritual de estudio': COL_RITUAL,
    'Estrategia anti-distracción': COL_ESTRATEGIA,
}

# Preparar datos para clustering
df_cl = df[list(cluster_cols.values())].copy()
for col in cluster_cols.values():
    if df_cl[col].dtype in ['int64', 'float64']:
        df_cl[col] = df_cl[col].fillna(df_cl[col].median())
    else:
        df_cl[col] = df_cl[col].fillna('No responde')

df_cl_orig = df_cl.copy()  # Conservar datos originales para interpretación

# Codificar para K-Means
df_encoded_cl = pd.DataFrame()
for col in cluster_cols.values():
    le = LabelEncoder()
    df_encoded_cl[col] = le.fit_transform(df_cl[col].astype(str))

# Aplicar K-Means con 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cl_orig['Perfil'] = kmeans.fit_predict(df_encoded_cl)

print(f"\n  Se agruparon a los estudiantes en {Color.BOLD}{Color.WHITE}4 perfiles{Color.RESET}"
      f" usando {Color.BOLD}{len(cluster_cols)}{Color.RESET} variables.")
separador()

# --- Interpretar cada cluster ---
# Nombres de perfiles solicitados
nombres_perfil = {
    'dependiente': '🎧 Altamente dependiente de la música',
    'distraido':   '📱 Fácilmente distraído',
    'relajante':   '🎵 Usa música relajante',
    'multitarea':  '⚡ Estudiante multitarea',
}

def puntuar_cluster(cl_data):
    """Asigna puntajes a cada arquetipo de perfil basándose en los datos del cluster."""
    scores = {}
    n = len(cl_data)
    if n == 0:
        return {'dependiente': 0, 'distraido': 0, 'relajante': 0, 'multitarea': 0}

    # Dependiente: alta frecuencia + usa ritual
    freq_alta = cl_data[COL_FREQ].isin(['Siempre', 'Casi siempre']).sum() / n
    ritual = cl_data[COL_RITUAL].astype(str).str.contains('Sí, ponerme', na=False).sum() / n
    scores['dependiente'] = freq_alta * 2 + ritual * 2

    # Distraído: alta distracción + ha dejado de estudiar + canta
    alta_distr = (cl_data[COL_DISTR] >= 4).sum() / n
    dejo = cl_data[COL_DEJO].isin(['Siempre', 'Casi siempre', 'Algunas veces']).sum() / n
    canta = (cl_data[COL_CANTAR] == 'Sí').sum() / n
    scores['distraido'] = alta_distr * 2 + dejo * 2 + canta

    # Relajante: busca calma + volumen bajo
    calma = cl_data[COL_BUSCA].astype(str).str.contains('Calma', na=False).sum() / n
    vol_bajo = (cl_data[COL_VOL] == 'Bajo').sum() / n
    scores['relajante'] = calma * 2 + vol_bajo * 2

    # Multitarea: sigue trabajando con canción favorita + sin estrategia (flujo natural)
    sigue = cl_data[COL_REACCION].astype(str).str.contains('sigo trabajando', na=False).sum() / n
    ignora = cl_data[COL_REACCION].astype(str).str.contains('ignoro', na=False).sum() / n
    scores['multitarea'] = sigue * 2 + ignora * 2

    return scores

# Asignar nombres a clusters
asignaciones = {}
usados = set()
cluster_scores = {}

for perfil_id in range(4):
    cl_data = df_cl_orig[df_cl_orig['Perfil'] == perfil_id]
    cluster_scores[perfil_id] = puntuar_cluster(cl_data)

# Asignar perfiles de mayor a menor puntaje para evitar conflictos
all_assignments = []
for perfil_id in range(4):
    for profile_key, score in sorted(cluster_scores[perfil_id].items(), key=lambda x: -x[1]):
        all_assignments.append((score, perfil_id, profile_key))

all_assignments.sort(key=lambda x: -x[0])
for score, perfil_id, profile_key in all_assignments:
    if perfil_id not in asignaciones and profile_key not in usados:
        asignaciones[perfil_id] = profile_key
        usados.add(profile_key)

# Mostrar perfiles con características
colores_perfil = [Color.CYAN, Color.GREEN, Color.YELLOW, Color.MAGENTA]
perfiles_conteo = df_cl_orig['Perfil'].value_counts().sort_index()
max_perfil = perfiles_conteo.max()

print(f"\n  {Color.BOLD}Distribución y características de los perfiles:{Color.RESET}\n")

for perfil_id in range(4):
    cl_data = df_cl_orig[df_cl_orig['Perfil'] == perfil_id]
    conteo = len(cl_data)
    porcentaje = (conteo / len(df_cl_orig)) * 100
    color = colores_perfil[perfil_id % len(colores_perfil)]
    nombre = nombres_perfil.get(asignaciones.get(perfil_id, 'dependiente'), f'Perfil {perfil_id}')

    barra = barra_horizontal(conteo, max_perfil, ancho=20, color=color)

    print(f"    {color}{Color.BOLD}{'─' * 60}{Color.RESET}")
    print(f"    {color}{Color.BOLD}{nombre}{Color.RESET}")
    print(f"    {barra}  {Color.WHITE}{Color.BOLD}{conteo}{Color.RESET} estudiantes ({color}{porcentaje:.1f}%{Color.RESET})")

    # Características del cluster
    freq_mode = cl_data[COL_FREQ].mode().iloc[0] if len(cl_data[COL_FREQ].mode()) > 0 else 'N/A'
    conc_mean = cl_data[COL_CONC].mean()
    distr_mean = cl_data[COL_DISTR].mean()
    busca_mode = cl_data[COL_BUSCA].mode().iloc[0] if len(cl_data[COL_BUSCA].mode()) > 0 else 'N/A'
    ritual_pct = cl_data[COL_RITUAL].astype(str).str.contains('Sí, ponerme', na=False).mean() * 100
    vol_mode = cl_data[COL_VOL].mode().iloc[0] if len(cl_data[COL_VOL].mode()) > 0 else 'N/A'

    print(f"      {Color.DIM}• Frecuencia música:  {Color.WHITE}{freq_mode}{Color.RESET}")
    print(f"      {Color.DIM}• Concentración media: {Color.WHITE}{conc_mean:.1f}/5{Color.RESET}")
    print(f"      {Color.DIM}• Distracción media:   {Color.WHITE}{distr_mean:.1f}/5{Color.RESET}")
    print(f"      {Color.DIM}• Busca al estudiar:   {Color.WHITE}{busca_mode}{Color.RESET}")
    print(f"      {Color.DIM}• Volumen preferido:   {Color.WHITE}{vol_mode}{Color.RESET}")
    print(f"      {Color.DIM}• Usa ritual de estudio: {Color.WHITE}{ritual_pct:.0f}%{Color.RESET}")

separador()
total = perfiles_conteo.sum()
print(f"\n  {Color.DIM}Total de estudiantes analizados: {Color.WHITE}{Color.BOLD}{total}{Color.RESET}")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: REGULACIÓN EMOCIONAL
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("5. ¿La música funciona como regulación emocional?", Color.CYAN)

# --- 5.1 Reducción de ansiedad ---
subtitulo("5.1", "¿Escuchar música reduce la ansiedad al estudiar?")
ansiedad_vals = df[COL_ANSIEDAD].value_counts(normalize=True) * 100
max_a = ansiedad_vals.max()
colores_ans = {'Sí': Color.GREEN, 'No': Color.RED, 'Tal vez': Color.YELLOW, 'No sé': Color.DIM}
for resp, pct in ansiedad_vals.items():
    c = colores_ans.get(resp, Color.WHITE)
    barra = barra_horizontal(pct, max_a, ancho=25, color=c)
    print(f"    {barra} {c}{pct:5.1f}%{Color.RESET}  {resp}")

pct_si_ansiedad = ansiedad_vals.get('Sí', 0)
conclusion(f"{pct_si_ansiedad:.1f}% de los estudiantes siente menos ansiedad cuando escucha música.")
separador()

# --- 5.2 ¿Qué buscan al poner música? ---
subtitulo("5.2", "¿Qué buscan los estudiantes al poner música para estudiar?")
busca_vals = df[COL_BUSCA].value_counts()
max_b = busca_vals.max()
colores_busca = [Color.CYAN, Color.GREEN, Color.YELLOW, Color.MAGENTA, Color.BLUE, Color.RED]
for i, (motivo, cnt) in enumerate(busca_vals.head(6).items()):
    c = colores_busca[i % len(colores_busca)]
    pct = (cnt / len(df)) * 100
    barra = barra_horizontal(cnt, max_b, ancho=25, color=c)
    print(f"    {barra} {c}{Color.BOLD}{cnt:3d}{Color.RESET} ({pct:.1f}%)  {motivo}")

# Calcular porcentajes de motivaciones emocionales
total_resp = len(df)
busca_calma_n = df[COL_BUSCA].astype(str).str.contains('Calma', na=False).sum()
busca_motiv_n = df[COL_BUSCA].astype(str).str.contains('Motivación', na=False).sum()
busca_conc_n = df[COL_BUSCA].astype(str).str.contains('Concentración', na=False).sum()
print(f"\n    {Color.DIM}Motivaciones emocionales:{Color.RESET}")
print(f"      Calma:         {Color.CYAN}{(busca_calma_n / total_resp * 100):.1f}%{Color.RESET}")
print(f"      Motivación:    {Color.GREEN}{(busca_motiv_n / total_resp * 100):.1f}%{Color.RESET}")
print(f"      Concentración: {Color.YELLOW}{(busca_conc_n / total_resp * 100):.1f}%{Color.RESET}")
separador()

# --- 5.3 Relación: buscar calma → menor ansiedad ---
subtitulo("5.3", 'Relación estadística: ¿Buscar "calma" se asocia con sentir menos ansiedad?')
df['_busca_calma'] = df[COL_BUSCA].astype(str).str.contains('Calma', na=False).astype(int)
df['_menos_ansiedad'] = (df[COL_ANSIEDAD] == 'Sí').astype(int)
contingencia_emo = pd.crosstab(df['_busca_calma'], df['_menos_ansiedad'])
chi2_emo, p_emo, dof_emo, ex_emo = stats.chi2_contingency(contingencia_emo)

print(f"    Prueba: Chi-cuadrado de independencia")
print(f"    Chi² = {Color.WHITE}{Color.BOLD}{chi2_emo:.4f}{Color.RESET}  |  gl = {dof_emo}  |  p-valor = {p_valor_color(p_emo)}")
print(f"    → {resultado_sig(p_emo)}")
separador()

# --- 5.4 Percepción del paso del tiempo ---
subtitulo("5.4", "¿Cómo influye la música en la percepción del tiempo?")
percep_vals = df[COL_PERCEP_T].value_counts(normalize=True) * 100
max_p = percep_vals.max()
percep_short = {
    'Siento que el tiempo pasa más rápido.': '⏩ El tiempo pasa más rápido',
    'No noto ninguna diferencia en mi percepción del tiempo.': '⏸ Sin diferencia',
    'Siento que el tiempo pasa más lento.': '⏪ El tiempo pasa más lento',
}
colores_p = [Color.GREEN, Color.YELLOW, Color.RED]
for i, (resp, pct) in enumerate(percep_vals.items()):
    short = percep_short.get(resp, resp[:50])
    c = colores_p[i % len(colores_p)]
    barra = barra_horizontal(pct, max_p, ancho=25, color=c)
    print(f"    {barra} {c}{pct:5.1f}%{Color.RESET}  {short}")

pct_rapido = percep_vals.get('Siento que el tiempo pasa más rápido.', 0)
separador()

# --- Conclusión de regulación emocional ---
print(f"\n  {Color.BG_CYAN}{Color.WHITE}{Color.BOLD}  ▶ CONCLUSIÓN: Regulación Emocional  {Color.RESET}")
print(f"""
    La evidencia sugiere que la música {Color.GREEN}{Color.BOLD}SÍ funciona como herramienta
    de regulación emocional{Color.RESET} durante el estudio:

    {Color.WHITE}•{Color.RESET} El {Color.GREEN}{Color.BOLD}{pct_si_ansiedad:.1f}%{Color.RESET} reporta menor ansiedad con música.
    {Color.WHITE}•{Color.RESET} Los estudiantes buscan activamente estados emocionales
      como calma ({busca_calma_n / total_resp * 100:.1f}%) y motivación ({busca_motiv_n / total_resp * 100:.1f}%).
    {Color.WHITE}•{Color.RESET} El {Color.GREEN}{Color.BOLD}{pct_rapido:.1f}%{Color.RESET} percibe que el tiempo pasa más rápido,
      indicando un estado de "flujo" o absorción en la tarea.
""")


# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
titulo_seccion("Resumen Final del Análisis", Color.BG_BLUE)

print(f"""
  {Color.WHITE}{Color.BOLD}Se analizaron {len(df)} respuestas de estudiantes.{Color.RESET}

  {Color.CYAN}{Color.BOLD}Hallazgos principales:{Color.RESET}

    {Color.GREEN}✓{Color.RESET} El {Color.BOLD}{pct_escucha:.1f}%{Color.RESET} de los estudiantes escucha música al estudiar.
    {Color.GREEN}✓{Color.RESET} El nivel educativo con mayor frecuencia de uso es {Color.BOLD}{nivel_freq.index[0] if len(nivel_freq) > 0 else 'N/A'}{Color.RESET}.
    {Color.GREEN}✓{Color.RESET} La distracción por notificaciones es {'alta' if media_distr >= 3 else 'moderada'} (media {media_distr:.2f}/5).
    {Color.GREEN}✓{Color.RESET} El {pct_ritual_si:.1f}% usa la música como ritual para iniciar el estudio.

  {Color.YELLOW}{Color.BOLD}Correlaciones:{Color.RESET}
    {'✓' if p < 0.05 else '✗'} Música ↔ Concentración: p = {p:.4f}
    {'✓' if p_r < 0.05 else '✗'} Género musical ↔ Rendimiento: p = {p_r:.4f}
    {'✓' if p_sp < 0.05 else '✗'} Distracción ↔ Tiempo eligiendo música: ρ = {rho:.4f}, p = {p_sp:.4f}

  {Color.MAGENTA}{Color.BOLD}Machine Learning:{Color.RESET}
    • Predicción de distracción con teléfono: {acc1:.0%} accuracy
    • Predicción de mejora en rendimiento: {acc2:.0%} accuracy
    • Variables top para rendimiento: {', '.join(top3)}
    • 4 perfiles de estudiantes identificados por K-Means

  {Color.CYAN}{Color.BOLD}Regulación emocional:{Color.RESET}
    • {pct_si_ansiedad:.1f}% siente menos ansiedad con música
    • {pct_rapido:.1f}% percibe que el tiempo pasa más rápido
    • La música SÍ funciona como herramienta de regulación emocional
""")

# Limpiar columnas temporales
df.drop(columns=['_tiempo_num', '_busca_calma', '_menos_ansiedad'], inplace=True, errors='ignore')

print(f"{Color.DIM}{'═' * 70}{Color.RESET}")
print(f"{Color.DIM}  Fin del reporte. Proyecto Final de Minería de Datos.{Color.RESET}")
print()