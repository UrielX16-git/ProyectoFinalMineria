# Minería de Datos: Música y Estudio

Proyecto final de **Minería de Datos** enfocado en analizar y determinar cómo influye escuchar música mientras se estudia en la concentración, regulación emocional y rendimiento académico de los estudiantes.

---

## Objetivo
El objetivo principal es identificar patrones, relaciones estadísticas y perfiles de comportamiento a partir de un conjunto de datos obtenidos mediante encuestas a estudiantes.

---

## Estructura del Análisis (`app.py`)
El script realiza un análisis completo dividido en las siguientes secciones:

1. **Análisis Descriptivo:** Estadísticas básicas sobre hábitos de estudio con música, actividades donde más ayuda, volumen preferido, el uso de música como "ritual de estudio" y la distracción digital.
2. **Análisis de Correlación:** Pruebas estadísticas (Chi-cuadrado y Spearman) para verificar relaciones significativas (ej. Música ↔ Concentración) y análisis de importancia de variables con *Random Forest*.
3. **Machine Learning Supervisado (Clasificación):** Modelos de Regresión Logística entrenados para predecir si un estudiante sufrirá distracción digital alta y si percibirá mejoras en su rendimiento académico.
4. **Clustering (K-Means):** Segmentación que agrupa a los estudiantes en **4 perfiles** o arquetipos de comportamiento:
   - Altamente dependiente de la música
   - Fácilmente distraído
   - Usa música relajante
   - Estudiante multitarea
5. **Regulación Emocional:** Análisis estadístico de cómo influye la música en la reducción de ansiedad, la búsqueda de calma/motivación y la alteración de la percepción del tiempo.

---

## Cómo Ejecutar

Para correr todo el análisis y generar el reporte, ejecuta:

```bash
python app.py
```
