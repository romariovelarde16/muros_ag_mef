# Análisis y optimización de muros estructurales de concreto armado

Programa de la tesis «Análisis y optimización de muros estructurales de
concreto armado mediante elementos finitos y algoritmos genéticos en Ayacucho,
2025» (Universidad Nacional de San Cristóbal de Huamanga, Escuela Profesional
de Ingeniería Civil). Autor: Romario Velarde Chavez.

Para cada muro, el programa lee las fuerzas de ETABS, arma la demanda en
cualquier cota con un modelo de elementos finitos del alma, verifica el armado
y busca con un algoritmo genético el armado de menor peso de acero que cumple
la norma. Hay dos versiones con la misma estructura:

- `MuroRectangular_ACI/`: diseño con ACI 318-25 (capítulo 18, §18.10).
- `MuroRectangular_E060/`: diseño con la NTE E.060 (capítulo 21, §21.9).

Admite tres tipos de sección: I (rectangular), II (ala corta en medio) y III
(ala corta en el extremo).

## Requisitos

Python 3.12 con las librerías de `requirements.txt`:

    pip install -r requirements.txt

## Uso

Desde la carpeta de cada versión:

    python ejecutar.py edificacion_01/datos_P17.py     verificación de un muro
    python optimizar.py edificacion_01/datos_P17.py    optimización del armado

Sin argumento se usa el muro de `MURO_A_EJECUTAR`, al inicio de `ejecutar.py`.
La versión ACI tiene además dos estudios del modelo de elementos finitos, que
se corren igual: `convergencia_malla.py` y `estudio_mef.py`.

Los informes y figuras se escriben en `edificacion_0X/resultados/<MURO>/`.
`optimizar.py` reescribe `datos_<MURO>_optimo.py` junto a los datos del muro.

## Contenido

| Carpeta o archivo | Qué tiene |
|---|---|
| `mef.py` | Lectura de las fuerzas de ETABS, ejes, modelo de elementos finitos y demanda en cualquier cota |
| `diseno_aci.py`, `diseno_e060.py` | Sección, barras, flexocompresión, cortante, elementos de borde, detallado, cota de corte y peso del acero |
| `ejecutar.py` | Orden de los cálculos e informe de verificación |
| `figuras.py` | Figuras del informe |
| `optimizar.py` | Algoritmo genético en dos etapas y búsqueda local |
| `convergencia_malla.py`, `estudio_mef.py` | Estudios del modelo de elementos finitos (solo en la versión ACI) |
| `edificacion_01/` | Bloque I, Universidad Nacional Autónoma de Huanta: muros P17, P11 y P1 |
| `edificacion_02/` | Módulo 06, I.E. Los Libertadores (Ayacucho): muros M5, M10 y M2 |
| `edificacion_0X/datos_<MURO>.py` | Datos de cada muro con su armado de partida |
| `edificacion_0X/datos_<MURO>_optimo.py` | Datos con el armado óptimo hallado |
| `edificacion_0X/etabs/` | Tabla *Pier Forces* de ETABS y desplazamientos de los muros |

## Licencia

Licencia MIT (archivo `LICENSE`). El programa es una herramienta de
investigación: sus resultados no reemplazan la revisión de un ingeniero
responsable del diseño.
