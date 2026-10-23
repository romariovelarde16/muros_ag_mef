"""DATOS DE UN MURO. Es el único archivo que hay que editar.

Los cálculos están en la carpeta de arriba (mef.py, diseno_e060.py,
ejecutar.py) y no hace falta tocarlos. Para otro muro, copia este archivo en
la carpeta de su edificación con el nombre datos_<MURO>.py, cambia sus datos
y córrelo con:
    python ejecutar.py edificacion_0X/datos_<MURO>.py

UNIDADES: kN y m en fuerzas, momentos y longitudes. Los materiales (f'c, fy,
Es, Ec) en MPa, como los escribe la norma; f'c y fy se escriben en kgf/cm²
(FC_KGF_CM2, FY_KGF_CM2) y se pasan a MPa con 1 kgf/cm² = 0.0980665 MPa.
"""

import math

# -----------------------------------------------------------------------------
# 1. ETABS: ARCHIVO, MURO, PISOS Y COMBINACIONES
# -----------------------------------------------------------------------------
# La tabla "Pier Forces" exportada a Excel, en la carpeta etabs/ de la
# edificación, tal como la da ETABS (una fila de título, la cabecera y una
# fila de unidades). Puede venir en kN y kN-m o en tonf y tonf-m: el programa
# pasa todo a kN.
ARCHIVO_FUERZAS = "etabs/pier_forces.xlsx"
HOJA_FUERZAS = "Pier Forces"

MURO = "M10"                 # nombre del pier en ETABS

# Los pisos de ETABS, de ABAJO hacia ARRIBA, y la altura de cada entrepiso
# (en el mismo orden), m.
PISOS = ["Nivel sc", "Nivel E1", "Nivel E2", "Nivel E3", "Nivel E4"]
ALTURAS = [1.50, 3.15, 3.15, 3.15, 2.91]

# Las combinaciones de diseño de ETABS. Las que llevan sismo vienen como
# envolventes (filas Max y Min).
COMBINACIONES = ["Comb-1", "Comb-2", "Comb-3", "Comb-4", "Gravitacional"]

# Las combinaciones 0.9·CM ± CS (§9.2.3, ec. 9-5): de ellas sale la fuerza
# normal Nu del corte por fricción en las juntas de construcción (§21.9.8),
# la menor compresión de sus estados.
COMBINACIONES_DE_LA_JUNTA = ["Comb-3", "Comb-4"]

# -----------------------------------------------------------------------------
# 2. EJES: DE LOS DEL PIER EN ETABS A LOS DE ESTE PROGRAMA
# -----------------------------------------------------------------------------
# El programa dibuja SIEMPRE el muro igual (ver la figura de los tipos):
#   +X a lo largo del alma, del extremo IZQUIERDO al DERECHO
#      (en el tipo III, el ala queda en el extremo derecho);
#   +Y a través del espesor, HACIA DONDE SALE EL ALA (en el tipo I, hacia
#      cualquiera de las dos caras).
# AXIS_ANGLE es el ángulo del eje 2 del pier, medido desde +X global en
# sentido antihorario (en ETABS: tabla Pier Section Properties, columna
# AxisAngle). El eje 3 es el 2 girado 90° antihorario:
#     AxisAngle   0:  +2 = +X global   +3 = +Y global
#     AxisAngle  90:  +2 = +Y global   +3 = −X global
#     AxisAngle 180:  +2 = −X global   +3 = −Y global
#     AxisAngle 270:  +2 = −Y global   +3 = +X global
# EJE_X y EJE_Y dicen qué eje de ETABS ("+2", "-2", "+3" o "-3") apunta hacia
# +X y hacia +Y del programa. El informe escribe hacia dónde queda cada
# extremo y la planta lo dibuja: compáralo con la planta del edificio.
AXIS_ANGLE = 0.0             # ángulo del eje 2 del pier, grados
EJE_X = "-2"                 # eje de ETABS que apunta hacia +X del programa
EJE_Y = "-3"                 # eje de ETABS que apunta hacia +Y del programa

# Control de AXIS_ANGLE: dos casos estáticos de ETABS que empujan en +X y en
# +Y global. En la base, el cortante y el momento tienen que ir hacia donde
# empuja la carga; si no, el veredicto sale NO CUMPLE. Para no comprobarlo,
# CASOS_ESTATICOS = {}.
CASOS_ESTATICOS = {"X": "Sismo Estatico X", "Y": "Sismo Estatico Y"}

# -----------------------------------------------------------------------------
# 3. SISMO
# -----------------------------------------------------------------------------
# Desplazamiento de diseño δu del techo, en mm, en la dirección X del programa
# (el plano del muro): el inelástico de E.030, δu = 0.75·R·δelástico
# (estructura regular), con el sismo en la dirección global del alma. Solo lo
# usa el elemento de borde (§21.9.7.4, ec. 21-6). En Y no hace falta: el muro
# solo trabaja como muro en su plano.
DELTA_U_MM = 27.306

# Las combinaciones que llevan el sismo en la dirección global del alma (la X
# del programa). De ellas salen el eje neutro c del elemento de borde
# (§21.9.7.4) y la relación Mu/4Vu (§21.9.7.4(b) y §21.9.5.5).
COMBINACIONES_DEL_SISMO_EN_X = ["Comb-1", "Comb-3"]

# R0: el coeficiente básico de reducción de la E.030 (6 en muros
# estructurales, 7 en sistemas duales, 8 en pórticos). En el cortante de
# diseño, §21.9.5.5 no toma el cociente Mn/Mua mayor que 0.5·R0.
R0 = 6

# -----------------------------------------------------------------------------
# 4. GEOMETRÍA Y MATERIALES
# -----------------------------------------------------------------------------
# TIPO "I":   rectángulo.
# TIPO "II":  rectángulo con un ala corta en medio.
# TIPO "III": rectángulo con un ala corta en el extremo derecho.
# El programa admite SOLO estos tres tipos (otro = DATO INVÁLIDO).
TIPO = "II"
LW = 1.70                    # ℓm: largo total en X (en el tipo III, hasta la cara exterior del ala), m
ESPESOR = 0.30               # b: espesor del alma, m

# Solo tipos II y III (en el tipo I no se usan):
ALA_ANCHO = 0.30             # b': espesor del ala (medido en X), m
ALA_LARGO = 0.50             # l'w: largo del ala en Y, desde la cara EXTERIOR del alma (con la intersección), m
ALA_X = 0.85                 # solo tipo II: eje del ala desde el extremo izquierdo, m

FC_KGF_CM2 = 210.0
FC = FC_KGF_CM2 * 0.0980665  # f'c en MPa
FY_KGF_CM2 = 4200.0
FY = FY_KGF_CM2 * 0.0980665  # fy en MPa
ES = 200000.0                # MPa
EC = 4700.0 * math.sqrt(FC)  # §8.5.2, concreto de peso normal, MPa (solo para el MEF)
NU = 0.15                    # Poisson del concreto

# Barras corrugadas comerciales en el Perú: ASTM A615 Grado 60 (NTP 341.031
# Grado 420). Valores NOMINALES: diámetro (mm), área (cm²) y masa (kg/m).
BARRAS = {
    "6mm":     (6.0, 0.28, 0.222),
    "8mm":     (8.0, 0.50, 0.395),
    '3/8"':    (9.5, 0.71, 0.560),
    "12mm":    (12.0, 1.13, 0.888),
    '1/2"':    (12.7, 1.29, 0.994),
    '5/8"':    (15.9, 1.99, 1.552),
    '3/4"':    (19.1, 2.84, 2.235),
    '1"':      (25.4, 5.10, 3.973),
    '1 3/8"':  (35.8, 10.06, 7.907),
}

RECUBRIMIENTO = 0.04         # libre, hasta el estribo (o hasta la malla horizontal), m
HORIZONTAL_POR_FUERA = True  # la malla horizontal va por fuera de la vertical
AGREGADO_MM = 19.0           # tamaño máximo del agregado (§3.3.2(c): libre ≥ 4/3 del agregado)

# Juntas de construcción (en la cimentación y en cada losa, §21.9.8): el
# coeficiente de fricción de §11.7.4.3: 1.4 (concreto colocado
# monolíticamente), 1.0 (contra concreto endurecido con la superficie
# intencionalmente rugosa, §11.7.9) o 0.6 (no rugosa).
MU_JUNTA = 1.0

# -----------------------------------------------------------------------------
# 5. ARMADO DE BASE (el que se verifica desde z = 0)
# -----------------------------------------------------------------------------
# ELEMENTOS CONFINADOS: uno en cada extremo y, en los tipos II y III, uno más
# para el ala (BORDE_ALA, con toda la intersección). En cada uno:
#   "barra":  el diámetro de sus barras verticales (uno solo);
#   "nx":     barras en cada cara paralela a X (a lo largo del muro), ≥ 2;
#   "ny":     barras en cada cara paralela a Y (a través del espesor), ≥ 2;
#             van en el perímetro del estribo: 2·nx + 2·(ny − 2) barras;
#   "estribo", "separacion_estribo": el estribo cerrado y su separación, m,
#             múltiplo de 2 o 5 cm (replanteo);
#   "separacion_estribo_arriba": opcional, solo en la base: la separación
#             encima de la altura de cambio (la calcula el programa, ver el
#             informe, 3.3), hasta la cota de corte; sin ella, la misma;
#   "longitud": solo en los extremos: el largo confinado en X, m. Es
#               OBLIGATORIA; la norma no pide un mínimo, pero si se exige
#               borde confinado tiene que llegar a máx(c − 0.1·ℓm, c/2)
#               (§21.9.7.6(a)). En el tipo III, si el derecho no llega a b',
#               se confina todo el ancho del ala; si pasa de b', al menos una
#               columna de sus barras tiene que quedar fuera del ala.
# Opcional:
#   "grapas":   "alternas" (lo mínimo de §7.10.5.3, por defecto) o "todas".
# El programa coloca las barras y comprueba la separación libre mínima
# (§7.6.3) y la máxima entre barras apoyadas (hx ≤ 350 mm, §21.6.4.6). El
# informe (módulo 3) dice si el muro exige elemento de borde confinado y qué
# longitud y estribos pide.
BORDE_IZQUIERDO = {"barra": '5/8"', "nx": 2, "ny": 2, "longitud": 0.26, "estribo": '3/8"',
                   "separacion_estribo": 0.10}
BORDE_DERECHO = {"barra": '5/8"', "nx": 2, "ny": 2, "longitud": 0.26, "estribo": '3/8"',
                 "separacion_estribo": 0.10}
# Tipos II y III: el ala entera, con la intersección (b' × l'w), sin
# «longitud». En el tipo I: BORDE_ALA = None.
BORDE_ALA = {"barra": '5/8"', "nx": 2, "ny": 4, "estribo": '3/8"', "separacion_estribo": 0.10}

# MALLA DEL ALMA, en los tramos libres entre elementos (si el hueco de un
# tramo no pasa de la separación, no lleva malla vertical). Con "separacion":
# None, el programa toma la mayor que da la cuantía mínima (0.0025 de
# §11.10.10; si hm/ℓm ≤ 2, además ρv ≥ ρh, §21.9.5.4), múltiplo de 2 o 5 cm
# (replanteo, como la separación que se declare), sin pasar de 3 veces el espesor ni de
# 40 cm (§21.9.4.2).
MALLA_VERTICAL = {"barra": '3/8"', "cortinas": 2, "separacion": 0.15}
MALLA_HORIZONTAL = {"barra": '3/8"', "cortinas": 2, "separacion": 0.15}

# -----------------------------------------------------------------------------
# 6. ARMADO REDUCIDO (el que se quiere poner más arriba)
# -----------------------------------------------------------------------------
# Se declara solo lo que cambia; lo demás se toma de la base:
#   "malla_vertical", "malla_horizontal": como en la base; la "barra" de cada
#       malla es la misma de la base (cambia solo la separación; la de la
#       vertical, k veces la de la base: una de cada k líneas);
#   "borde_izquierdo", "borde_derecho" y, en los tipos II y III, "borde_ala":
#       "barra", "nx", "ny", "estribo", "separacion_estribo", "grapas" y, en
#       los extremos, "longitud" (no más larga que la de la base y terminada
#       en una columna de barras de la base: si no, el programa da la lista).
# Cada barra vertical del reducido continúa una barra de la base, en su misma
# posición y de diámetro igual o menor. El programa calcula desde qué losa se
# puede poner (la cota de corte, módulo 1.6, con su junta) y avisa si lleva
# más acero que la base. Para no reducir: ARMADO_REDUCIDO = None
ARMADO_REDUCIDO = {
    "malla_vertical": {"barra": '3/8"', "separacion": 0.15},
    "malla_horizontal": {"barra": '3/8"', "separacion": 0.20},
    "borde_izquierdo": {"barra": '5/8"', "nx": 2, "ny": 2, "longitud": 0.26,
                        "estribo": '3/8"', "separacion_estribo": 0.15},
    "borde_derecho": {"barra": '5/8"', "nx": 2, "ny": 2, "longitud": 0.26, "estribo": '3/8"',
                      "separacion_estribo": 0.15},
    "borde_ala": {"barra": '5/8"', "nx": 2, "ny": 4, "estribo": '3/8"',
                  "separacion_estribo": 0.15},
}

# -----------------------------------------------------------------------------
# 7. OPCIONES DEL CÁLCULO (normalmente no se cambian)
# -----------------------------------------------------------------------------
MALLA_MEF = 0.20             # tamaño de los elementos finitos, m

# Peso propio del muro dentro de cada entrepiso. True: el MEF lo reparte en la
# altura y P varía en recta entre la base y el tope de ETABS. False: P
# constante en el entrepiso, el de la base.
AXIAL_CON_PESO_PROPIO = True

# Sismo dentro de cada entrepiso. True: se reconstruye con la base y el tope
# de ETABS (exacto para SRSS y CQC, ver mef.py). False: por estática desde la
# base.
SISMO_ENTRE_BASE_Y_TOPE = True

# Las combinaciones con sismo vienen como envolventes (filas Max y Min).
# True: se prueban los 8 signos de la parte de sismo en P, Mx y My, como hace
# ETABS en su diseño. False: solo las filas Max y Min tal como vienen.
PERMUTAR_SIGNOS = True
PASO_DE_COTAS = 0.10         # paso para buscar la cota de corte, m
PUNTOS_DEL_DIAGRAMA = 40     # valores de c por sentido en el diagrama P-M
ANGULOS_BIAXIAL = 72         # giros del eje neutro (cada 5°)
PROFUNDIDADES_BIAXIAL = 100  # valores de c por giro
DIBUJAR = True               # guarda las figuras PNG
