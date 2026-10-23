"""OPTIMIZACIÓN DEL ARMADO CON UN ALGORITMO GENÉTICO.

Busca el armado que lleva MENOS ACERO TOTAL —vertical, malla horizontal y
estribos con sus grapas— sin violar ninguna comprobación del programa, en dos
etapas en secuencia (ETAPAS):
  1. el armado de base, el mismo en toda la altura;
  2. sobre ese óptimo, fijo, el armado reducido de encima de la cota de corte
     (la cota la calcula el programa, como en ejecutar.py; el peso es el de la
     base hasta el corte más el del reducido encima).
Se corre igual que ejecutar.py:

    python optimizar.py                    (el muro de MURO_A_EJECUTAR, en ejecutar.py)
    python optimizar.py edificacion_01/datos_P11.py

o con el botón ▷ de VS Code sobre este archivo. Escribe en
resultados/<MURO>/5_optimizacion/ el informe de la búsqueda y la curva de
convergencia, y junto a los datos el archivo datos_<MURO>_optimo.py (los datos
del muro con el armado óptimo), que se verifica con el programa completo
(su informe va a resultados/<MURO>_optimo/).

CÓMO FUNCIONA
  - La demanda del análisis (ETABS y MEF en cada cota) no depende del
    armado: se calcula una vez y la comparten todos los candidatos. El
    cortante de diseño sí (§21.9.5.5 lo amplifica con el Mn de la base): cada
    candidato lo calcula con su acero.
  - Cada candidato es una lista de GENES, cada uno un índice dentro de un
    catálogo de valores comerciales (barras de BARRAS, separaciones de
    estribos y mallas múltiplos de 2 o 5 cm, longitudes confinadas
    desde LONGITUD_MINIMA, múltiplos de 2 o 5 cm; en el reducido, las que terminan en
    una columna de barras de la base, `A.longitudes_sobre_la_base`).
  - Evaluación rápida: primero el detallado que no depende de la demanda
    (barras que caben, separaciones, cuantías, hx, diámetro del estribo); si
    pasa, la verificación completa en la base (flexocompresión biaxial con una
    superficie más gruesa, corte, borde confinado, estribos, junta…) y, en la
    base y el tope de cada entrepiso, el D/C biaxial, la sección crítica
    única, el cortante y, en cada losa, la junta de construcción.
  - Etapa 2: los genes son los del reducido (mismos catálogos, salvo la
    barra de cada malla, que es la de la base; la «longitud» de cada
    extremo, que no pasa de la de la base y termina en una columna de barras
    de la base; y la barra y las barras por cara
    de cada elemento, no mayores que en la base, porque cada barra del
    reducido continúa una de la base: criterios del proyecto). Por
    cada uno, la cota de corte con menos cotas (el biaxial cada
    PASO_DE_COTAS_BUSQUEDA y en las losas; lo que falla por el reducido, en
    las caras de cada entrepiso), llevada a la losa más próxima por encima;
    un reducido cuyo corte cae en el techo no sirve.
  - Reglas de Deb: un armado que cumple gana a uno que no; entre dos que
    cumplen, el más liviano (a igual peso, el de mallas más abiertas); entre
    dos que no, el que falla menos.
  - Al final de cada búsqueda, una búsqueda local desde el mejor: cada gen un
    paso arriba y abajo; si no mejora, cada pareja de genes de un mismo
    elemento o malla con todos sus valores; y si tampoco, todas las
    combinaciones de los genes de cada elemento o malla.
  - ARRANQUES búsquedas independientes (semillas distintas, memoria común);
    se toma la mejor.
  - El mejor se verifica al final con la resolución completa de los datos y
    todas las cotas, como ejecutar.py; si algo menor falla, se toma el
    siguiente. En la etapa 2 se verifican los 5 mejores (la cota de corte
    completa puede moverse un poco) y se toma el más liviano que cumple.

Lo que sigue (PARÁMETROS) es lo único que conviene tocar.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date

sys.dont_write_bytecode = True      # sin carpetas __pycache__ junto a los archivos

import ejecutar  # carga el archivo de datos (argumento o MURO_A_EJECUTAR) y los cálculos
from ejecutar import A, D, F, mef

# =============================================================================
# PARÁMETROS
# =============================================================================
POBLACION = 60               # armados por generación
GENERACIONES = 120           # como máximo
PARADA = 30                  # se para si en tantas generaciones no mejora el mejor
SEMILLA = 1                  # semilla del primer arranque (los siguientes: SEMILLA + 1, + 2…)
ARRANQUES = 5                # búsquedas independientes; se queda con la mejor (1 = más rápido)
PROCESOS = 0                 # núcleos en paralelo; 0 = todos menos uno; 1 = sin paralelo
RESOLUCION_BIAXIAL = (36, 40)    # giros y profundidades durante la búsqueda (la final usa los de los datos)
VERIFICAR_COMPLETO = True    # al final, correr ejecutar.py con el armado óptimo

# Qué se optimiza:
#   "las dos":  etapa 1 (el armado de base) y, sobre su óptimo, etapa 2 (el
#               armado reducido de encima de la cota de corte)
#   "base":     solo la etapa 1 (el mismo armado en toda la altura)
#   "reducido": solo la etapa 2, sobre el armado de base declarado en los datos
ETAPAS = "las dos"
# Etapa 2: durante la búsqueda, el biaxial del reducido se mira cada tanto
# (además de las losas) y su detallado en las caras de cada entrepiso; la
# verificación final (ejecutar.py) lo mira todo, en las cotas de los datos.
PASO_DE_COTAS_BUSQUEDA = 0.50    # m

# Criterio del proyecto (el mismo del programa con el ACI): las barras
# longitudinales de los elementos confinados son de 5/8" o más (con menos, la
# separación de estribos que pide la norma, 8·db en el borde confinado, queda
# muy cerrada para construir).
BARRA_MINIMA_BORDE = '5/8"'

# Catálogos (los nombres de barra tienen que estar en BARRAS de los datos)
EXTREMO_NX = (2, 3, 4, 5, 6)         # barras por cara paralela a X en los extremos
EXTREMO_NY = (2, 3, 4)               # barras por cara paralela a Y (a través del espesor)
ALA_NX = (2, 3, 4)                   # en el ala: a través de b'
ALA_NY = (2, 3, 4, 5, 6, 7, 8)       # en el ala: a lo largo de l'w
LONGITUD_MINIMA = 0.15               # «longitud» confinada mínima, m (la norma no pide un mínimo)
LONGITUD_MAXIMA = 0.55               # «longitud» confinada máxima, en fracción de ℓm
ESTRIBOS = ('3/8"', '1/2"')
# Separaciones en m, múltiplos de 2 o 5 cm (replanteo); los estribos, desde
# 8 cm por construcción (más juntos cuesta vaciar) hasta 20 cm. En la base,
# la de abajo (hasta la altura de cambio) y la de arriba; en el reducido, una
SEPARACIONES_ESTRIBO = tuple(k / 100 for k in range(8, 21) if k % 2 == 0 or k % 5 == 0)  # 8, 10, 12, … 20
BARRAS_MALLA = ('3/8"', '1/2"')
SEPARACIONES_MALLA_VERTICAL = tuple(k / 100 for k in range(10, 31) if k % 2 == 0 or k % 5 == 0)
SEPARACIONES_MALLA_HORIZONTAL = SEPARACIONES_MALLA_VERTICAL                # 10, 12, 14, 15, 16, …, 30

# Probabilidades del algoritmo
CRUCE = 0.9                  # probabilidad de cruzar dos padres
MUTACION = 1.5               # genes que muta, en promedio, cada hijo
ELITE = 2                    # los mejores pasan sin cambios

# =============================================================================
# LOS GENES
# =============================================================================


def simetrico(bloques_base: dict | None = None) -> bool:
    """Si la planta es simétrica (el tipo I, o el tipo II con el ala en medio
    del alma, ALA_X = ℓm/2), el izquierdo y el derecho llevan el mismo armado
    (criterio del proyecto). En la etapa 2, además, la base tiene que serlo."""
    sec = A.seccion()
    planta = sec.tipo == "I" or (sec.tipo == "II" and abs(D.ALA_X - D.LW / 2) < 1e-6)
    return planta and (bloques_base is None or bloques_base["izquierdo"] == bloques_base["derecho"])


def catalogo_de_genes(bloques_base: dict | None = None) -> list[tuple[tuple[str, str], tuple]]:
    """[((elemento, clave), valores posibles), ...]. Con la planta simétrica,
    un solo elemento «extremos» para los dos (`simetrico`). En la etapa 2 (bloques_base,
    la base fija), criterios del proyecto: la barra de cada malla es la de la
    base (mismo diámetro en la base y en el reducido: un solo valor), la
    «longitud» de cada extremo no pasa de la de la base y las barras de cada
    elemento continúan las de la base (diámetro y barras por cara, no más que
    en ella; que cada una caiga sobre una de la base lo comprueba `A.armar`)."""
    sec = A.seccion()
    for b in (BARRA_MINIMA_BORDE, *ESTRIBOS, *BARRAS_MALLA):
        if b not in D.BARRAS:
            raise A.DatoInvalido(f"optimizar.py: la barra {b!r} del catálogo no está en BARRAS")
    minimo = A.diametro(BARRA_MINIMA_BORDE)
    borde = tuple(sorted((b for b in D.BARRAS if A.diametro(b) >= minimo - 1e-9), key=A.diametro))
    # desde LONGITUD_MINIMA, los múltiplos de 2 o 5 cm (en cm enteros)
    desde = round(LONGITUD_MINIMA * 100)
    tope = math.floor(LONGITUD_MAXIMA * D.LW * 100 + 1e-9)
    longitudes = [k / 100 for k in range(desde, tope + 1) if k % 2 == 0 or k % 5 == 0]

    def hasta_la_base(nombre):
        """Las longitudes del extremo, desde LONGITUD_MINIMA, múltiplos de 2 o 5 cm; en la
        etapa 2, las que terminan en una columna de barras de la base, hasta
        la de la base (el borde confinado se decide en la base, §21.9.7.4:
        arriba no se confina más)."""
        if bloques_base is None:
            return tuple(longitudes)
        lado = "izquierdo" if nombre == "extremos" else nombre
        base = armado_de(bloques_base)
        valores = A.longitudes_sobre_la_base(base.elemento(lado), base)
        # hasta la declarada en la base, aunque su caja sea otra (tipo III: un
        # derecho de menos de b' lo absorbe el ala, su caja mide b')
        tope = bloques_base[lado]["longitud"]
        return tuple(sorted({L for L in valores if L <= tope + 1e-9} | {tope}))

    def hasta(nombre, clave, valores):
        """En la etapa 2, las barras del reducido continúan las de la base
        (criterio del proyecto): de diámetro igual o menor y, por cara, no
        más que en la base (cada una sobre una barra de la base)."""
        if bloques_base is None:
            return tuple(valores)
        de_base = bloques_base["izquierdo" if nombre == "extremos" else nombre][clave]
        if clave == "barra":
            return tuple(b for b in valores if A.diametro(b) <= A.diametro(de_base) + 1e-9)
        return tuple(v for v in valores if v <= de_base)

    def separaciones(nombre):
        """En la base, la de abajo y la de arriba de la altura de cambio; en
        el reducido, una sola."""
        if bloques_base is not None:
            return [((nombre, "separacion_estribo"), SEPARACIONES_ESTRIBO)]
        return [((nombre, "separacion_estribo"), SEPARACIONES_ESTRIBO),
                ((nombre, "separacion_estribo_arriba"), SEPARACIONES_ESTRIBO)]

    def extremo(nombre):
        return [((nombre, "barra"), hasta(nombre, "barra", borde)),
                ((nombre, "nx"), hasta(nombre, "nx", EXTREMO_NX)),
                ((nombre, "ny"), hasta(nombre, "ny", EXTREMO_NY)),
                ((nombre, "longitud"), hasta_la_base(nombre)), ((nombre, "estribo"), ESTRIBOS),
                *separaciones(nombre)]

    genes = extremo("extremos") if simetrico(bloques_base) else extremo("izquierdo") + extremo("derecho")
    if sec.tipo != "I":
        genes += [(("ala", "barra"), hasta("ala", "barra", borde)), (("ala", "nx"), hasta("ala", "nx", ALA_NX)),
                  (("ala", "ny"), hasta("ala", "ny", ALA_NY)),
                  (("ala", "estribo"), ESTRIBOS), *separaciones("ala")]
    for malla, separaciones in (("malla_vertical", SEPARACIONES_MALLA_VERTICAL),
                                ("malla_horizontal", SEPARACIONES_MALLA_HORIZONTAL)):
        barras = BARRAS_MALLA if bloques_base is None else (bloques_base[malla]["barra"],)
        if bloques_base is not None and malla == "malla_vertical":
            # la del reducido va sobre una de cada k líneas de la base: k veces
            # su separación (criterio del proyecto)
            sb = bloques_base[malla]["separacion"]
            separaciones = tuple(round(k * sb, 4) for k in range(1, 10)
                                 if k * sb <= max(SEPARACIONES_MALLA_VERTICAL) + 1e-9)
        genes += [((malla, "barra"), barras), ((malla, "separacion"), separaciones)]
    return genes


def genes_en(catalogo, genes: tuple | None, completo) -> tuple | None:
    """Unos genes del catálogo completo, en el de la etapa 2 (cada valor, en
    su nuevo índice; None si alguno no está)."""
    if genes is None:
        return None
    salida = []
    for g, (_, valores), (_, todos) in zip(genes, catalogo, completo):
        iguales = [i for i, v in enumerate(valores) if v == todos[g]]
        if not iguales:
            return None
        salida.append(iguales[0])
    return tuple(salida)


def decodificar(genes: tuple, catalogo) -> dict:
    """Los genes → {"izquierdo": {...}, "derecho": {...}, "ala": {...},
    "malla_vertical": {...}, "malla_horizontal": {...}} con el formato de los
    datos."""
    bloques: dict = {}
    for ((elemento, clave), valores), i in zip(catalogo, genes):
        bloques.setdefault(elemento, {})[clave] = valores[i]
    if "extremos" in bloques:                     # planta simétrica
        bloques["izquierdo"] = dict(bloques["extremos"])
        bloques["derecho"] = dict(bloques.pop("extremos"))
    for malla in ("malla_vertical", "malla_horizontal"):
        bloques[malla]["cortinas"] = 2
    return bloques


def armado_de(bloques: dict, nombre: str = "base", debajo=None):
    """El armado de unos bloques; debajo: en un reducido, la base."""
    elementos = {n: bloques[n] for n in A.nombres_de_elementos()}
    return A.armar(elementos, bloques["malla_vertical"], bloques["malla_horizontal"], nombre, debajo)


def absorbido(armado) -> bool:
    """Tipo III: el derecho no tiene barras propias (las pone el ala)."""
    der = armado.elemento("derecho")
    return der is not None and A.absorbido(der, armado)


def limpiar(bloques: dict, armado) -> dict:
    """Si el ala absorbe al derecho (tipo III), sus barras y su estribo no
    existen: se escriben como los del ala, para que los datos no muestren
    valores que no se usan."""
    if absorbido(armado):
        ala = bloques["ala"]
        bloques["derecho"].update(barra=ala["barra"], nx=2, ny=2, estribo=ala["estribo"],
                                  separacion_estribo=ala["separacion_estribo"])
        if "separacion_estribo_arriba" in ala:
            bloques["derecho"]["separacion_estribo_arriba"] = ala["separacion_estribo_arriba"]
    # la separación de arriba igual a la de abajo no se escribe
    for d in bloques.values():
        if d.get("separacion_estribo_arriba") is not None and \
                abs(d["separacion_estribo_arriba"] - d.get("separacion_estribo", 0.0)) < 1e-9:
            del d["separacion_estribo_arriba"]
    return bloques


def bloques_declarados() -> dict:
    """El armado de base de los datos, con el formato de `decodificar`."""
    bloques = {"izquierdo": dict(D.BORDE_IZQUIERDO), "derecho": dict(D.BORDE_DERECHO),
               "malla_vertical": dict(D.MALLA_VERTICAL), "malla_horizontal": dict(D.MALLA_HORIZONTAL)}
    if A.seccion().tipo != "I":
        bloques["ala"] = dict(D.BORDE_ALA)
    return bloques


def genes_del_armado_declarado(catalogo) -> tuple | None:
    """El armado de base de los datos como genes, si sus valores están en los
    catálogos (para sembrarlo en la población inicial); si no, None."""
    datos = {"izquierdo": D.BORDE_IZQUIERDO, "derecho": D.BORDE_DERECHO,
             "ala": getattr(D, "BORDE_ALA", None) or {},
             "malla_vertical": D.MALLA_VERTICAL, "malla_horizontal": D.MALLA_HORIZONTAL}
    if simetrico():
        if D.BORDE_IZQUIERDO != D.BORDE_DERECHO:
            return None
        datos["extremos"] = D.BORDE_IZQUIERDO
    genes = []
    for (elemento, clave), valores in catalogo:
        valor = datos.get(elemento, {}).get(clave)
        if clave == "separacion_estribo_arriba" and valor is None:
            valor = datos.get(elemento, {}).get("separacion_estribo")     # sin ella, la misma
        if isinstance(valor, float):
            iguales = [i for i, v in enumerate(valores) if isinstance(v, float) and abs(v - valor) < 1e-6]
        else:
            iguales = [i for i, v in enumerate(valores) if v == valor]
        if not iguales:
            return None
        genes.append(iguales[0])
    return tuple(genes)


# =============================================================================
# LA EVALUACIÓN DE UN CANDIDATO
# =============================================================================


@dataclass
class Resultado:
    genes: tuple
    peso: float                      # kg (inf si no se pudo calcular)
    violacion: float                 # 0 si cumple; cuánto falla si no
    motivo: str = ""                 # qué falla más
    partes: dict = field(default_factory=dict)
    mallas: float = 0.0              # separación de la malla vertical + la horizontal, m
    saltos: float = 0.0              # Σ |separación de estribos de arriba − la de abajo|, m

    @property
    def cumple(self) -> bool:
        return self.violacion <= 0.0

    @property
    def clave(self) -> tuple:
        """Reglas de Deb: primero los que cumplen (por peso), luego los demás
        (por cuánto fallan). A igual peso, las mallas más abiertas (p. ej. una
        malla vertical que no pone barras porque sus tramos son cortos) y,
        después, la separación de estribos de arriba más cerca de la de abajo
        (p. ej. si la base no llega por encima de la altura de cambio)."""
        return ((0, round(self.peso, 6), -self.mallas, self.saltos) if self.cumple
                else (1, self.violacion, 0.0, 0.0))


def exceso(f: dict) -> float:
    """Cuánto falla una comprobación de detallado, relativo a su límite."""
    valor, limite, tipo = f.get("valor"), f.get("limite"), f.get("tipo", "≥")
    try:
        if tipo in ("≥", ">") and limite:
            return max((limite - valor) / abs(limite), 1e-3)
        if tipo in ("≤", "<") and limite:
            return max((valor - limite) / abs(limite), 1e-3)
    except TypeError:
        pass
    return 1.0


def violacion_de(v) -> tuple[float, str]:
    """Cuánto falla una verificación completa en una cota, y qué falla más."""
    partes = [(min(v.dc_biaxial, 100.0) - 1.0, "flexocompresión biaxial"),
              (v.dc_corte - 1.0, "corte"), (v.dc_borde - 1.0, "longitud del borde confinado"),
              (min(v.dc_junta, 100.0) - 1.0, "junta de construcción")]
    partes += [(exceso(f), f"[{f['art']}] {f['texto']}") for f in v.faltas]
    malas = [(x, t) for x, t in partes if x > 0]
    if not malas:
        return 0.0, ""
    return sum(x for x, _ in malas), max(malas)[1]


# Lo que cada proceso calcula una vez (`preparar`)
CONTEXTO: dict = {}


def preparar() -> None:
    """La demanda, una vez por proceso: no depende del armado."""
    if CONTEXTO:
        return
    A.validar_datos()
    mef.ejes()
    sec = A.seccion()
    fuerzas = mef.mapear(mef.leer_fuerzas_etabs())
    dem = mef.Demanda(mef.resolver(fuerzas), fuerzas, (sec.xe - sec.xc, sec.ye - sec.yc))
    cotas = [(dem.z_base[s], s) for s in range(len(D.ALTURAS))]
    cotas += [(dem.z_tope[s], s) for s in range(len(D.ALTURAS))]
    # el cortante del análisis (sin amplificar) en la base: con él, el
    # detallado previo usa los límites más holgados
    vu = max(abs(e.V) for e in dem.en(0.0, 0))
    CONTEXTO.update(demanda=dem, estados={}, cotas=cotas, vu_base=vu,
                    catalogo=catalogo_de_genes())


def evaluar_armado(armado, genes: tuple = (), resolucion=RESOLUCION_BIAXIAL) -> Resultado:
    """La evaluación rápida (ver la cabecera)."""
    preparar()
    dis = A.Diseno(CONTEXTO["demanda"], D.DELTA_U_MM / 1000.0, resolucion, CONTEXTO["estados"])
    barras = dis.barras(armado)
    # 1) el detallado que no depende de la demanda, con los límites MENOS
    #    estrictos (sin borde confinado y con el cortante del análisis): lo
    #    que falla aquí falla seguro
    filas = A.detallado(armado, barras, CONTEXTO["vu_base"], 0, set())
    # la separación de arriba, sin borde confinado (lo más holgado)
    arriba = A.con_separacion_de_arriba(armado)
    if arriba is not armado:
        filas += A.detallado(arriba, barras, CONTEXTO["vu_base"], 0, set())
    malas = [f for f in filas if not f["cumple"]]
    if malas:
        return Resultado(genes, math.inf, 100.0 + sum(exceso(f) for f in malas),
                         f"[{malas[0]['art']}] {malas[0]['texto']}")
    # 2) la verificación completa en la base
    vb = dis.verificar(armado, 0.0, 0)
    total, motivo = violacion_de(vb)
    # 3) en la base y el tope de cada entrepiso: biaxial, sección crítica
    #    única y corte; en cada losa, la junta de construcción
    base = max(vb.dc_biaxial, 1e-9)
    for z, s in CONTEXTO["cotas"]:
        if z <= A.TOL:
            continue
        dc, _ = A.dc_biaxial_en(dis, armado, z, s)
        dc = min(dc, 100.0)
        estados = dis.estados(z, s)
        dcc = max(A.corte_en_el_plano(armado, e)["dc"] for e in estados)
        dcj = A.friccion_en_la_junta(barras, estados)["dc"] if dis.es_junta(z, s) else 0.0
        for x, texto in ((dc - 1.0, f"biaxial en z = {z:.2f} m"),
                         (dc / base - 1.0 if dc > base + 1e-9 else 0.0,
                          f"sección crítica única en z = {z:.2f} m"),
                         (dcc - 1.0, f"corte en z = {z:.2f} m"),
                         (min(dcj, 100.0) - 1.0, f"junta de construcción en z = {z:.2f} m")):
            if x > 0:
                total += x
                motivo = motivo or texto
    partes = A.peso_de_acero(dis, armado)
    return Resultado(genes, partes["total"], total, motivo, partes)


def peso_del_armado(armado) -> float:
    """El peso de acero de un armado cualquiera (aunque no cumpla)."""
    preparar()
    dis = A.Diseno(CONTEXTO["demanda"], D.DELTA_U_MM / 1000.0, RESOLUCION_BIAXIAL, CONTEXTO["estados"])
    dis.verificar(armado, 0.0, 0)                 # decide las zonas de estribos
    return A.peso_de_acero(dis, armado)["total"]


# -----------------------------------------------------------------------------
# Etapa 2: el armado reducido sobre una base fija
# -----------------------------------------------------------------------------


def preparar_etapa(bloques_base: dict | None) -> None:
    """Etapa 1 (None) o etapa 2 (los bloques de la base fija). En la etapa 2,
    un Diseno con la base ya verificada en z = 0 (decide el cortante de
    diseño y el borde confinado, como en ejecutar.py), que comparten todos los
    reducidos del proceso; y las cotas de la búsqueda."""
    preparar()
    CONTEXTO.pop("etapa2", None)
    CONTEXTO["catalogo"] = catalogo_de_genes(bloques_base)
    if bloques_base is None:
        return
    base = armado_de(bloques_base)
    dis = A.Diseno(CONTEXTO["demanda"], D.DELTA_U_MM / 1000.0, RESOLUCION_BIAXIAL, CONTEXTO["estados"])
    vb = dis.verificar(base, 0.0, 0)
    cotas = dis.cotas()
    paso = max(1, round(PASO_DE_COTAS_BUSQUEDA / D.PASO_DE_COTAS))
    CONTEXTO["etapa2"] = {
        "base": base, "dis": dis, "dc_base": vb.dc_biaxial,
        "cotas": [c for i, c in enumerate(cotas) if c[2] or i % paso == 0],
        "caras": [c for c in cotas if c[2]],
        "peso_base": A.peso_de_acero(dis, base)["total"]}


def evaluar_reducido(red, genes: tuple = ()) -> Resultado:
    """Un reducido sobre la base fija: su detallado más holgado (lo que falla
    ahí falla seguro), la cota de corte (con menos cotas, ver
    PASO_DE_COTAS_BUSQUEDA) y el peso de la base hasta el corte más el del
    reducido encima. Un reducido cuyo corte cae en el techo no se puede usar:
    cuenta como que no cumple."""
    e2 = CONTEXTO["etapa2"]
    dis, base = e2["dis"], e2["base"]
    try:
        # sin borde confinado y sin cortante: los límites más holgados
        filas = A.detallado(red, dis.barras(red), 0.0, len(D.ALTURAS) - 1, set())
        malas = [f for f in filas if not f["cumple"]]
        if malas:
            return Resultado(genes, math.inf, 100.0 + sum(exceso(f) for f in malas),
                             f"[{malas[0]['art']}] {malas[0]['texto']}")
        cc = A.cota_de_corte(dis, base, red, e2["dc_base"], e2["cotas"], e2["caras"])
        if cc.corte >= A.H - A.TOL:
            # cuánto pasa la mayor cota de la norma de la última losa bajo el techo
            losas = dis.demanda.z_tope[:-1]
            z_norma = max(z for _, z in cc.cotas_de_la_norma)
            sobra = max(0.0, z_norma - (losas[-1] if losas else 0.0)) / A.H
            motivo = (cc.fallas_del_reducido[0][3] if cc.fallas_del_reducido
                      else "el acero vertical no se puede cortar antes del techo")
            return Resultado(genes, math.inf, 1.0 + sobra + (0.5 if cc.fallas_del_reducido else 0.0),
                             f"no se puede usar: {motivo}")
        partes = A.peso_de_acero(dis, base, red, cc.corte)
        partes["corte"] = cc.corte
        return Resultado(genes, partes["total"], 0.0, "", partes)
    finally:
        dis.olvidar(red)


def verificacion_del_reducido(base, red) -> tuple[bool, list[str], float, dict]:
    """Como el veredicto de ejecutar.py con base y reducido, con la resolución
    y todas las cotas de los datos: la cota de corte completa, la verificación
    en la base, cada losa y el corte y el acero de base bajo el corte.
    Devuelve (cumple, fallas, corte, peso)."""
    preparar()
    dis = A.Diseno(CONTEXTO["demanda"], D.DELTA_U_MM / 1000.0, None, CONTEXTO["estados"])
    vb = dis.verificar(base, 0.0, 0)
    cc = A.cota_de_corte(dis, base, red, vb.dc_biaxial)
    fallas = []
    if cc.corte >= A.H - A.TOL:
        fallas.append(f"el corte cae en el techo ({cc.gobierna}): el reducido no se usa")
    for etiqueta, arm, z, piso in ejecutar.puntos_de_control(dis, base, red, cc.corte):
        v = dis.verificar(arm, z, piso)
        if not v.cumple:
            fallas.append(f"{etiqueta}: {violacion_de(v)[1] or 'D/C > 1'}")
    bajo = A.revisar_bajo_el_corte(dis, base, cc.corte, vb.dc_biaxial)
    if bajo["falla_biaxial"] is not None or bajo["falla_unica"] is not None:
        fallas.append(f"entre losas: D/C biaxial {bajo['peor_dc']:.3f} en z = {bajo['peor_z']:.2f} m")
    partes = A.peso_de_acero(dis, base, red, cc.corte)
    partes["corte"] = cc.corte
    partes["gobierna"] = cc.gobierna
    return not fallas, fallas, cc.corte, partes


def evaluar(genes: tuple) -> Resultado:
    preparar()
    etapa2 = "etapa2" in CONTEXTO
    try:
        bloques = decodificar(genes, CONTEXTO["catalogo"])
        armado = (armado_de(bloques, "reducido", CONTEXTO["etapa2"]["base"]) if etapa2
                  else armado_de(bloques))
    except A.DatoInvalido as error:
        return Resultado(genes, math.inf, 1000.0, f"no se puede armar: {error}")
    try:
        r = evaluar_reducido(armado, genes) if etapa2 else evaluar_armado(armado, genes)
    except (ValueError, ZeroDivisionError) as error:      # una sección imposible
        return Resultado(genes, math.inf, 500.0, f"error de cálculo: {error}")
    r.mallas = armado.malla_vertical.separacion + armado.malla_horizontal.separacion
    r.saltos = sum(abs(e.separacion_arriba - e.separacion_estribo) for e in armado.elementos
                   if e.separacion_arriba)
    return r


def verificacion_completa(armado) -> tuple[bool, list[str]]:
    """Como el veredicto de ejecutar.py, con la resolución de los datos: la
    verificación en la base y en cada losa, y el biaxial y la sección crítica
    única en todas las cotas."""
    preparar()
    dis = A.Diseno(CONTEXTO["demanda"], D.DELTA_U_MM / 1000.0, None, CONTEXTO["estados"])
    vb = dis.verificar(armado, 0.0, 0)
    fallas = []
    for etiqueta, arm, z, piso in ejecutar.puntos_de_control(dis, armado, None, A.H):
        v = dis.verificar(arm, z, piso)
        if not v.cumple:
            fallas.append(f"{etiqueta}: {violacion_de(v)[1] or 'D/C > 1'}")
    bajo = A.revisar_bajo_el_corte(dis, armado, A.H, vb.dc_biaxial)
    if bajo["falla_biaxial"] is not None or bajo["falla_unica"] is not None:
        fallas.append(f"entre losas: D/C biaxial {bajo['peor_dc']:.3f} en z = {bajo['peor_z']:.2f} m")
    return not fallas, fallas


# =============================================================================
# EL ALGORITMO GENÉTICO
# =============================================================================


def aleatorio(rng: random.Random, catalogo) -> tuple:
    return tuple(rng.randrange(len(valores)) for _, valores in catalogo)


def torneo(rng: random.Random, poblacion: list[Resultado]) -> Resultado:
    a, b = rng.sample(poblacion, 2)
    return a if a.clave <= b.clave else b


def cruzar(rng: random.Random, p1: tuple, p2: tuple) -> tuple:
    if rng.random() > CRUCE:
        return p1
    return tuple(a if rng.random() < 0.5 else b for a, b in zip(p1, p2))


def mutar(rng: random.Random, genes: tuple, catalogo) -> tuple:
    """Cada gen cambia con probabilidad MUTACION/n: casi siempre un paso arriba
    o abajo en su catálogo, a veces a cualquier valor."""
    salida = list(genes)
    p = MUTACION / len(genes)
    for i, (_, valores) in enumerate(catalogo):
        if len(valores) > 1 and rng.random() < p:
            if rng.random() < 0.8:
                salida[i] = min(max(salida[i] + rng.choice((-1, 1)), 0), len(valores) - 1)
            else:
                salida[i] = rng.randrange(len(valores))
    return tuple(salida)


def evaluar_lote(lote: list[tuple], memoria: dict, ejecutor) -> int:
    """Evalúa los genes que aún no están en la memoria; devuelve cuántos."""
    nuevos = list(dict.fromkeys(g for g in lote if g not in memoria))
    if nuevos:
        resultados = ejecutor.map(evaluar, nuevos, chunksize=2) if ejecutor else map(evaluar, nuevos)
        for r in resultados:
            memoria[r.genes] = r
    return len(nuevos)


class EnSerie:
    """Sustituto del ejecutor paralelo cuando PROCESOS = 1."""

    def map(self, f, datos, chunksize=1):
        return map(f, datos)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def un_arranque(semilla: int, inicial: list[tuple], memoria: dict, ejecutor, catalogo,
                salida, t0: float) -> dict:
    """Una búsqueda completa: el algoritmo genético desde una población
    aleatoria (más `inicial`) y la búsqueda local desde su mejor. La memoria
    es común a todos los arranques (lo ya evaluado no se repite)."""
    rng = random.Random(semilla)
    poblacion_genes = (inicial + [aleatorio(rng, catalogo) for _ in range(POBLACION)])[:POBLACION]
    historia = []
    mejor_peso, sin_mejorar = math.inf, 0
    for generacion in range(1, GENERACIONES + 1):
        nuevos = evaluar_lote(poblacion_genes, memoria, ejecutor)
        poblacion = sorted((memoria[g] for g in poblacion_genes), key=lambda r: r.clave)
        factibles = [r for r in poblacion if r.cumple]
        mejor = poblacion[0]
        historia.append({"generacion": generacion, "mejor": mejor.peso if mejor.cumple else math.nan,
                         "factibles": len(factibles), "nuevos": nuevos,
                         "evaluados": len(memoria), "tiempo": time.perf_counter() - t0})
        salida(f"  generación {generacion:3d}: mejor "
               + (f"{mejor.peso:8.1f} kg" if mejor.cumple else
                  f"(ninguno cumple; falla {mejor.violacion:.3f}: {mejor.motivo})")
               + f"   {len(factibles):2d}/{len(poblacion)} cumplen   {len(memoria):5d} evaluados   "
                 f"{time.perf_counter() - t0:6.0f} s")
        if mejor.cumple and mejor.peso < mejor_peso - 1e-6:
            mejor_peso, sin_mejorar = mejor.peso, 0
        else:
            sin_mejorar += 1
        if sin_mejorar >= PARADA:
            salida(f"  sin mejorar en {PARADA} generaciones: se para")
            break
        hijos = [r.genes for r in poblacion[:ELITE]]
        while len(hijos) < POBLACION:
            h = cruzar(rng, torneo(rng, poblacion).genes, torneo(rng, poblacion).genes)
            hijos.append(mutar(rng, h, catalogo))
        poblacion_genes = hijos
    # con élite, el primero de la última población es el mejor del arranque
    del_ag = poblacion[0]
    final, pasos = busqueda_local(del_ag, memoria, ejecutor, catalogo)
    if pasos:
        salida(f"  búsqueda local desde el mejor: {pasos} paso(s) de mejora → {final.peso:.1f} kg")
    else:
        salida("  búsqueda local desde el mejor: ningún vecino (un gen un paso arriba o abajo, dos "
               "genes o todos los de un mismo elemento o malla) lo mejora")
    return {"semilla": semilla, "historia": historia, "del_ag": del_ag, "final": final,
            "pasos_locales": pasos}


def optimizar(salida=print, inicial: tuple | None = None, bloques_base: dict | None = None,
              que_es: str = "el armado declarado") -> dict:
    """ARRANQUES búsquedas independientes (semillas SEMILLA, SEMILLA + 1, …)
    con una memoria común; el resultado es el mejor de todas. bloques_base:
    None en la etapa 1; en la etapa 2, la base fija (los genes buscan el
    reducido). inicial: unos genes que entran en la población inicial del
    primer arranque (que_es dice cuáles)."""
    preparar_etapa(bloques_base)
    catalogo = CONTEXTO["catalogo"]
    procesos = PROCESOS or max(1, (os.cpu_count() or 2) - 1)
    memoria: dict = {}
    t0 = time.perf_counter()
    arranques = []
    ejecutor_ctx = (ProcessPoolExecutor(max_workers=procesos, initializer=preparar_etapa,
                                        initargs=(bloques_base,))
                    if procesos > 1 else EnSerie())
    with ejecutor_ctx as ejecutor:
        for k in range(max(1, ARRANQUES)):
            semilla = SEMILLA + k
            salida(f"  Arranque {k + 1} de {max(1, ARRANQUES)} (semilla {semilla}"
                   + (f", con {que_es} en la población inicial)" if k == 0 and inicial else ")") + ":")
            arranques.append(un_arranque(semilla, [inicial] if (k == 0 and inicial) else [],
                                         memoria, ejecutor, catalogo, salida, t0))
    ranking = sorted(memoria.values(), key=lambda r: r.clave)
    return {"ranking": ranking, "arranques": arranques, "procesos": procesos,
            "tiempo": time.perf_counter() - t0, "catalogo": catalogo}


def vecinos_de_un_paso(genes: tuple, catalogo) -> list[tuple]:
    """Cada gen, un paso arriba y uno abajo en su catálogo."""
    vecinos = []
    for i, (_, valores) in enumerate(catalogo):
        for d in (-1, 1):
            j = genes[i] + d
            if 0 <= j < len(valores):
                vecinos.append(genes[:i] + (j,) + genes[i + 1:])
    return vecinos


def vecinos_por_parejas(genes: tuple, catalogo) -> list[tuple]:
    """Dos genes del MISMO elemento o malla a la vez, con todos sus valores
    (p. ej. pasar la malla horizontal de 1/2"@30 a 3/8"@17, o la barra y la
    separación de estribos juntas): lo que un paso de un solo gen no alcanza."""
    vecinos = []
    for i, ((bloque_i, _), valores_i) in enumerate(catalogo):
        for j in range(i + 1, len(catalogo)):
            (bloque_j, _), valores_j = catalogo[j]
            if bloque_j != bloque_i:
                continue
            for a in range(len(valores_i)):
                for b in range(len(valores_j)):
                    if a != genes[i] and b != genes[j]:
                        g = list(genes)
                        g[i], g[j] = a, b
                        vecinos.append(tuple(g))
    return vecinos


def vecinos_por_elemento(genes: tuple, catalogo) -> list[tuple]:
    """Todos los genes de un MISMO elemento o malla a la vez, con todas sus
    combinaciones (el resto queda fijo): cambios de varios genes juntos que ni
    un paso ni una pareja alcanzan (p. ej. un extremo de 4×2, L = 0.50 m,
    3/8"@8 a 3×2, L = 0.35 m, 1/2"@9). Casi todos fallan el detallado barato
    en milisegundos."""
    bloques: dict = {}
    for i, ((bloque, clave), _) in enumerate(catalogo):
        # la separación de arriba casi no se cruza con lo demás del elemento:
        # va sola (si no, las combinaciones se multiplican por su catálogo)
        bloques.setdefault((bloque, clave == "separacion_estribo_arriba"), []).append(i)
    vecinos = []
    for indices in bloques.values():
        for valores in itertools.product(*(range(len(catalogo[i][1])) for i in indices)):
            g = list(genes)
            for i, v in zip(indices, valores):
                g[i] = v
            vecinos.append(tuple(g))
    return vecinos


def busqueda_local(mejor: Resultado, memoria: dict, ejecutor, catalogo) -> tuple[Resultado, int]:
    """Desde el mejor: prueba cada gen un paso arriba y abajo en su catálogo;
    si ninguno mejora, cada pareja de genes de un mismo elemento o malla con
    todos sus valores; si tampoco, todas las combinaciones de cada elemento o
    malla. Se mueve al mejor vecino mientras alguno mejore (reglas de Deb).
    Devuelve el armado al que llega y los pasos dados."""
    pasos = 0
    while True:
        for vecindad in (vecinos_de_un_paso, vecinos_por_parejas, vecinos_por_elemento):
            vecinos = vecindad(mejor.genes, catalogo)
            evaluar_lote(vecinos, memoria, ejecutor)
            candidato = min((memoria[g] for g in vecinos), key=lambda r: r.clave)
            if candidato.clave < mejor.clave:
                mejor, pasos = candidato, pasos + 1
                break
        else:
            return mejor, pasos


# =============================================================================
# EL INFORME
# =============================================================================


def _dic(d: dict, orden) -> str:
    """El diccionario como en los datos: primero las claves de `orden`, luego
    las demás (p. ej. «grapas» de un armado declarado)."""
    claves = [k for k in orden if k in d] + [k for k in d if k not in orden]
    return "{" + ", ".join(f'"{k}": {d[k]!r}' for k in claves) + "}"


MARCA_DEL_OPTIMO = "# ARMADO ÓPTIMO — lo escribió optimizar.py"


def datos_originales() -> str:
    """El archivo de datos sin lo que optimizar.py le añadió en otra corrida
    (si se optimiza un datos_<MURO>_optimo.py)."""
    texto = open(D.__file__, encoding="utf-8").read()
    i = texto.find(MARCA_DEL_OPTIMO)
    if i >= 0:
        texto = texto[:texto.rfind("# " + "=" * 77, 0, i)]
    return texto.rstrip()


CLAVES_ELEMENTO = ("barra", "nx", "ny", "longitud", "estribo", "separacion_estribo", "separacion_estribo_arriba")
CLAVES_MALLA = ("barra", "cortinas", "separacion")


def bloque_de_datos(bloques: dict, peso: float, bloques_red: dict | None,
                    corte: float | None) -> str:
    """El armado óptimo con el formato de los datos (pisa al de arriba).
    bloques: el armado de base; bloques_red: el reducido (None: sin reducido)."""
    sec = A.seccion()
    lineas = ["", "",
              "# " + "=" * 77,
              f"{MARCA_DEL_OPTIMO} el {date.today():%d/%m/%Y} (algoritmo",
              f"# genético, acero total {peso:.1f} kg). PISA AL ARMADO DE ARRIBA. Su informe va a",
              f"# resultados/{D.MURO}_optimo/ para no pisar el del muro.",
              "# " + "=" * 77,
              f'CARPETA_RESULTADOS = "{D.MURO}_optimo"']
    lineas.append(f"BORDE_IZQUIERDO = {_dic(bloques['izquierdo'], CLAVES_ELEMENTO)}")
    lineas.append(f"BORDE_DERECHO = {_dic(bloques['derecho'], CLAVES_ELEMENTO)}")
    lineas.append(f"BORDE_ALA = {_dic(bloques['ala'], CLAVES_ELEMENTO)}" if sec.tipo != "I"
                  else "BORDE_ALA = None")
    lineas.append(f"MALLA_VERTICAL = {_dic(bloques['malla_vertical'], CLAVES_MALLA)}")
    lineas.append(f"MALLA_HORIZONTAL = {_dic(bloques['malla_horizontal'], CLAVES_MALLA)}")
    if bloques_red is None:
        lineas.append("ARMADO_REDUCIDO = None        # el mismo armado en toda la altura")
        return "\n".join(lineas) + "\n"
    # el reducido, entero (las grapas también: si no, heredaría las de la base)
    lineas.append(f"# Etapa 2: el reducido. La cota de corte la calcula el programa (en la "
                  f"verificación del optimizador salió {corte:.2f} m).")
    lineas.append("ARMADO_REDUCIDO = {")
    for n in A.nombres_de_elementos():
        d = {**bloques_red[n], "grapas": bloques_red[n].get("grapas", "alternas")}
        lineas.append(f'    "borde_{n}": {_dic(d, CLAVES_ELEMENTO + ("grapas",))},')
    for malla in ("malla_vertical", "malla_horizontal"):
        lineas.append(f'    "{malla}": {_dic(bloques_red[malla], CLAVES_MALLA)},')
    lineas.append("}")
    return "\n".join(lineas) + "\n"


def resumen_de_arranques(out, r: dict) -> None:
    out("  Resumen de los arranques (el algoritmo genético y, después, la búsqueda local):")
    finales = []
    for k, a in enumerate(r["arranques"], 1):
        ag, fin = a["del_ag"], a["final"]
        finales.append(fin.peso if fin.cumple else math.inf)
        out(f"    {k}. semilla {a['semilla']}: {len(a['historia']):3d} generaciones, "
            + (f"AG {ag.peso:7.1f} kg → búsqueda local {fin.peso:7.1f} kg" if fin.cumple
               else "ningún armado cumple"))
    mejor_final = min(finales)
    if math.isfinite(mejor_final):
        llegan = sum(abs(p - mejor_final) < 0.05 for p in finales)
        out(f"    llegan al mejor ({mejor_final:.1f} kg): {llegan} de {len(finales)} arranques"
            + ("" if llegan == len(finales) else
               " (los otros quedan en óptimos locales: por eso se hacen varios arranques)"))


def cabecera(out, catalogo) -> None:
    espacio = math.prod(len(v) for _, v in catalogo)
    out(f"  Objetivo: el acero total en toda la altura (H = {A.H:.2f} m): vertical, malla horizontal "
        f"y estribos con grapas y ganchos, sin traslapes.")
    out("  Restricciones: todas las comprobaciones del programa (flexocompresión biaxial, sección")
    out("  crítica única, corte, junta de construcción, borde confinado, hx, separaciones, cuantías,")
    out("  separación libre, cota de corte del reducido…); el mejor se verifica al final con la resolución")
    out("  completa y en todas las cotas. Norma E.060.")
    out(f"  Criterio del proyecto: barras de los elementos confinados de {BARRA_MINIMA_BORDE} o más"
        + ("; planta simétrica (tipo I, o tipo II con el ala en medio): izquierdo = derecho."
           if simetrico() else "."))
    out({"las dos": "  Etapas: 1 (el armado de base) y, sobre su óptimo, 2 (el armado reducido).",
         "base": "  Etapa 1 sola: el mismo armado en toda la altura, sin reducido.",
         "reducido": "  Etapa 2 sola: el reducido, sobre el armado de base de los datos."}[ETAPAS])
    out("")
    out(f"  {len(catalogo)} genes por armado (los mismos catálogos en las dos etapas, salvo en la etapa 2 "
        f"la barra de las mallas, la de la base, y la longitud, sobre las columnas de barras de "
        f"la base); "
        f"{espacio:,.0f} armados posibles:".replace(",", " "))
    for (elemento, clave), valores in catalogo:
        texto = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in valores)
        out(f"    {elemento + ' · ' + clave:<34} {texto}")
    out(f"  Población {POBLACION}, hasta {GENERACIONES} generaciones (para si en {PARADA} no mejora), "
        f"cruce {CRUCE}, {MUTACION} genes mutados por hijo, élite {ELITE}; {max(1, ARRANQUES)} "
        f"arranque(s) independiente(s) (semillas {SEMILLA} a {SEMILLA + max(1, ARRANQUES) - 1}), "
        f"cada uno con búsqueda local final; superficie biaxial de búsqueda "
        f"{RESOLUCION_BIAXIAL[0]} × {RESOLUCION_BIAXIAL[1]}.")
    out("")


def etapa_1(out, catalogo, ok_d: bool, peso_d: float) -> dict:
    """La búsqueda del armado de base. {"r", "res", "bloques", "armado"}
    (res None si ninguno cumple la verificación completa)."""
    out.titulo("ETAPA 1 — ARMADO DE BASE (el mismo en toda la altura)")
    out("  Búsqueda:")
    r = optimizar(out, genes_del_armado_declarado(catalogo), None, "el armado declarado")
    out(f"  {len(r['ranking'])} armados evaluados en {r['tiempo']:.0f} s con {r['procesos']} proceso(s)")
    out("")
    resumen_de_arranques(out, r)
    out.titulo("ETAPA 1 — LOS MEJORES ARMADOS (verificación completa, resolución de los datos)")
    elegido = None
    vistos = 0
    textos = set()
    for res in r["ranking"]:
        if not res.cumple or vistos >= 8:
            break
        bloques = decodificar(res.genes, catalogo)
        armado = armado_de(bloques)
        texto = A.texto_del_armado(armado)
        if texto in textos:              # el mismo armado (genes que no cuentan, p. ej. un derecho absorbido)
            continue
        textos.add(texto)
        vistos += 1
        # el armado, con los datos tal como se escriben (el derecho absorbido,
        # como el ala): es la base que la etapa 2 continúa
        bloques = limpiar(bloques, armado)
        armado = armado_de(bloques)
        ok, fallas = verificacion_completa(armado)
        p = res.partes
        out(f"  {vistos}. {res.peso:8.1f} kg (vertical {p['vertical']:.1f}, malla horizontal "
            f"{p['horizontal']:.1f}, estribos {p['estribos']:.1f})  → "
            + ("CUMPLE" if ok else "no: " + "; ".join(fallas[:2])))
        out(f"     {texto}")
        if ok and elegido is None:
            elegido = (res, bloques, armado)
        if elegido is not None and vistos >= 5:
            break
    if elegido is None:
        out("  Ningún armado de la búsqueda cumple la verificación completa: amplía los catálogos "
            "(más barras por cara, longitudes, separaciones) o sube GENERACIONES.")
        return {"r": r, "res": None}
    res, bloques, armado = elegido
    out("")
    cambio = 100 * (res.peso / peso_d - 1)
    out(f"  ÓPTIMO DE LA ETAPA 1: {res.peso:.1f} kg, {abs(cambio):.1f} % "
        f"{'menos' if cambio < 0 else 'más'} que el declarado ({peso_d:.1f} kg"
        + (")" if ok_d else ", que no cumple)"))
    return {"r": r, "res": res, "bloques": bloques, "armado": armado}


def etapa_2(out, catalogo, bloques_base: dict, base, genes_base: tuple | None,
            peso_base: float) -> dict:
    """La búsqueda del reducido sobre la base fija. {"r", "bloques", "armado",
    "peso", "corte"} (bloques None si ningún reducido ahorra acero)."""
    out.titulo("ETAPA 2 — ARMADO REDUCIDO (encima de la cota de corte)")
    out(f"  Base fija: {A.texto_del_armado(base)}")
    out(f"  Sin reducido (la base en toda la altura): {peso_base:.1f} kg.")
    out("  Cada candidato es un reducido con los mismos genes y catálogos que la base, salvo la barra")
    out("  de cada malla, que es la de la base (criterio del proyecto: mismo diámetro en la base y en")
    out(f"  el reducido; malla vertical {bloques_base['malla_vertical']['barra']}, horizontal "
        f"{bloques_base['malla_horizontal']['barra']}), y la «longitud» confinada de")
    out("  cada extremo, que no pasa de la de la base (el borde confinado se decide en la base, §21.9.7.4,")
    out("  y arriba no se confina más) y termina en una columna de barras de la base, sin otro mínimo")
    longitudes = {n: A.longitudes_sobre_la_base(base.elemento(n), base) for n in ("izquierdo", "derecho")}
    out("  (§21.9.7.7 no pide longitud: " + "; ".join(f"{n}: " + ", ".join(f"{L:g}" for L in v) + " m"
                                                    for n, v in longitudes.items())
        + "). Cada barra vertical del reducido")
    out("  continúa una barra de la base, en su misma posición y de diámetro igual o menor (criterio del")
    out("  proyecto): en cada elemento, barra y barras por cara no mayores que en la base; la malla")
    out("  vertical va sobre líneas de barras de la base. La cota de corte la calcula el programa como en")
    out("  ejecutar.py (la mayor de §21.9.4.5(a) con §12.11.3, z* + 0.8·ℓm con el biaxial y la sección")
    out("  crítica única, y §21.9.7.4(b); y encima de donde el reducido falla por sí mismo) y se lleva a")
    out("  la losa más próxima por encima, donde se pide §12.11.5(a); el peso es el de la base hasta el")
    out("  corte más el del reducido encima. Un reducido cuyo corte cae en el techo no se puede usar:")
    out("  cuenta como que no cumple. En la búsqueda, el biaxial se mira cada")
    out(f"  {PASO_DE_COTAS_BUSQUEDA:.2f} m y en las losas, y el detallado del reducido en las caras de cada "
        "entrepiso;")
    out("  la verificación final lo mira todo.")
    out("")
    out("  Búsqueda:")
    r = optimizar(out, genes_en(catalogo_de_genes(bloques_base), genes_base, catalogo), bloques_base,
                  "el reducido igual a la base")
    catalogo = r["catalogo"]
    out(f"  {len(r['ranking'])} reducidos evaluados en {r['tiempo']:.0f} s con {r['procesos']} proceso(s)")
    out("")
    resumen_de_arranques(out, r)
    out.titulo("ETAPA 2 — LOS MEJORES REDUCIDOS (verificación completa, resolución de los datos)")
    out("  (la cota de corte y el peso, con todas las cotas: pueden diferir algo de los de la búsqueda)")
    texto_base = A.texto_del_armado(base)
    textos = set()
    verificados = []
    for res in r["ranking"]:
        if not res.cumple or len(textos) >= 8 or len(verificados) >= 5:
            break
        bloques = decodificar(res.genes, catalogo)
        red = armado_de(bloques, "reducido", base)
        texto = A.texto_del_armado(red)
        if texto in textos:
            continue
        textos.add(texto)
        if texto == texto_base:
            out(f"  {len(textos)}. el reducido igual a la base (no reduce nada): {peso_base:.1f} kg")
            continue
        ok, fallas, corte, partes = verificacion_del_reducido(base, red)
        out(f"  {len(textos)}. {partes['total']:8.1f} kg con corte a {corte:.2f} m (en la búsqueda "
            f"{res.peso:.1f} kg, {res.partes['corte']:.2f} m; vertical {partes['vertical']:.1f}, malla "
            f"horizontal {partes['horizontal']:.1f}, estribos {partes['estribos']:.1f})  → "
            + ("CUMPLE" if ok else "no: " + "; ".join(fallas[:2])))
        out(f"     reducido: {texto}")
        if ok:
            verificados.append((partes["total"], corte, bloques, red, partes))
    utiles = [v for v in verificados if v[0] < peso_base - 0.05]
    if not utiles:
        out("")
        out(f"  Ningún reducido verificado lleva menos acero que la base sola ({peso_base:.1f} kg): "
            f"no conviene reducir (ARMADO_REDUCIDO = None).")
        return {"r": r, "bloques": None, "armado": None, "peso": peso_base, "corte": None}
    peso, corte, bloques, red, partes = min(utiles, key=lambda v: v[0])
    out("")
    out(f"  ÓPTIMO DE LA ETAPA 2: {peso:.1f} kg con el reducido desde {corte:.2f} m "
        f"(gobierna: {partes['gobierna']}), {100 * (1 - peso / peso_base):.1f} % menos que la base "
        f"sola ({peso_base:.1f} kg)")
    return {"r": r, "bloques": limpiar(bloques, red), "armado": red, "peso": peso, "corte": corte}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    if ETAPAS not in ("las dos", "base", "reducido"):
        raise A.DatoInvalido(f'optimizar.py: ETAPAS es "las dos", "base" o "reducido"; es {ETAPAS!r}')
    out = ejecutar.Informe()
    out.titulo(f"OPTIMIZACIÓN DEL ARMADO — MURO {D.MURO}, TIPO {A.seccion().tipo} — algoritmo genético")
    preparar()
    catalogo = CONTEXTO["catalogo"]
    cabecera(out, catalogo)
    declarado = A.armado_de_base()
    ok_d, fallas_d = verificacion_completa(declarado)
    peso_d = peso_del_armado(declarado)
    out(f"  Armado de base declarado en los datos: {peso_d:.1f} kg en toda la altura  → "
        + ("cumple" if ok_d else "NO CUMPLE (" + "; ".join(fallas_d[:2]) + ")"))
    out(f"    {A.texto_del_armado(declarado)}")

    e1 = e2 = None
    if ETAPAS in ("las dos", "base"):
        e1 = etapa_1(out, catalogo, ok_d, peso_d)
        if e1["res"] is None:
            guardar(out, e1, None, None)
            return 1
        bloques_base, base = e1["bloques"], e1["armado"]
        genes_base, peso_base = e1["res"].genes, e1["res"].peso
    else:
        if not ok_d:
            out("")
            out("  El armado de base de los datos no cumple: la etapa 2 necesita una base que cumpla "
                "(corre antes la etapa 1).")
            guardar(out, None, None, None)
            return 1
        bloques_base, base = bloques_declarados(), declarado
        genes_base, peso_base = genes_del_armado_declarado(catalogo), peso_d
    if ETAPAS in ("las dos", "reducido"):
        e2 = etapa_2(out, catalogo, bloques_base, base, genes_base, peso_base)

    peso = e2["peso"] if e2 else peso_base
    out.titulo("RESULTADO")
    out(f"  Acero total: {peso:.1f} kg" + (f" (base {peso_base:.1f} kg sin reducido)"
                                           if e2 and e2["bloques"] else ""))
    out(f"  base:     {A.texto_del_armado(base)}")
    if e2 and e2["bloques"]:
        out(f"  reducido: {A.texto_del_armado(e2['armado'])}  (desde {e2['corte']:.2f} m)")
    ruta = guardar(out, e1, e2, (bloques_base, peso))
    if VERIFICAR_COMPLETO:
        out("")
        out(f"  Corriendo ejecutar.py con {ruta.name} (informe completo en resultados/{D.MURO}_optimo/)…")
        proceso = subprocess.run([sys.executable, str(ejecutar.CARPETA / "ejecutar.py"), str(ruta)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        lineas = proceso.stdout.splitlines()
        corte = next((linea.strip() for linea in lineas if linea.strip().startswith("CORTE A")), None)
        if corte:
            out(f"  {corte}")
        out("  " + next((linea.strip() for linea in lineas if "VEREDICTO" in linea),
                        proceso.stderr.strip()[-300:]))
        guardar(out, e1, e2, (bloques_base, peso))
    return 0


def guardar(out, e1: dict | None, e2: dict | None, armado: tuple | None):
    """El informe y las curvas en resultados/<MURO>/5_optimizacion/ y, si hay
    óptimo, datos_<MURO>_optimo.py junto al archivo de datos: los datos del muro
    (sin lo que optimizar.py les añadió antes) y el armado óptimo. armado:
    (bloques de la base, peso total)."""
    carpeta = F.carpeta_del_muro() / "5_optimizacion"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = None
    if armado is not None:
        bloques, peso = armado
        red = e2["bloques"] if e2 else None
        ruta = mef.carpeta_de_los_datos() / f"datos_{D.MURO}_optimo.py"
        ruta.write_text(datos_originales() + "\n" + bloque_de_datos(bloques, peso, red,
                                                                    e2["corte"] if e2 else None),
                        encoding="utf-8")
    (carpeta / f"informe_{D.MURO}_optimizacion.txt").write_text("\n".join(out.lineas) + "\n",
                                                                encoding="utf-8")
    if e1:
        curva(e1["r"]["arranques"], carpeta / f"convergencia_{D.MURO}.png",
              f"Muro {D.MURO} — etapa 1 (armado de base)", "Acero total del mejor armado (kg)")
    if e2:
        curva(e2["r"]["arranques"], carpeta / f"convergencia_reducido_{D.MURO}.png",
              f"Muro {D.MURO} — etapa 2 (armado reducido)", "Acero total con el reducido (kg)")
    return ruta


ESTILOS_DE_ARRANQUE = [("k", "-", "o"), ("0.35", "--", "^"), ("0.6", ":", "v"), ("0.2", "-.", "D"),
                       ("0.45", (0, (5, 1, 1, 1)), "s")]


def curva(arranques: list[dict], ruta, titulo: str, eje_y: str) -> None:
    """Por arranque, el peso del mejor armado que cumple en cada generación y,
    al final, el de la búsqueda local (cuadro blanco)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with plt.rc_context(F.ESTILO_FIGURAS):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        sin_factible = []
        for k, a in enumerate(arranques):
            color, linea, marca = ESTILOS_DE_ARRANQUE[k % len(ESTILOS_DE_ARRANQUE)]
            historia = a["historia"]
            gen = [h["generacion"] for h in historia]
            final = a["final"].peso if a["final"].cumple else math.nan
            ax.plot(gen, [h["mejor"] for h in historia], color=color, ls=linea, lw=1.2, marker=marca,
                    ms=2.5, label=f"arranque {k + 1} (semilla {a['semilla']}): {final:.1f} kg")
            if a["pasos_locales"] and gen:
                ax.plot([gen[-1], gen[-1] + 1], [historia[-1]["mejor"], final], color=color, lw=0.9,
                        ls=":")
            if gen:
                ax.plot([gen[-1] + 1], [final], ls="none", marker="s", ms=5, mfc="white", mec=color)
            primera = next((h["generacion"] for h in historia if not math.isnan(h["mejor"])), None)
            if primera and primera > 1:
                sin_factible.append(f"{k + 1}: hasta la {primera - 1}")
        ax.legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="0.6",
                  title="□ = tras la búsqueda local", title_fontsize=7.5)
        ax.set_xlabel("Generación")
        ax.set_ylabel(eje_y)
        ax.grid(True, color="0.82", ls=":", lw=0.6)
        if sin_factible:
            ax.text(0.0, -0.2, "Generaciones sin ningún armado que cumpla (no se dibujan), por arranque: "
                    + "; ".join(sin_factible) + ".",
                    transform=ax.transAxes, fontsize=7.5, style="italic", va="top")
        fig.suptitle(titulo, fontsize=11)
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except A.DatoInvalido as error:
        print(f"\nDATO INVÁLIDO en {getattr(D, '__file__', 'datos')}: {error}", file=sys.stderr)
        sys.exit(2)
