"""Diseño del muro con la Norma E.060: MEF + flexocompresión + corte + borde.

Qué muro se corre: el de MURO_A_EJECUTAR, aquí abajo (1, 2 o 3), con el
botón ▷ de VS Code o con `python ejecutar.py`. Desde la terminal también:
    python ejecutar.py edificacion_01/datos_P11.py    (manda sobre MURO_A_EJECUTAR)

Lee los datos, resuelve el MEF, hace el diseño en los tres módulos y escribe
el informe en la consola y en resultados/<MURO>/, en la carpeta de la
edificación (la del archivo de datos): el informe completo, informe_<MURO>.txt,
y una carpeta por módulo con su parte del informe y sus figuras PNG (si
matplotlib está instalado).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True      # sin carpetas __pycache__ junto a los archivos

# =============================================================================
# QUÉ MURO SE CORRE: escribe 1, 2 o 3 y pulsa ▷ (Run Python File) en VS Code.
#     1 → edificacion_01/datos_P17.py   (tipo I)
#     2 → edificacion_01/datos_P11.py   (tipo II)
#     3 → edificacion_01/datos_P1.py    (tipo III)
# Para otro muro, escribe su archivo entre comillas, con la carpeta de su
# edificación, p. ej. MURO_A_EJECUTAR = "edificacion_02/datos_M5.py". Es lo
# único que se cambia en este archivo.
# =============================================================================
MURO_A_EJECUTAR = 1

CARPETA = Path(__file__).resolve().parent
ARCHIVOS_POR_NUMERO = {1: "edificacion_01/datos_P17.py", 2: "edificacion_01/datos_P11.py",
                       3: "edificacion_01/datos_P1.py"}
DATOS_POR_DEFECTO = ARCHIVOS_POR_NUMERO[1]


def archivo_a_ejecutar() -> str:
    """El archivo de datos: el que se escribe en la terminal o, si no, el de
    MURO_A_EJECUTAR (1, 2, 3 o el nombre de un archivo)."""
    if len(sys.argv) >= 2:
        return sys.argv[1]
    elegido = MURO_A_EJECUTAR
    if isinstance(elegido, str) and elegido.strip():
        return elegido.strip()
    if not isinstance(elegido, bool) and elegido in ARCHIVOS_POR_NUMERO:
        return ARCHIVOS_POR_NUMERO[elegido]
    _parar("ejecutar.py", f"MURO_A_EJECUTAR es 1, 2, 3 o el nombre de un archivo de datos entre "
                          f"comillas; es {elegido!r}")


def _parar(donde, mensaje: str) -> None:
    """Sale con el código 2 (dato inválido) y un mensaje claro, sin traceback."""
    print(f"\nDATO INVÁLIDO en {donde}: {mensaje}", file=sys.stderr)
    sys.exit(2)


def _cargar_datos() -> None:
    """Carga el archivo de datos (ver `archivo_a_ejecutar`) como el módulo
    `datos` antes de importar los cálculos (que hacen `import datos`)."""
    nombre = archivo_a_ejecutar()
    ruta = Path(nombre)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = CARPETA / ruta
    if not ruta.exists():
        _parar(nombre, "no existe ese archivo de datos (se busca también junto a ejecutar.py)")
    ruta = ruta.resolve()
    spec = importlib.util.spec_from_file_location("datos", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["datos"] = modulo
    try:
        spec.loader.exec_module(modulo)
    except SyntaxError as error:           # un paréntesis sin cerrar, una coma de más…
        _parar(ruta, f"error de escritura en la línea {error.lineno}: {error.msg}")
    except Exception as error:             # un nombre mal escrito, una cuenta imposible…
        linea = next((t.lineno for t in reversed(traceback.extract_tb(error.__traceback__))
                      if Path(t.filename) == ruta), "?")
        _parar(ruta, f"error en la línea {linea}: {type(error).__name__}: {error}")
    problema = _problema_basico(modulo)
    if problema:
        _parar(ruta, problema)


def _problema_basico(d) -> str:
    """Lo mínimo que tienen que traer los datos para que los cálculos se
    puedan cargar (usan estos valores al empezar). Devuelve el problema, o ""
    si no hay. El resto se comprueba después (`validar_datos`)."""

    def numero(nombre, minimo=0.0, incluido=False):
        v = getattr(d, nombre, None)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (
                v >= minimo if incluido else v > minimo):
            return (f"{nombre} tiene que ser un número {'≥' if incluido else '>'} {minimo:g}; "
                    f"es {v!r}")
        return ""

    for nombre in ("FC", "FY", "ES", "EC", "LW", "ESPESOR", "DELTA_U_MM", "R0", "MALLA_MEF",
                   "PASO_DE_COTAS"):
        problema = numero(nombre)
        if problema:
            return problema
    if d.FC < 17.0:
        return (f"FC = {d.FC:.2f} MPa: §9.4.1 pide f'c ≥ 17 MPa (y 21 MPa en los elementos que "
                f"resisten sismo, §21.3.2.1)")
    if numero("NU", 0.0, incluido=True) or d.NU >= 0.5:
        return f"NU (Poisson del concreto) va de 0 a 0.5; es {getattr(d, 'NU', None)!r}"
    problema = numero("AXIS_ANGLE", -360.0, incluido=True)
    if problema:
        return problema
    for nombre, minimo in (("PUNTOS_DEL_DIAGRAMA", 10), ("ANGULOS_BIAXIAL", 8),
                           ("PROFUNDIDADES_BIAXIAL", 10)):
        v = getattr(d, nombre, None)
        if isinstance(v, bool) or not isinstance(v, int) or v < minimo:
            return f"{nombre} es un entero ≥ {minimo}; es {v!r}"
    pisos, alturas = getattr(d, "PISOS", None), getattr(d, "ALTURAS", None)
    if (not isinstance(pisos, (list, tuple)) or not pisos
            or not all(isinstance(p, str) and p for p in pisos) or len(set(pisos)) != len(pisos)):
        return f"PISOS es una lista de nombres de piso de ETABS, distintos, de abajo arriba; es {pisos!r}"
    if (not isinstance(alturas, (list, tuple)) or len(alturas) != len(pisos)
            or any(isinstance(h, bool) or not isinstance(h, (int, float)) or h <= 0 for h in alturas)):
        return (f"ALTURAS es una lista de {len(pisos)} alturas positivas (una por piso de PISOS); "
                f"es {alturas!r}")
    for nombre in ("MURO", "ARCHIVO_FUERZAS", "HOJA_FUERZAS"):
        v = getattr(d, nombre, None)
        if not isinstance(v, str) or not v.strip():
            return f"{nombre} es un texto; es {v!r}"
    if not (Path(d.__file__).resolve().parent / d.ARCHIVO_FUERZAS).exists():
        return (f"no existe el archivo de ETABS {d.ARCHIVO_FUERZAS} (se busca junto al archivo "
                f"de datos)")
    if not isinstance(getattr(d, "DIBUJAR", None), bool):
        return f"DIBUJAR es True o False; es {getattr(d, 'DIBUJAR', None)!r}"
    otra = getattr(d, "CARPETA_RESULTADOS", None)          # opcional
    if otra is not None and (not isinstance(otra, str) or not otra.strip()
                             or any(c in otra for c in '\\/:*?"<>|')):
        return f"CARPETA_RESULTADOS es un nombre de carpeta (sin \\ / : * ? \" < > |); es {otra!r}"
    return ""


_cargar_datos()

import numpy as np  # noqa: E402

import datos as D  # noqa: E402
import diseno_e060 as A  # noqa: E402
import mef  # noqa: E402
import figuras as F  # noqa: E402
from figuras import (DEMANDA, MODULO_1, MODULO_2, MODULO_3,  # noqa: E402
                     VERIFICACION, carpeta_del_muro)

ANCHO = 92


class Informe:
    """Imprime cada línea y la guarda para los archivos .txt: el informe
    completo y, en la carpeta de cada parte, su tramo."""

    def __init__(self) -> None:
        self.lineas: list[str] = []
        self.partes: list[tuple[str, int]] = [(DEMANDA, 0)]   # (carpeta, primera línea)

    def __call__(self, texto: str = "") -> None:
        print(texto)
        self.lineas.append(texto)

    def parte(self, carpeta: str) -> None:
        """Desde aquí, las líneas van también a la carpeta `carpeta`."""
        self.partes.append((carpeta, len(self.lineas)))

    def titulo(self, texto: str) -> None:
        self("")
        self("=" * ANCHO)
        self(texto)
        self("=" * ANCHO)

    def modulo(self, texto: str) -> None:
        self("")
        self("#" * ANCHO)
        self(f"# {texto}")
        self("#" * ANCHO)

    def guardar(self, carpeta: Path) -> list[Path]:
        """El informe completo en `carpeta` y el tramo de cada parte en su
        subcarpeta (con el título del muro arriba). Devuelve las rutas."""
        carpeta.mkdir(parents=True, exist_ok=True)
        completo = carpeta / f"informe_{D.MURO}.txt"
        completo.write_text("\n".join(self.lineas) + "\n", encoding="utf-8")
        rutas = [completo]
        cabecera = self.lineas[:4]                # el título: MURO …, TIPO …
        limites = [i for _, i in self.partes[1:]] + [len(self.lineas)]
        for (nombre, desde), hasta in zip(self.partes, limites):
            sub = carpeta / nombre
            sub.mkdir(exist_ok=True)
            ruta = sub / f"informe_{D.MURO}_{nombre.split('_', 1)[1]}.txt"
            tramo = self.lineas[desde:hasta] if desde > 0 else self.lineas[:hasta]
            ruta.write_text("\n".join((cabecera if desde > 0 else []) + tramo) + "\n",
                            encoding="utf-8")
            rutas.append(ruta)
        return rutas


def si_no(cumple: bool) -> str:
    return "CUMPLE" if cumple else "NO CUMPLE"


NOMBRE_DEL_TIPO = {"I": "rectangular", "II": "con ala corta en medio",
                   "III": "con ala corta en el extremo derecho"}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    t0 = time.perf_counter()
    out = Informe()

    # =========================================================================
    # DATOS (todo lo que se puede comprobar, antes de escribir nada)
    # =========================================================================
    A.validar_datos()
    mef.ejes()
    sec = A.seccion()
    base = A.armado_de_base()
    red = A.armado_reducido(base)
    for arm in (base, red):
        if arm is not None:
            A.colocar_barras(arm)
    out.titulo(f"MURO {D.MURO} — TIPO {sec.tipo} ({NOMBRE_DEL_TIPO[sec.tipo]}) — "
               f"NORMA E.060, muro estructural (§21.9)")
    out(f"  f'c = {D.FC:.2f} MPa ({D.FC / 0.0980665:.0f} kgf/cm²)   fy = {D.FY:.2f} MPa "
        f"({D.FY / 0.0980665:.0f} kgf/cm²)   Es = {D.ES:.0f} MPa   Ec = {D.EC:.0f} MPa (§8.5.2)")
    for aviso in A.avisos_de_datos():
        out(f"  AVISO: {aviso}")
    out(f"  β1 = {A.BETA1:.3f} (§10.2.7.3)   εcu = {A.EPS_CU} (§10.2.3)   "
        f"εy = fy/Es = {A.EPS_Y:.5f}")
    out(f"  ℓm = {sec.lw:.3f} m   b = {sec.b:.3f} m   Ag = {sec.area:.4f} m²   "
        f"Acw = ℓm·b = {sec.acw:.4f} m²   hm = {A.H:.2f} m "
        f"({' + '.join(f'{h:.2f}' for h in D.ALTURAS)})   hm/ℓm = {A.ESBELTEZ:.2f}")
    if sec.tipo != "I":
        out(f"  ala: b' = {sec.ancho_ala:.3f} m, l'w = {sec.largo_ala:.3f} m (desde la cara exterior "
            f"del alma), de x = {sec.ala.x0:.3f} a {sec.ala.x1:.3f} m")
    out(f"  Recubrimiento libre {D.RECUBRIMIENTO * 100:.1f} cm; malla horizontal "
        f"{'por fuera' if D.HORIZONTAL_POR_FUERA else 'por dentro'} de la vertical; "
        + (f"agregado {D.AGREGADO_MM:.0f} mm" if getattr(D, "AGREGADO_MM", None)
           else "agregado no declarado (§3.3.2(c) no se comprueba)"))
    out(f"  R0 = {D.R0:g} (E.030) → tope del cociente Mn/Mua: 0.5·R0 = {0.5 * D.R0:.2f} (§21.9.5.5);   "
        f"juntas de construcción con μ = {D.MU_JUNTA:g} (§11.7.4.3)")

    seccion_y_ejes(out, sec)

    out.titulo("ARMADO")
    armados = [x for x in (base, red) if x is not None]
    for arm in armados:
        out(f"  {arm.nombre.upper():<9} {arm.texto()}   → {A.kg_por_metro(arm):.2f} kg/m vertical")
        out(f"  {'':<9} malla horizontal {arm.malla_horizontal.texto()}, ρh = "
            f"{arm.malla_horizontal.cuantia:.5f}")
    if red is not None:
        out("  El reducido va encima de la cota de corte (1.6): cambia el acero vertical, el horizontal")
        out("  (mallas y estribos) y, si se declara, la longitud confinada de los extremos (figura")
        out(f"  planta_reducido_{D.MURO}.png). Criterio del proyecto: cada barra vertical del reducido")
        out("  continúa (empalma con) una barra de la base, en su misma posición y de diámetro igual o")
        out("  menor; su malla vertical toma una de cada k líneas de barras de la base, con k el mayor que")
        out("  no pasa su separación (que es, por eso, una separación máxima).")
        for aviso in A.avisos_del_reducido(base, red):
            out(f"  AVISO: {aviso} (no es un error, pero no es una reducción)")
    for arm in armados:
        tabla_de_elementos(out, arm)
    for arm in armados:
        tabla_de_barras(out, arm, A.colocar_barras(arm))

    # =========================================================================
    # DEMANDA: MEF Y CONTRASTE CON ETABS
    # =========================================================================
    out.titulo("DEMANDA: MEF DEL MURO Y CONTRASTE CON ETABS")
    crudas = mef.leer_fuerzas_etabs()            # tal como las da ETABS (V2, V3, M2, M3)
    fuerzas = mef.mapear(crudas)
    res = mef.resolver(fuerzas)
    m = res.malla
    alt = np.diff(m.zs)
    out(f"  {len(fuerzas)} estados de ETABS: " + ", ".join(f"{c} {p}".strip() for c, p in fuerzas))
    out(f"  Malla: {res.n_elementos} elementos Q4 de tensión plana en el alma "
        f"({m.nx - 1} a lo largo × {len(m.zs) - 1} en altura), "
        f"{np.diff(m.xs).max():.3f} m × {alt.min():.3f}–{alt.max():.3f} m; "
        f"{len(m.zs)} planos de nudos; empotrado en la base")
    out(f"  Fuerzas en los ejes del programa: {texto_de_ejes()}  (M < 0 comprime la cara +X / +Y)")
    ejemplo = next(k for k in fuerzas if k[0] != "Gravitacional")
    bas = [(fuerzas[ejemplo][s]["Bottom"]["P"], fuerzas[ejemplo][s]["Bottom"]["Vx"],
            fuerzas[ejemplo][s]["Bottom"]["Mx"]) for s in range(len(D.PISOS))]
    top = [fuerzas[ejemplo][s]["Top"]["P"] for s in range(len(D.PISOS))]
    peso = mef.con_peso_propio()
    out(f"  Cargas en las losas, {ejemplo[0]} {ejemplo[1]} (i: base del entrepiso i; tope,i: su tope):")
    if peso:
        out("    H = V_i − V_(i+1)    N = P_tope,i − P_(i+1)    C = (M_i − M_(i+1)) + h_i·V_i")
        out("    y el peso propio del entrepiso, W = P_i − P_tope,i, repartido en sus planos interiores")
        cargas = mef.cargas_por_losa(bas, top)
        pesos = mef.pesos_propios(bas, top)
    else:
        out("    H = V_i − V_(i+1)    N = P_i − P_(i+1)    C = (M_i − M_(i+1)) + h_i·V_i")
        out("    (AXIAL_CON_PESO_PROPIO = False: el peso de cada entrepiso queda en la losa de arriba)")
        cargas, pesos = mef.cargas_por_losa(bas), [0.0] * len(bas)
    for s, ((h, n, c), w) in enumerate(zip(cargas, pesos)):
        out(f"    losa en z = {m.zs[m.losas[s]]:5.2f} m:  H = {h:8.1f} kN   N = {n:8.1f} kN   "
            f"C = {c:8.1f} kN·m" + (f"   (peso del entrepiso {D.PISOS[s]}: W = {w:6.1f} kN)"
                                     if peso else ""))

    con = mef.contraste_con_etabs(res, fuerzas)
    out("")
    out("  Fuerzas de sección en el plano: ETABS frente al MEF (kN, kN·m; P negativo en compresión)")
    cab = (f"{'P ETABS':>10}{'P MEF':>10}{'Vx ETABS':>10}{'Vx MEF':>10}{'Mx ETABS':>10}{'Mx MEF':>10}")
    out("  En la BASE de cada entrepiso, todos los estados:")
    out(f"  {'estado':<22}{'piso':>7}{cab}")
    for clave, s, et, me in con["base"]:
        out(f"  {(clave[0] + ' ' + clave[1]).strip():<22}{D.PISOS[s]:>7}"
            f"{et[0]:10.1f}{me[0]:10.1f}{et[1]:10.1f}{me[1]:10.1f}{et[2]:10.1f}{me[2]:10.1f}")
    pe = con["peor"]
    out(f"  Mayor diferencia en las bases: P {pe['P']:.3f} kN, V {pe['V']:.3f} kN, "
        f"M {pe['M']:.3f} kN·m  → el MEF devuelve las fuerzas de ETABS.")
    out("")
    out("  En el TOPE de cada entrepiso, la parte ESTÁTICA: la gravitacional y la parte de gravedad")
    out("  de cada envolvente (la media de Max y Min), que cumplen M_tope = M_base + V·h:")
    out(f"  {'estado':<22}{'piso':>7}{cab}")
    for nombre, s, et, me in con["tope"]:
        out(f"  {nombre:<22}{D.PISOS[s]:>7}"
            f"{et[0]:10.1f}{me[0]:10.1f}{et[1]:10.1f}{me[1]:10.1f}{et[2]:10.1f}{me[2]:10.1f}")
    pt = con["peor_tope"]
    out(f"  Mayor diferencia en los topes: P {pt['P']:.3f} kN, V {pt['V']:.3f} kN, "
        f"M {pt['M']:.3f} kN·m" + (" (redondeo de ETABS)" if not pisos_con_cargas_dentro(fuerzas) else ""))
    nota_de_cargas_dentro(out, fuerzas)
    if peso:
        out("  (P coincide porque el peso propio se carga con la diferencia de ETABS; lo que comprueba el")
        out("  tope es que V y M salen por estática.)")
    else:
        out("  P no coincide: con AXIAL_CON_PESO_PROPIO = False el peso del entrepiso queda en la losa")
        out("  de arriba y el MEF lleva en el tope el P de la base.")
    out("")
    out("  En el TOPE, la parte de SISMO de cada envolvente (la semidiferencia de Max y Min, en valor")
    out("  absoluto). Es una combinación modal (SRSS o CQC) y no cumple la estática: cargando el MEF con")
    out("  la fila de la base, el tope sale |S_base − h·S_V|. Como el momento de cada modo es una recta")
    out("  en el entrepiso, ETABS tiene que caer entre |S_base − h·S_V| y S_base + h·S_V, y dentro vale")
    out("      S(ζ)² = (1 − ζ)·S_base² + ζ·S_tope² − ζ·(1 − ζ)·h²·S_V²,   ζ = (z − z_base)/h")
    out("  exacta para SRSS y CQC (ver mef.py). El cortante y el axial de sismo no cambian en el entrepiso.")
    out(f"  {'':<21}{'Mx de sismo':^45}{'a media altura':^27}")
    out(f"  {'combinación':<14}{'piso':>7}{'base':>9}{'tope':>9}{'estát.':>9}{'cotas de ETABS':>18}"
        f"{'estát.':>9}{'exacta':>9}{'recta':>9}{'Vx b/t':>14}  coherente")
    no_coherente = "no: recta" if mef.sismo_entre_base_y_tope() else "no"
    for f in con["sismo"]:
        b, t, e = f["base"], f["tope"], f["estatica"]
        me, mx, mr = f["medio"]
        out(f"  {f['combo']:<14}{D.PISOS[f['piso']]:>7}{abs(b[2]):9.1f}{abs(t[2]):9.1f}{abs(e[2]):9.1f}"
            f"{f['inferior']:9.1f}–{f['superior']:<8.1f}{me:9.1f}{mx:9.1f}{mr:9.1f}"
            f"{abs(b[1]):7.1f}/{abs(t[1]):<6.1f}  {'sí' if f['coherente'] else no_coherente}")
    if mef.sismo_entre_base_y_tope():
        out("  Dentro de cada entrepiso, la parte de sismo de Mx y My sale de esa fórmula; la de gravedad, del")
        out("  MEF. Donde el tope no es coherente («no: recta») se usa la recta entre base y tope, del lado")
        out("  seguro. Es un aviso de cómo se lee ETABS, no una verificación: no entra en el veredicto.")
        out("  (SISMO_ENTRE_BASE_Y_TOPE = True)")
    else:
        out("  SISMO_ENTRE_BASE_Y_TOPE = False: la parte de sismo sale por estática desde la base: es la")
        out("  cota inferior y subestima el sismo donde el momento del modo dominante cambia de signo.")
    out("  (La dirección Y, fuera del plano, se contrasta abajo, después del traslado al centroide.)")

    if D.DIBUJAR:
        out("  La envolvente total de Mx, Vx, My y Vy en altura (el máximo y el mínimo de todos los estados,")
        out(f"  acotada en cada piso, con los momentos como los da ETABS): figuras envolventes_X_{D.MURO}.png")
        out(f"  y envolventes_Y_{D.MURO}.png (abajo, frente a ETABS).")

    traslado = (sec.xe - sec.xc, sec.ye - sec.yc)
    dem = mef.Demanda(res, fuerzas, traslado)
    dis = A.Diseno(dem, D.DELTA_U_MM / 1000.0, base=base)
    out("")
    if any(abs(t) > 1e-12 for t in traslado):
        out("  Momentos llevados al centroide de la sección real (ver SECCIÓN Y EJES):")
        out(f"    Mx = Mx_ETABS + P·({sec.xe:.4f} − {sec.xc:.4f}) = Mx_ETABS + P·({con_signo(traslado[0])})   "
            f"My = My_ETABS + P·({sec.ye:.4f} − {sec.yc:.4f}) = My_ETABS + P·({con_signo(traslado[1])})")
    contraste_en_y(out, dem, crudas, fuerzas, traslado[1])
    envolventes_frente_a_etabs(out, dis, fuerzas, *traslado)
    out("")
    out(f"  Desplazamiento de diseño del techo en el plano del muro (archivo de datos): δu = "
        f"{D.DELTA_U_MM:.2f} mm, δu/hm = {dis.deriva:.5f}")
    est0 = dis.estados(0.0, 0)
    out("")
    if getattr(D, "PERMUTAR_SIGNOS", True):
        out("  ETABS da las combinaciones con sismo como ENVOLVENTES: cada componente de la fila Max")
        out("  es su propio máximo, no el de un mismo instante. Como hace ETABS en su diseño con")
        out("  espectros, se separa la parte de gravedad (media de Max y Min) y la de sismo (la")
        out("  semidiferencia) y se prueban los 8 signos de la de sismo en P, Mx y My: «Comb-2 +−+»")
        out("  lleva P y My como en la fila Max y Mx como en la Min (V va con Mx y Vy con My).")
        out("  «+++» es la fila Max y «−−−» la Min. (PERMUTAR_SIGNOS en el archivo de datos.)")
    else:
        out("  PERMUTAR_SIGNOS = False: solo las filas Max y Min de ETABS, tal como vienen.")
    out("  Las combinaciones son las de ETABS (E.060 §9.2: 1.4CM + 1.7CV, 1.25(CM + CV) ± CS y")
    out("  0.9CM ± CS), tal como vienen.")
    out("")
    out("  Demanda en la BASE (z = 0): Vua es el cortante del análisis y Vu el de diseño (2.1):")
    out(f"  {'estado':<18}{'Pu':>9}{'Mux':>10}{'Vua':>9}{'Vu':>9}{'Muy':>10}")
    for e in est0:
        out(f"  {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{e.V:9.1f}{e.Vu:9.1f}{e.My:10.2f}")
    out("  (kN y kN·m; x = en el plano del alma, y = a través del espesor)")

    # =========================================================================
    # MÓDULO 1 · FLEXOCOMPRESIÓN
    # =========================================================================
    out.parte(MODULO_1)
    out.modulo("MÓDULO 1 · DISEÑO A FLEXOCOMPRESIÓN")
    # el elemento de borde se decide aquí, en la base, con el armado de base,
    # y rige en las demás cotas
    vb = dis.verificar(base, 0.0, 0)
    b0 = dis.barras(base)
    # la cota de corte se calcula ya (se explica en 1.6): dice en qué
    # entrepisos va cada armado
    corte = A.H
    cc = None
    if red is not None:
        cc = A.cota_de_corte(dis, base, red, vb.dc_biaxial)
        corte = cc.corte
    # las cuantías mínimas de §11.10 dependen del cortante: con el entrepiso
    # de mayor |Vu| de los que lleva cada armado
    refuerzos = {arm.nombre: refuerzo_de(dis, arm, pisos_del_armado(dis, arm, corte))
                 for arm in armados}

    out.titulo("1.1 CORTINAS DE REFUERZO — §21.9.4.3 y §11.10.10.5")
    vu = max(abs(e.Vu) for e in est0)
    umbral = 0.17 * A.RAIZ_FC * mef.MPA * sec.acw
    out("  Dos capas si el espesor es ≥ 200 mm o si Vu > 0.17·√f'c·Acw (basta una). Vu es el")
    out("  cortante de diseño (§21.9.5.5):")
    out(f"    espesor = {D.ESPESOR * 1000:.0f} mm  frente a 200 mm  → "
        f"{'SÍ' if D.ESPESOR >= 0.200 - 1e-9 else 'no'}")
    out(f"    Vu = {vu:.1f} kN  frente a  0.17·√{D.FC:.2f}·{sec.acw:.3f}·1000 = {umbral:.1f} kN  → "
        f"{'SÍ' if vu > umbral else 'no'}")
    exige2 = D.ESPESOR >= 0.200 - 1e-9 or vu > umbral
    for arm in armados:
        mv, mh = arm.malla_vertical, arm.malla_horizontal
        out(f"  Declaradas, {arm.nombre:<9}: {mv.cortinas} vertical, {mh.cortinas} horizontal  → "
            f"{si_no(not exige2 or (mv.cortinas >= 2 and mh.cortinas >= 2))}")

    cuantia_longitudinal(out, dis, base, red, refuerzos)

    out.titulo("1.3 DIAGRAMA DE INTERACCIÓN P-Mx EN LA BASE (REFERENCIA) — §10.2, §10.3 y φ de §9.3.2.2")
    out("  La flexocompresión se verifica con el D/C BIAXIAL (1.4), en cada nivel. Este diagrama, en el")
    out("  plano del muro (eje neutro paralelo a Y), queda de referencia: da φ, Pb y el equilibrio")
    out("  para comprobar a mano, y su D/C no entra en el veredicto.")
    diag = dis.diagrama(base)
    p0 = A.p0_nominal(b0)
    out(f"  As = {b0.As * 1e4:.2f} cm²   P0 = 0.85·f'c·(Ag − Ast) + fy·Ast = {p0:.1f} kN   "
        f"tope 0.80·P0 = {0.8 * p0:.1f} kN (§10.3.6.2)")
    out(f"  Tracción pura: fy·As = {D.FY * mef.MPA * b0.As:.1f} kN.")
    out("  φ (§9.3.2.2): 0.90 en tracción y flexión pura; 0.70 en compresión, que sube en recta a 0.90")
    out("  cuando φPn baja de P_lím = mín(0.1·f'c·Ag, φPb) a cero: φ = 0.90/(1 + 0.20·Pn/P_lím).")
    out(f"    0.1·f'c·Ag = {0.1 * D.FC * mef.MPA * sec.area:.1f} kN;  Pb = {-diag.p_bal[1]:.1f} kN "
        f"(sentido +1) y {-diag.p_bal[-1]:.1f} kN (sentido −1), con c_b = dt·εcu/(εcu + εy) (§10.3.2)")
    out(f"    → P_lím = {diag.p_lim[1]:.1f} kN (sentido +1) y {diag.p_lim[-1]:.1f} kN (sentido −1)")
    out(f"  c de 0.02·ℓm a 4·ℓm en {D.PUNTOS_DEL_DIAGRAMA} pasos geométricos, por sentido; eje "
        f"neutro paralelo a Y. Sentido +1 comprime el extremo derecho (M < 0). Momentos")
    out(f"  respecto del centroide de la sección real (x = {sec.xc:.4f} m).")
    out(f"  {'#':>3}{'sent.':>6}{'c (m)':>8}{'Pn':>10}{'Mn':>10}{'εt':>9}{'φ':>7}"
        f"{'φPn':>10}{'φMn':>10}")
    for i, p in enumerate(diag.puntos):
        sentido = "" if math.isinf(p.c) else f"{int(p.n[0]):+d}"
        c = "∞" if math.isinf(p.c) else f"{p.c:.4f}"
        eps = "—" if math.isinf(p.eps_t) else f"{p.eps_t:.5f}"
        out(f"  {i:3d}{sentido:>6}{c:>8}{p.P:10.1f}{p.Mx:10.1f}{eps:>9}{p.phi:7.3f}"
            f"{p.phi * p.P:10.1f}{p.phi * p.Mx:10.1f}")
    out("  (kN y kN·m; P negativo en compresión)")

    out("")
    out("  D/C en el plano por estado en la base (referencia). D/C = cuánto hay que escalar (Pu, Mu)")
    out("  desde el origen hasta tocar la curva de diseño (medida radial, con P y M normalizados).")
    out(f"  {'estado':<18}{'Pu':>9}{'Mu':>10}{'φMn a Pu':>11}{'D/C':>8}")
    for e in est0:
        mn = A.phi_Mn(diag, e.P, 1.0 if e.M > 0 else -1.0)
        out(f"  {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{mn:11.1f}{A.dc_radial(diag, e.P, e.M):8.3f}")
    eg = vb.estado_flexion
    out(f"  El mayor en el plano: {eg.nombre}, D/C = {vb.dc_flexion:.3f} (referencia; el que se "
        f"verifica es el biaxial, {vb.dc_biaxial:.3f})")
    equilibrio(out, dis, base, eg)

    out.titulo("1.4 FLEXOCOMPRESIÓN BIAXIAL EN CADA NIVEL — §21.9.6.1 y §10.2 (EL D/C QUE SE VERIFICA)")
    out(f"  Eje neutro girado {D.ANGULOS_BIAXIAL} veces (cada {360 / D.ANGULOS_BIAXIAL:.0f}°) "
        f"con {D.PROFUNDIDADES_BIAXIAL} profundidades; zona comprimida exacta (polígonos);")
    out("  φ de §9.3.2.2 con el Pb de cada giro. A cada Pu se corta la superficie: D/C = cuánto hay")
    out("  que escalar (Mux, Muy) hasta el contorno. En la base, todos los estados:")
    sup = dis.superficie(base)
    out(f"  {'estado':<18}{'Pu':>9}{'Mux':>10}{'Muy':>9}{'D/C':>8}")
    for e in est0:
        out(f"  {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{e.My:9.2f}{sup.dc(e.P, e.M, e.My):8.3f}")
    out(f"  GOBIERNA {vb.estado_biaxial.nombre}: D/C = {vb.dc_biaxial:.3f}  → "
        f"{si_no(vb.dc_biaxial <= 1.0)}")
    biaxial_por_nivel(out, dis, base, red, corte)

    flexion_minima(out, dis, base, vb)

    out.titulo("1.6 COTA DE CORTE DEL ARMADO REDUCIDO — §21.9.4.5(a), §12.11.3, §21.9.7.4(b) y §12.11.5")
    if red is None:
        out("  No se declaró armado reducido: el acero de base va en toda la altura.")
    else:
        cotas = dis.cotas()
        out("  El reducido empieza en el entrepiso inmediatamente encima de la MAYOR de las cotas de la")
        out("  norma (criterio del proyecto) y encima de donde falla por sí mismo. Sus barras verticales")
        out("  continúan barras de la base (criterio del proyecto, ver ARMADO). La E.060 no tiene el acero")
        out("  mínimo de extremo del ACI (§18.10.2.4), así que esa cota no entra.")
        out("")
        out("  (1) §21.9.4.5(a) y §12.11.3, ACERO VERTICAL. Se busca z*: la cota MÁS BAJA desde la que el")
        out(f"  reducido cumple la flexocompresión biaxial en todo lo que hay encima. Se baja desde el techo "
            f"por {len(cotas)} cotas (cada {D.PASO_DE_COTAS:.2f} m")
        out("  como máximo, más los planos del MEF) y el cruce se interpola. El acero de base sigue, más")
        out("  allá de z*, la mayor de d y 12·db, con d = 0.8·ℓm:")
        out(f"    máx(0.8·{D.LW:.2f}, 12·db) = {cc.prolongacion:.2f} m")
        filas = [("flexocompresión biaxial D/C ≤ 1 [21.9.6.1]", cc.z_biaxial),
                 (f"sección crítica única: D/C biaxial ≤ el de la base = {vb.dc_biaxial:.3f} "
                  f"[21.9.7.4]", cc.z_unica)]
        for texto, z in filas:
            marca = "  <- MANDA" if abs(z - cc.z_estrella) < 1e-9 and z > 0 else ""
            out(f"    {z:6.2f} m  {texto}{marca}")
        out(f"    z* = {cc.z_estrella:.2f} m  +  {cc.prolongacion:.2f} m  =  {cc.z_vertical:.2f} m")
        out(f"    (sin la regla de sección crítica única sería {cc.z_biaxial + cc.prolongacion:.2f} m)")
        if cc.z_borde > 0.0:
            out(f"  (2) §21.9.7.4(b), BORDE CONFINADO (3.1): su refuerzo sube máx(ℓm, Mu/4Vu) = "
                f"{cc.z_borde:.2f} m")
        else:
            out("  (2) §21.9.7.4(b), BORDE CONFINADO (3.1): no se exige; no condiciona")
        art, z_norma = max(cc.cotas_de_la_norma, key=lambda t: t[1])
        out(f"    la mayor: {z_norma:.2f} m (§{art})")
        out("")
        out("  ACERO HORIZONTAL Y DETALLADO. En todas las cotas, el reducido frente al de base: el cortante")
        out("  (con su malla horizontal), la longitud del borde, la junta de construcción, los estribos y las")
        out("  cuantías y separaciones de las mallas. El reducido no puede ir donde falla por él (lo que falla")
        out("  igual con el de base se arregla aparte y sale en la verificación completa); el corte queda")
        out("  encima de la cota más alta donde falla:")
        if not cc.fallas_del_reducido:
            out("    el reducido no falla por sí mismo en ninguna cota  → no condiciona el corte")
        for _, z, s, texto in cc.fallas_del_reducido:
            out(f"    hasta z = {z:5.2f} m ({D.PISOS[s]}): {texto}")
        if cc.fallas_del_reducido:
            s = cc.fallas_del_reducido[0][2]
            out(f"    → corte ≥ {cc.z_detallado:.2f} m (la losa encima del {D.PISOS[s]}, el entrepiso más "
                f"alto donde falla)")
        out("")
        out(f"  CORTE REQUERIDO {cc.corte_requerido:.2f} m = máx({z_norma:.2f}, "
            f"{cc.z_detallado:.2f})")
        out("  El armado cambia en la losa más próxima por encima del corte requerido, al empezar el")
        out("  entrepiso inmediatamente superior (criterio del proyecto). Ahí termina el acero de base que")
        out("  no sigue, y el sismo la pone en tracción, así")
        out("  que §12.11.5(a) pide Vu ≤ 2/3·φVn en ese punto (φVn con la malla horizontal del reducido):")
        for t in cc.terminaciones:
            out(f"    losa de {t['z']:.2f} m: |Vu| = {t['Vu']:.1f} kN ({t['estado'].nombre}) frente a "
                f"2/3·{t['phiVn']:.1f} = {t['limite']:.1f} kN  → {si_no(t['cumple'])}")
        out(f"  (gobierna: {cc.gobierna})")
        if cc.corte < A.H - A.TOL:
            out(f"  CORTE A {cc.corte:.2f} m: el reducido va desde el {D.PISOS[dis.piso_de(cc.corte)]} "
                f"hasta el techo; el empalme, en esa losa")
            out("  (§21.9.4.5(c): donde puede fluir el acero vertical, sus longitudes de desarrollo son 1.25")
            out("  veces las de fy; nota del plano)")
        else:
            out(f"  CORTE A {cc.corte:.2f} m: no queda ninguna losa por encima del corte requerido antes "
                f"del techo,")
            out("  así que el reducido no se usa y el acero de base va en toda la altura.")
        bajo = A.revisar_bajo_el_corte(dis, base, corte, vb.dc_biaxial)
        out(f"  Acero de base por debajo del corte: mayor D/C biaxial {bajo['peor_dc']:.3f} en "
            f"z = {bajo['peor_z']:.2f} m; "
            + ("cumple D/C ≤ 1 y la sección crítica única"
               if not any(bajo[k] for k in ("falla_unica", "falla_biaxial"))
               else f"FALLA: {bajo}"))

    # =========================================================================
    # MÓDULO 2 · CORTE
    # =========================================================================
    out.parte(MODULO_2)
    out.modulo("MÓDULO 2 · DISEÑO A CORTE")
    cortante_de_diseno(out, dis)

    out.titulo("2.2 RESISTENCIA A CORTANTE EN EL PLANO — §11.10.3 a §11.10.6, §11.10.10.1 y φ de §9.3.2.3")
    r0 = A.resistencia_en_el_plano(base)
    out(f"  Vn = Vc + Vs ≤ 0.83·√f'c·Acw (§11.10.4);  Acw = ℓm·b = {sec.acw:.4f} m²;  hm/ℓm = "
        f"{r0['esbeltez']:.2f} → αc = {r0['alpha_c']:.2f} (§11.10.5); ρh = {r0['rho_h']:.5f}")
    out(f"  Vc = αc·√f'c·Acw = {r0['alpha_c']:.2f}·√{D.FC:.2f}·{sec.acw:.3f}·1000 = {r0['Vc']:.1f} kN   "
        f"Vs = Acw·ρh·fy = {sec.acw:.3f}·{r0['rho_h']:.5f}·{D.FY:.2f}·1000 = {r0['Vs']:.1f} kN (ec. 11-40)")
    out(f"  Vn = {r0['Vc'] + r0['Vs']:.1f} kN;  tope 0.83·√f'c·Acw = {r0['tope']:.1f} kN"
        + ("  → manda el tope" if r0["recortado"] else ""))
    if red is not None and red.malla_horizontal != base.malla_horizontal:
        rr = A.resistencia_en_el_plano(red)
        out(f"  Reducido ({red.malla_horizontal.texto()}): ρh = {rr['rho_h']:.5f}; Vs = "
            f"{rr['Vs']:.1f} kN; Vn = {rr['Vn']:.1f} kN")
    out("  Con tracción axial (§11.10.6), Vc se multiplica por (1 − 0.29·Nu/Ag) ≥ 0, Nu/Ag en MPa.")
    out("  El tope de §21.9.5.3 es de muros con aberturas: no aplica. φ = 0.85 (§9.3.2.3).")
    out("  D/C = |Vu|/(φ·Vn) en la base:")
    out(f"    {'estado':<18}{'Pu':>9}{'Vu':>9}{'Vn':>9}{'φVn':>9}{'D/C':>7}")
    for e in est0:
        c = A.corte_en_el_plano(base, e)
        out(f"    {e.nombre:<18}{e.P:9.1f}{e.Vu:9.1f}{c['Vn']:9.1f}{c['phiVn']:9.1f}{c['dc']:7.3f}"
            + ("" if e.con_sismo else "   (sin sismo: Vu = Vua)"))
    for etiqueta, arm, z, piso in puntos_de_control(dis, base, red, corte):
        v = dis.verificar(arm, z, piso)
        out(f"  {etiqueta:<20} ({arm.nombre:<8}): gobierna {v.estado_corte.nombre}, "
            f"|Vu| = {abs(v.estado_corte.Vu):.1f} kN, φVn = {v.corte['phiVn']:.1f} kN, "
            f"D/C = {v.dc_corte:.3f}  → {si_no(v.dc_corte <= 1)}")

    out("  El corte en Y (fuera del plano del alma) no se verifica: en esa dirección el muro no trabaja")
    out("  como muro" + ("; las alas son columnas de los pórticos y su cortante va con el pórtico."
                       if sec.tipo != "I" else "."))

    filas_corte = cortante_por_piso(out, dis, base, red, corte)
    cuantia_transversal(out, dis, base, red, corte, refuerzos)
    juntas_de_construccion(out, dis, base, red, corte)

    # =========================================================================
    # MÓDULO 3 · ELEMENTOS DE BORDE
    # =========================================================================
    out.parte(MODULO_3)
    modulo_3(out, dis, base, red, corte, vb)

    # =========================================================================
    # VERIFICACIÓN COMPLETA
    # =========================================================================
    out.parte(VERIFICACION)
    out.modulo("VERIFICACIÓN COMPLETA Y RESUMEN")
    out("  La flexocompresión se verifica con el D/C biaxial (1.4); el de P-Mx de 1.3 es de referencia.")
    out(f"  {'cota':<22}{'armado':<10}{'biaxial':>8}{'corte':>7}{'junta':>7}"
        f"{'borde':>7}{'detallado':>11}   resultado")
    verifs = []
    for etiqueta, arm, z, piso in puntos_de_control(dis, base, red, corte):
        v = dis.verificar(arm, z, piso)
        verifs.append((etiqueta, v))
        det = "ok" if not v.faltas else f"{len(v.faltas)} falla(s)"
        out(f"  {etiqueta:<22}{arm.nombre:<10}{v.dc_biaxial:8.3f}"
            f"{v.dc_corte:7.3f}{v.dc_junta:7.3f}{v.dc_borde:7.3f}{det:>11}   "
            f"{si_no(v.cumple)}")
        for f in v.faltas:
            out(f"      [{f['art']}] {f['texto']}: {f['valor']:.4g} frente a {A.texto_limite(f['limite'])}")
    out("  (D/C; junta = corte por fricción de §21.9.8; borde = longitud exigida / confinada donde se")
    out("  exige elemento de borde confinado)")
    # entre losas: el biaxial del armado de base en todas las cotas bajo el
    # corte (encima, el del reducido ya lo asegura la cota de corte)
    bajo = A.revisar_bajo_el_corte(dis, base, corte, vb.dc_biaxial)
    entre_losas = not any(bajo[k] for k in ("falla_unica", "falla_biaxial"))
    out(f"  Entre losas (todas las cotas, armado de base{' bajo el corte' if red is not None else ''}): "
        f"mayor D/C biaxial {bajo['peor_dc']:.3f} en z = {bajo['peor_z']:.2f} m  → {si_no(entre_losas)}")
    out("")
    out("  Detallado en la base (todas las comprobaciones de armado y, donde hay borde confinado, sus")
    out("  dimensiones):")
    for f in vb.detalle:
        if not f.get("aplica", True):
            continue                         # p. ej. §21.9.6.5 sin tracción que lo active
        out(f"    [{f['art']:<12}] {f['texto']:<76} {f['valor']:10.4g} {f['tipo']:>5} "
            f"{A.texto_limite(f['limite']):<10}  {si_no(f['cumple'])}")
    out("")
    out("  Para el plano (lo que la norma pide dibujar y no es una cifra que se compruebe):")
    for art, texto in A.detalles_constructivos(dis, base, red):
        out(f"    [{art}] {texto}")

    todo = all(v.cumple for _, v in verifs)
    convencion = mef.comprobar_convencion()
    todo &= all(f["cumple"] for f in convencion)
    todo &= entre_losas
    peso = A.peso_de_acero(dis, base, red, corte)
    out("")
    out(f"  Peso de acero en toda la altura, sin traslapes: {peso['total']:.1f} kg (vertical "
        f"{peso['vertical']:.1f}, malla horizontal {peso['horizontal']:.1f}, estribos y grapas "
        f"{peso['estribos']:.1f})")
    out("")
    out(f"  VEREDICTO: {si_no(todo)}")

    if D.DIBUJAR:
        F.figuras(out, dis, base, red, vb, cc, filas_corte, corte)
    destino = carpeta_del_muro()
    out("")
    out(f"  ({time.perf_counter() - t0:.1f} s)  Informe completo: {destino / f'informe_{D.MURO}.txt'}")
    out(f"  y una carpeta por parte en {destino}: " + ", ".join(n for n, _ in out.partes))
    out.guardar(destino)
    return 0 if todo else 1


def flexion_minima(out, dis, base, vb) -> None:
    """1.5: §10.3.5 (la deformación del acero con poca compresión) y
    §21.9.6.5 (la resistencia no menor que el momento de agrietamiento), en la
    base; la verificación completa los repite en cada losa y en el corte."""
    sec = A.seccion()
    out.titulo("1.5 DEFORMACIÓN DEL ACERO Y MOMENTO DE AGRIETAMIENTO — §10.3.5 y §21.9.6.5")
    umbral = 0.1 * D.FC * mef.MPA * sec.area
    out("  §10.3.5 (§21.9.6.1 remite a §10.3): con Pu < 0.1·f'c·Ag la deformación del acero más")
    out("  traccionado, al llegar a la resistencia nominal, es εt ≥ 0.004 (así se da por cumplido el")
    out(f"  tope de acero de §10.3.4). 0.1·f'c·Ag = {umbral:.1f} kN. En el plano, con los estados del sismo")
    out("  en X (los del elemento de borde, 3.1) y comprimiendo cada extremo:")
    for caso in vb.casos_borde:
        for e, c, eps in caso["por_estado"]:
            aplica = -e.P < umbral
            out(f"    extremo {caso['zona']:<10}{nombre_por_P(e):<28} Pu = {e.P:8.1f} kN   c = {c:.3f} m   "
                f"εt = {eps:.5f}" + (f"  → {si_no(eps >= A.DEFORMACION_MINIMA)}" if aplica
                                     else "  (Pu ≥ 0.1·f'c·Ag: no aplica)"))
    if A.deformacion_minima(vb.casos_borde) is None:
        out("  Ningún estado tiene Pu < 0.1·f'c·Ag: §10.3.5 no aplica en la base.")
    out("")
    fr = 0.62 * math.sqrt(D.FC)
    out("  §21.9.6.5: donde la tracción de la fibra extrema por Pu y Mu (sección bruta) pasa de 0.63·√f'c")
    out(f"  = {0.63 * math.sqrt(D.FC):.2f} MPa, φMn ≥ Mcr = (fr + Pu/Ag)·Ig/yt, con fr = 0.62·√f'c = {fr:.2f} MPa")
    out("  (§9.6.2.3, ec. 9-12) y Pu en compresión positiva. En la base, la mayor tracción de cada estado:")
    out(f"    {'estado':<18}{'Pu':>9}{'Mu':>10}{'σt (MPa)':>10}{'Mcr':>9}{'φMn':>9}   resultado")
    diag = dis.diagrama(base)
    for e in dis.estados(0.0, 0):
        sigma = max(e.P / sec.area + e.M * (x - sec.xc) / sec.Ix for x in (0.0, sec.lw)) / mef.MPA
        if sigma <= 0.63 * math.sqrt(D.FC):
            out(f"    {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{sigma:10.2f}{'':>9}{'':>9}   no aplica")
            continue
        m = A.momento_de_agrietamiento(diag, [e])
        out(f"    {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{sigma:10.2f}{m['Mcr']:9.1f}{m['phiMn']:9.1f}   "
            f"{si_no(m['phiMn'] >= m['Mcr'])}")
    out("  (kN y kN·m; σt negativa: la fibra queda comprimida. La verificación completa lo repite en cada")
    out("  losa y en el corte.)")


def cortante_de_diseno(out, dis) -> None:
    """2.1: el cortante de diseño de §21.9.5.5, estado por estado en la base."""
    amp = dis.amplificacion()
    out.titulo("2.1 CORTANTE DE DISEÑO — §21.9.5.5")
    out("  Vu = Vua·(Mn/Mua): Vua y Mua del análisis; Mn, la resistencia nominal en el plano con el acero")
    out("  colocado en la base y el Pu del estado (sin φ), comprimiendo el extremo que comprime Mua. El")
    out(f"  cociente no se toma mayor que 0.5·R0 = 0.5·{D.R0:g} = {amp['tope']:.2f} ni menor que 1 (Vu ≥ Vua).")
    out("  Por estado con sismo, en la base (la sección crítica); los estados sin sismo no se amplifican:")
    out(f"    {'estado':<18}{'Pu':>9}{'Mua':>10}{'Mn':>10}{'Mn/Mua':>9}{'factor':>8}{'Vua':>9}{'Vu':>9}")
    for e in dis.estados(0.0, 0):
        f = amp["factores"].get((e.combo, e.paso))
        if f is None:
            out(f"    {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{'':>10}{'':>9}{1.0:8.2f}{e.V:9.1f}{e.Vu:9.1f}"
                f"   (sin sismo)")
            continue
        cociente = "∞" if math.isinf(f["cociente"]) else f"{f['cociente']:.2f}"
        mn = "—" if math.isinf(f["Mn"]) else f"{f['Mn']:.1f}"
        out(f"    {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{mn:>10}{cociente:>9}{f['factor']:8.2f}"
            f"{e.V:9.1f}{e.Vu:9.1f}")
    g = amp["gobierna"]
    mu4vu = abs(g.M) / (4 * abs(g.V)) if g.V else 0.0
    out(f"  Altura: el artículo permite limitarlo a la mayor de ℓm = {D.LW:.2f} m, Mu/4Vu = "
        f"{abs(g.M):.1f}/(4·{abs(g.V):.1f}) = {mu4vu:.2f} m ({g.nombre}, el de mayor |Mu|)")
    out(f"  y los dos primeros pisos = {amp['dos_pisos']:.2f} m  → {amp['altura']:.2f} m. Como V no cambia "
        f"dentro de un entrepiso,")
    pisos = [D.PISOS[s] for s in range(len(D.ALTURAS)) if dis.amplifica(s)]
    resto = [D.PISOS[s] for s in range(len(D.ALTURAS)) if not dis.amplifica(s)]
    out(f"  se amplifican los entrepisos que empiezan por debajo: {', '.join(pisos)}"
        + (f"; en {', '.join(resto)}, Vu = Vua." if resto else "."))


def refuerzo_de(dis, arm, pisos: list[int]) -> dict:
    """Las cuantías mínimas de §11.10 de un armado, con el entrepiso de mayor
    |Vu| de los que lleva (la malla es la misma en todos ellos)."""
    return max((A.refuerzo_de_corte_requerido(arm, dis.estados(dis.demanda.z_base[s], s)) for s in pisos),
               key=lambda r: r["Vu"])


def cuantia_longitudinal(out, dis, base, red, refuerzos: dict) -> None:
    """1.2: cuantía mínima y separación del refuerzo vertical del alma, de
    cada armado con su malla horizontal y el cortante de sus entrepisos."""
    sec = A.seccion()
    out.titulo("1.2 CUANTÍA MÍNIMA DEL REFUERZO VERTICAL — §11.10.7, §11.10.10.3, §21.9.5.4, §21.9.4.2")
    armados = [x for x in (base, red) if x is not None]
    if base.malla_vertical.automatica:
        out(f"  Separación calculada: la mayor múltiplo de 2 o 5 cm (≤ 40 cm y ≤ 3·b) con ρv ≥ "
            f"{'máx(0.0025, ρh)' if A.ESBELTEZ <= 2.0 else '0.0025'}")
    umbral = 0.085 * A.RAIZ_FC * mef.MPA * sec.acw
    out(f"  Con el mayor |Vu| de los entrepisos de cada armado, frente a 0.085·√f'c·Acw = {umbral:.1f} kN:")
    out("  si es menor, §11.10.7(b): ρv ≥ 0.0015; si no (§11.10.8), §11.10.10.3:")
    out("      ρv ≥ 0.0025 + 0.5·(2.5 − hm/ℓm)·(ρh − 0.0025) ≥ 0.0025, sin pasar del ρh que pide la")
    out("      resistencia (§11.10.10.1). ρv = cortinas·Ab/(s·b), de la malla vertical:")
    for arm in armados:
        r = refuerzos[arm.nombre]
        mv, mh = arm.malla_vertical, arm.malla_horizontal
        if r["bajo"]:
            out(f"    {arm.nombre}: |Vu| = {r['Vu']:.1f} kN < {umbral:.1f} kN → §11.10.7(b): ρv ≥ 0.0015")
        else:
            out(f"    {arm.nombre}: |Vu| = {r['Vu']:.1f} kN ≥ {umbral:.1f} kN → 0.0025 + 0.5·(2.5 − "
                f"{A.ESBELTEZ:.2f})·({mh.cuantia:.5f} − 0.0025) = {r['ecuacion_111013']:.5f};")
            out(f"      ρh que pide la resistencia = {r['rho_h_resistencia']:.5f} → ρv ≥ {r['rho_v_min']:.5f}")
        out(f"      ρv = {mv.cortinas}·{A.area(mv.barra) * 1e4:.2f}/({mv.separacion * 100:.0f}·"
            f"{D.ESPESOR * 100:.0f}) = {mv.cuantia:.5f}  → {si_no(mv.cuantia >= r['rho_v_min'] * (1 - 1e-9))}")
    if A.ESBELTEZ <= 2.0:
        out(f"  §21.9.5.4: hm/ℓm = {A.ESBELTEZ:.2f} ≤ 2, así que ρv ≥ ρh (la malla horizontal del mismo "
            f"armado):")
        for arm in armados:
            mv, mh = arm.malla_vertical, arm.malla_horizontal
            out(f"    {arm.nombre:<9} ρv = {mv.cuantia:.5f} frente a ρh = {mh.cuantia:.5f}  → "
                f"{si_no(mv.cuantia >= mh.cuantia * (1 - 1e-9))}")
    else:
        out(f"  §21.9.5.4 (ρv ≥ ρh): solo si hm/ℓm ≤ 2; aquí hm/ℓm = {A.ESBELTEZ:.2f} → no aplica")
    tope = min(0.40, 3 * D.ESPESOR)
    out(f"  Separación de las barras verticales (§21.9.4.2, §11.10.10.4): ≤ mín(3·b, 400 mm) = "
        f"mín({3 * D.ESPESOR * 100:.0f}, 40) = {tope * 100:.0f} cm")
    for arm in armados:
        mv = arm.malla_vertical
        real = A.separacion_real_vertical(dis.barras(arm))
        out(f"    {arm.nombre}: malla s = {mv.separacion * 100:.0f} cm; la mayor separación real entre "
            f"líneas de barras verticales del alma = {real * 100:.1f} cm  → "
            f"{si_no(max(mv.separacion, real) <= tope * (1 + 1e-9))}")
    tramos = A.tramos_libres(base)
    out("  La malla va solo en los tramos libres entre elementos, sin pisar sus barras: "
        + ", ".join(f"{a * 100:.1f}–{b * 100:.1f} cm" for a, b in tramos)
        + f"; líneas a {base.malla_vertical.separacion * 100:.0f} cm, centradas en cada tramo (un tramo "
        f"cuyo hueco no pasa de la separación no lleva líneas).")
    out("  §21.9.4.4 (sin estribos que confinen la malla): ρv ≤ 0.01 → "
        + ", ".join(f"{arm.nombre} {arm.malla_vertical.cuantia:.5f}" for arm in armados)
        + f"  → {si_no(all(arm.malla_vertical.cuantia <= 0.01 for arm in armados))}")


def cortante_por_piso(out, dis, base, red, corte: float) -> list[dict]:
    """2.3: el cortante de diseño en cada entrepiso (V es constante entre
    losas), por combinación, con φVn y el D/C. φVn con la malla horizontal del
    armado del entrepiso. Devuelve las filas (para la figura)."""
    out.titulo("2.3 CORTANTE DE DISEÑO POR ENTREPISO — §21.9.5.5 y §11.10")
    out("  V no cambia dentro del entrepiso (entre losas no entra fuerza horizontal). Por combinación, el")
    out("  estado de mayor D/C: |Vua| del análisis, el factor de 2.1 (1 donde no se amplifica), |Vu| de")
    out("  diseño y φVn con la malla horizontal del armado del entrepiso (con su Pu, por §11.10.6).")
    out(f"    {'entrepiso':<10}{'armado':<10}{'combinación':<15}{'|Vua|':>8}{'factor':>8}{'|Vu|':>8}"
        f"{'φVn':>8}{'D/C':>7}")
    filas = []
    for s in range(len(D.ALTURAS)):
        z = dis.demanda.z_base[s]
        arm = A.armado_en(dis, base, red, corte, s)
        por_combo: dict = {}
        for e in dis.estados(z, s):
            c = A.corte_en_el_plano(arm, e)
            if e.combo not in por_combo or c["dc"] > por_combo[e.combo][1]["dc"]:
                por_combo[e.combo] = (e, c)
        peor = None
        for combo, (e, c) in por_combo.items():
            vua, vu = abs(e.V), abs(e.Vu)
            fila = {"piso": s, "z0": z, "z1": dis.demanda.z_tope[s], "combo": combo, "Vua": vua,
                    "Vu": vu, "phiVn": c["phiVn"], "dc": c["dc"], "armado": arm.nombre}
            filas.append(fila)
            if peor is None or fila["dc"] > peor["dc"]:
                peor = fila
            out(f"    {D.PISOS[s]:<10}{arm.nombre:<10}{combo:<15}{vua:8.1f}{(vu / vua if vua else 1.0):8.2f}"
                f"{vu:8.1f}{c['phiVn']:8.1f}{c['dc']:7.3f}")
        out(f"    {'':<10}→ gobierna {peor['combo']}: |Vu| = {peor['Vu']:.1f} kN, φVn = "
            f"{peor['phiVn']:.1f} kN, D/C = {peor['dc']:.3f}  → {si_no(peor['dc'] <= 1.0)}")
    out(f"  (kN. Figura: cortante_en_altura_{D.MURO}.png)")
    return filas


def cuantia_transversal(out, dis, base, red, corte: float, refuerzos: dict) -> None:
    """2.4: cuantía mínima y separación del refuerzo horizontal de cada
    armado, con el cortante de los entrepisos donde va."""
    sec = A.seccion()
    out.titulo("2.4 CUANTÍA MÍNIMA DEL REFUERZO HORIZONTAL — §11.10.7, §11.10.10.2 y §21.9.4.2")
    umbral = 0.085 * A.RAIZ_FC * mef.MPA * sec.acw
    out("  Cada malla horizontal se comprueba con el mayor |Vu| de los entrepisos donde va (en cada")
    out("  losa y en el corte lo repite la verificación completa). Si |Vu| < 0.085·√f'c·Acw = "
        f"{umbral:.1f} kN,")
    out("  §11.10.7(a): ρh ≥ 0.002; si no, §11.10.10.2: ρh ≥ 0.0025.")
    tope = min(0.40, 3 * D.ESPESOR)
    for arm in [x for x in (base, red) if x is not None]:
        r = refuerzos[arm.nombre]
        mh = arm.malla_horizontal
        pisos = pisos_del_armado(dis, arm, corte)
        out(f"  {arm.nombre.upper()} ({mh.texto()}; entrepisos {', '.join(D.PISOS[s] for s in pisos)}"
            + (", no se usa: el corte queda en el techo" if arm is red and corte >= A.H - A.TOL
               else "") + "):")
        out(f"    |Vu| = {r['Vu']:.1f} kN {'<' if r['bajo'] else '≥'} {umbral:.1f} kN → ρh ≥ {r['rho_h_min']}")
        out(f"    ρh = cortinas·Ab/(s·b) = {mh.cortinas}·{A.area(mh.barra) * 1e4:.2f}/"
            f"({mh.separacion * 100:.0f}·{D.ESPESOR * 100:.0f}) = {mh.cuantia:.5f}  → "
            f"{si_no(mh.cuantia >= r['rho_h_min'] * (1 - 1e-9))}")
        out(f"    separación ≤ mín(3·b, 400 mm) = {tope * 100:.0f} cm (§21.9.4.2, §11.10.10.2); "
            f"s = {mh.separacion * 100:.0f} cm  → {si_no(mh.separacion <= tope * (1 + 1e-9))}")
    out("  §21.9.5.4: refuerzo de corte repartido en las dos direcciones del plano del muro (malla")
    out("  vertical y horizontal); el ρh que pide la resistencia lo da el D/C de 2.2 y 2.3.")


def juntas_de_construccion(out, dis, base, red, corte: float) -> None:
    """2.5: §21.9.8 en la cimentación y en cada losa."""
    out.titulo("2.5 CORTE POR FRICCIÓN EN LAS JUNTAS DE CONSTRUCCIÓN — §21.9.8, §11.7.4 y §11.7.5")
    out("  φVn = 0.85·μ·(Nu + Av·fy) ≥ Vu (ec. 21-7), en la cimentación y en cada losa (las juntas de")
    out(f"  construcción). Nu: la menor compresión de 0.9·CM + CS ({', '.join(D.COMBINACIONES_DE_LA_JUNTA)}")
    out("  con los dos signos del sismo); Av: todo el acero vertical que cruza la junta (criterio del")
    out(f"  proyecto: §11.7.4.1); μ = {D.MU_JUNTA:g} (§11.7.4.3, MU_JUNTA); fy ≤ 420 MPa (§11.7.6). Vn no "
        f"pasa del tope")
    out("  de §11.7.5 con Ac = Ag. Vu: el de diseño (§21.9.5.5) en la junta.")
    out(f"    {'junta':<16}{'armado':<10}{'Nu':>9}{'Av·fy':>9}{'Vn':>9}{'tope':>9}{'φVn':>9}{'Vu':>9}"
        f"{'D/C':>7}   resultado")
    for s in range(len(D.ALTURAS)):
        z = dis.demanda.z_base[s]
        arm = A.armado_en(dis, base, red, corte, s)
        j = dis.verificar(arm, z, s).junta
        nombre = "cimentación" if s == 0 else f"losa {z:.2f} m"
        out(f"    {nombre:<16}{arm.nombre:<10}{j['Nu']:9.1f}{j['Av'] * j['fy']:9.1f}{j['Vn']:9.1f}"
            f"{j['tope']:9.1f}{j['phiVn']:9.1f}{j['Vu']:9.1f}{j['dc']:7.3f}   {si_no(j['dc'] <= 1.0)}")
    j0 = dis.verificar(base, 0.0, 0).junta
    out(f"  (kN; Nu > 0 en compresión, en la cimentación del estado {j0['estado_Nu'].nombre}; "
        f"tope = {j0['texto_tope']})")


REGLA = {"confinado": "CONFINADO §21.9.7.6", "concentrado": "concentrado §21.9.7.7",
         "ala central": "proyecto (ala central)"}


def modulo_3(out, dis, base, red, corte: float, vb) -> None:
    """Módulo 3: necesidad del borde confinado, sus dimensiones, los estribos
    de cada elemento por zonas y el resumen de §21.9.7."""
    sec = A.seccion()
    out.modulo("MÓDULO 3 · ELEMENTOS DE BORDE")
    out("  Los extremos se confinan siempre, en la longitud declarada (criterio del proyecto; en el")
    out("  reducido, no más que en la base). Aquí: 3.1 decide si la norma EXIGE elemento de borde")
    out("  confinado; 3.2 comprueba sus dimensiones; 3.3 diseña los estribos de cada elemento por zonas en")
    out("  altura, con borde confinado (§21.9.7.6) o con el refuerzo concentrado en el borde (§21.9.7.7);")
    out("  3.4 resume §21.9.7. Solo en la dirección larga (X): en Y el muro no trabaja como muro"
        + ("; las alas cortas son columnas de los pórticos." if sec.tipo != "I" else "."))

    # ---- 3.1 -------------------------------------------------------------------
    casos = vb.casos_borde
    c0 = casos[0]
    out.titulo("3.1 ¿SE EXIGE ELEMENTO DE BORDE CONFINADO? — §21.9.7.4 (sección crítica en la base)")
    out(f"  ℓm = {c0['lw']:.3f} m;  δu/hm = {dis.deriva:.5f} → se toma {c0['deriva']:.4f} (no menor que 0.005)")
    out(f"  Se confina si c > c_lím = ℓm/(600·δu/hm) = {c0['c_lim']:.3f} m (ec. 21-6)")
    out("  c a la fuerza axial amplificada y la resistencia nominal a momento, combinaciones del sismo en el")
    out(f"  plano ({', '.join(D.COMBINACIONES_DEL_SISMO_EN_X)}), con la sección real (con el ala si la hay):")
    for caso in casos:
        out(f"  extremo {caso['zona']} (comprimido):")
        for e, c, _ in caso["por_estado"]:
            out(f"    {nombre_por_P(e):<26} Pu = {e.P:8.1f} kN   c = {c:.3f} m")
        out(f"    el mayor: c = {caso['c']:.3f} m ({nombre_por_P(caso['estado'])})  →  c/c_lím = "
            f"{caso['indice']:.3f}  → "
            + ("SE EXIGE elemento de borde confinado" if caso["exige"]
               else "no se exige elemento de borde confinado"))
    equilibrio_de_borde(out, dis, base, max(casos, key=lambda cs: cs["indice"]))
    g = c0["gobierna"]
    out("  §21.9.7.4 vale para muros continuos con una sola sección crítica: el programa lo asegura con")
    out("  la regla de la sección crítica única (1.6).")
    out(f"  Altura (§21.9.7.4(b)): el borde sube sobre la sección crítica máx(ℓm, Mu/4Vu), con "
        f"{g.nombre} (el de mayor |Mu|")
    out("  y su cortante del análisis): "
        + (f"Mu/4Vu = {abs(g.M):.1f}/(4·{abs(g.V):.1f}) = {abs(g.M) / (4 * abs(g.V)):.2f} m"
           if g.V else "Vu = 0") + f" → {dis.altura_region:.2f} m.")
    alto = {zn: alt for zn, (ex, alt, _) in dis.region.items() if ex}    # altura del borde confinado
    if alto:
        out("  Por debajo de la base, los estribos del borde confinado entran en el apoyo (§21.9.7.6(g),")
        out("  nota del plano).")
    sig = [A.esfuerzo_de_fibra_extrema(e) for e in dis.estados(0.0, 0)]
    out("  Informativo, la otra vía (§21.9.7.5, por esfuerzos; no se usa): compresión elástica máxima en")
    out(f"  la base {max(sig) / 1000:.2f} MPa = {max(sig) / (D.FC * mef.MPA):.2f}·f'c (pediría borde con "
        f"0.2·f'c)")

    # ---- 3.2 -------------------------------------------------------------------
    out.titulo("3.2 DIMENSIONES DEL BORDE CONFINADO — §21.9.7.6(a)–(c), §21.9.7.2 y §21.9.7.3")
    resumen: dict = {}

    def anota(art: str, cumple: bool) -> None:
        resumen.setdefault(art, []).append(cumple)

    for caso in casos:
        ext = caso["extremo"]
        c = caso["c"]
        if caso["exige"]:
            hasta = alto[ext.nombre]
            pisos_region = [s for s in range(len(D.ALTURAS)) if dis.demanda.z_base[s] < hasta - 1e-9]
            out(f"  extremo {ext.nombre}: BORDE CONFINADO hasta {hasta:.2f} m sobre la base "
                f"(c = {c:.3f} m en la sección crítica)")
            for s in pisos_region:
                for f in A.dimensiones_de_borde(caso, s):
                    clave = f["art"]
                    if s != pisos_region[0] and clave == "21.9.7.6(a)":
                        continue                # la longitud no cambia con el entrepiso
                    anota(clave, f["cumple"])
                    if clave == "21.9.7.6(a)":
                        out(f"    (a) longitud ≥ {f['nota']} = {f['limite']:.3f} m;  confinada "
                            f"{f['valor']:.3f} m  → {si_no(f['cumple'])}")
                    elif clave == "21.9.7.6(b)":
                        out(f"    (b) ancho b de la zona comprimida en esa longitud (el menor, con el ala "
                            f"si la hay) = {f['valor']:.3f} m ≥ hu/16:")
                        out(f"        {f['nota']}  → {si_no(f['cumple'])}")
                    else:
                        out(f"    §21.9.7.2: espesor del elemento de borde {f['valor']:.3f} m ≥ "
                            f"{f['nota']}  → {si_no(f['cumple'])}")
            mayor = max(A.diametro(base.elemento(n).barra) for n in ext.elementos
                        if base.elemento(n) is not None)
            ok = D.RECUBRIMIENTO >= max(0.025, mayor) - 1e-9
            anota("21.9.7.3", ok)
            out(f"    §21.9.7.3: recubrimiento {D.RECUBRIMIENTO * 1000:.0f} mm ≥ máx(25 mm, barra más gruesa "
                f"{mayor * 1000:.1f} mm)  → {si_no(ok)}")
            if sec.tipo == "III" and ext.nombre == "derecho":
                out("    (c) sección con ala: el borde incluye el ancho efectivo del ala en compresión y entra ≥ "
                    "300 mm en")
                out("        el alma → nota para el plano (VERIFICACIÓN COMPLETA, «Para el plano»)")
            else:
                out("    (c) sin ala en este extremo: no aplica")
            out(f"    Encima de {hasta:.2f} m el extremo ya no lleva borde confinado: rige §21.9.7.7 (3.3).")
        else:
            out(f"  extremo {ext.nombre}: no exige borde confinado (c = {c:.3f} m ≤ c_lím) → no aplican "
                f"§21.9.7.6 ni §21.9.7.2;")
            out(f"    se confina igual en {ext.provista:.3f} m (criterio del proyecto), con los estribos de "
                f"§21.9.7.7 (3.3)")

    # ---- 3.3 -------------------------------------------------------------------
    out.titulo("3.3 ESTRIBOS DE LOS ELEMENTOS POR ZONAS — §21.9.7.6(d)–(f), §21.9.7.7, §21.6.4.1, "
               "§21.6.4.6, §7.10.5.3")
    out("  CONFINADO (§21.9.7.6): estribos cerrados y grapas del mismo diámetro y separación (§21.6.4.1),")
    out("    con cada barra de esquina y una sí y otra no apoyadas, ninguna a más de 150 mm libres de una")
    out("    apoyada (§7.10.5.3); ramas o grapas a ≤ 350 mm (§21.6.4.6); diámetro de (e); s ≤ la menor de")
    out("    8·db, la menor dimensión del elemento de borde y 250 mm (f).")
    out("  CONCENTRADO (§21.9.7.7, sin borde confinado y con el refuerzo vertical concentrado en el borde):")
    out("    lo mismo, con s ≤ la menor de 10·db, la menor dimensión del elemento y 250 mm.")
    out("  REPLANTEO (criterio del proyecto): s es múltiplo de 2 o 5 cm, y 8·db y 10·db se llevan al")
    out("    múltiplo de 2 o 5 cm más cercano, por debajo o por encima (p. ej. 8·db = 12.7 cm → 12 cm).")
    if sec.tipo == "II":
        out("  ALA CENTRAL (tipo II): no es borde del muro en X; se confina por criterio del proyecto con los")
        out("    límites de §21.9.7.7 (como columna en Y, su diseño va con el pórtico).")
    out("  db: la barra vertical más pequeña del borde («la barra longitudinal confinada de menor")
    out("  diámetro»; en el tipo III, del derecho y del ala). Diámetro mínimo del estribo (§21.9.7.6(e)): 8 mm")
    out("  con barras de hasta 5/8\", 3/8\" hasta 1\" y 1/2\" con las mayores. La norma no pide un área Ash")
    out("  en los muros (§21.9.7.6(d) remite solo a §21.6.4.1 y §21.6.4.6).")
    out("  Cada elemento lleva un estribo en el armado de base, con una separación hasta la altura de")
    out("  cambio y, si se declara «separacion_estribo_arriba», otra encima (criterio del proyecto), y un")
    out("  estribo y una separación en el reducido. La altura de cambio es la mayor de la de §21.9.7.4(b),")
    out("  máx(ℓm, Mu/4Vu), y la del borde confinado de cada extremo que lo exige, al cm hacia arriba:")
    out(f"  {dis.altura_de_cambio():.2f} m. «s ≤» da la máxima de cada zona.")
    zonas = A.zonas_de_estribos(dis, base, red, corte)
    for nombre in A.nombres_de_elementos():
        if nombre not in zonas:
            out("")
            out(f"  Elemento {nombre}: absorbido por el ala (sus barras caerían dentro de ella): sin barras ni "
                f"estribo propios; confina el ala.")
            continue
        filas = zonas[nombre]
        e = filas[0]["elemento"]
        suyos = [x.nombre for x in A.extremos() if nombre in x.elementos]
        donde = (f"extremo {', '.join(suyos)}" if suyos
                 else "ala central: no es borde del muro en X; en Y es columna del pórtico")
        cj = e.caja
        out("")
        estribos_por_armado, elemento_por_armado = {}, {}
        for f in filas:
            el = f["elemento"]
            elemento_por_armado.setdefault(f["armado"].nombre, el)
            # la base puede llevar dos separaciones (la de arriba, desde la
            # altura de cambio)
            texto = f"{el.estribo}@{el.separacion_estribo * 100:.0f} cm" if el.estribo else "sin estribo"
            lista = estribos_por_armado.setdefault(f["armado"].nombre, [])
            if not lista:
                lista.append(texto)
            elif texto != lista[-1].split(" desde ")[0]:
                lista.append(f"{texto} desde {f['z0']:.2f} m")
        estribos_por_armado = {a: " y ".join(t) for a, t in estribos_por_armado.items()}
        cajas_distintas = len({(round(el.caja.ancho, 9), round(el.caja.alto, 9))
                               for el in elemento_por_armado.values()}) > 1
        if cajas_distintas:
            texto_caja = "caja " + ", ".join(f"{el.caja.ancho * 100:.1f} × {el.caja.alto * 100:.1f} cm ({a})"
                                             for a, el in elemento_por_armado.items())
        else:
            texto_caja = f"caja {cj.ancho * 100:.1f} × {cj.alto * 100:.1f} cm"
        out(f"  Elemento {nombre} ({donde}): {texto_caja}; estribo "
            + ", ".join(f"{t} ({a})" for a, t in estribos_por_armado.items()))
        out(f"    {'zona z (m)':<14}{'armado':<10}{'barras':<8}{'regla':<24}{'hx':>7}"
            f"{'hx ≤':>7}{'s ≤':>7}{'s':>5}   resultado")
        for f in filas:
            est, el = f["est"], f["elemento"]
            if est is None:
                regla = (A.regla_de_estribos(f["exige"])[0] if suyos else "ala central")
                out(f"    {f['z0']:5.2f}–{f['z1']:<7.2f} {f['armado'].nombre:<10}{el.barra:<8}"
                    f"{REGLA[regla]:<24}   sin estribo declarado   NO CUMPLE")
                anota(f"estribos {regla}", False)
                continue
            out(f"    {f['z0']:5.2f}–{f['z1']:<7.2f} {f['armado'].nombre:<10}{el.barra:<8}"
                f"{REGLA[est['regla']]:<24}{est['apoyo']['hx'] * 100:7.1f}"
                f"{est['hx_lim'] * 100:7.1f}{est['s_max'] * 100:7.1f}{el.separacion_estribo * 100:5.0f}"
                f"   {si_no(f['cumple'])}")
            anota(f"estribos {est['regla']}", f["cumple"])
        out("    (hx, s en cm)")
        for f in filas:
            est, el = f["est"], f["elemento"]
            if est is None:
                continue
            ap = est["apoyo"]
            cols, fils = A.caras_del_estribo(el, f["armado"])
            out(f"    {f['z0']:.2f}–{f['z1']:.2f} m ({REGLA[est['regla']]}, db = {est['db'] * 1000:.1f} mm):")
            ok_s = el.separacion_estribo <= est["s_max"] * (1 + 1e-9)
            ok_hx = ap["hx"] <= est["hx_lim"] * (1 + 1e-9)
            ok_d = A.diametro(el.estribo) >= est["estribo_min"] * (1 - 1e-9)
            if est["regla"] == "confinado":
                anota("21.9.7.6(d)", ok_hx)
                anota("21.9.7.6(e)", ok_d)
                anota("21.9.7.6(f)", ok_s)
            out(f"      s ≤ la menor de {'; '.join(x['expr'] for x in est['limites'])} [{est['art_s']}] → "
                f"{est['s_max'] * 100:.1f} cm;  s = {el.separacion_estribo * 100:.0f} cm  → {si_no(ok_s)}")
            out(f"      apoyo lateral (§7.10.5.3 y §21.6.4.6): {ap['ramas_y']} de {len(cols)} barras por cara en X "
                f"y {ap['ramas_x']} de {len(fils)} por cara en Y, en esquina o grapa")
            out(f"        → hx = {ap['hx'] * 100:.1f} cm ≤ 35.0 cm  → {si_no(ok_hx)}")
            minimo = est["estribo_min"]
            out(f"      diámetro del estribo (§21.9.7.6(e)): barras hasta {est['db_max'] * 1000:.1f} mm → "
                f"≥ {minimo * 1000:.1f} mm; {el.estribo} = {A.diametro(el.estribo) * 1000:.1f} mm  → "
                f"{si_no(ok_d)}")

        for arm in [x for x in (base, red) if x is not None and nombre == "derecho"]:
            for y, cerca in A.esquinas_sin_barra(arm):
                out(f"    {arm.nombre}: esquina del estribo dentro del ala en y = {y * 100:+.1f} cm: el modelo "
                    f"no tiene ahí barra del ala (la más")
                out(f"    cercana a {cerca * 100:.1f} cm); en el plano va una fila del ala en esa cara "
                    f"(§7.10.5.3: las barras")
                out("    de esquina se apoyan en la esquina del estribo). Nota del plano.")

    # ---- 3.4 -------------------------------------------------------------------
    out.titulo("3.4 RESUMEN DE §21.9.7")
    exigidos = [cs for cs in casos if cs["exige"]]

    def estado(art: str) -> str:
        v = resumen.get(art)
        return "—" if v is None else si_no(all(v))

    if exigidos:
        out("  §21.9.7.6, borde confinado en el extremo "
            + " y ".join(f"{cs['zona']} hasta {alto[cs['zona']]:.2f} m" for cs in exigidos) + ":")
        ala_c = sec.tipo == "III" and any(cs["zona"] == "derecho" for cs in exigidos)
        filas = [("(a) longitud horizontal", "3.2", estado("21.9.7.6(a)")),
                 ("(b) ancho de la zona comprimida ≥ hu/16", "3.2", estado("21.9.7.6(b)")),
                 ("§21.9.7.2 espesor ≥ hu/16 y ≥ 150 mm", "3.2", estado("21.9.7.2")),
                 ("§21.9.7.3 recubrimiento ≥ 25 mm y ≥ db", "3.2", estado("21.9.7.3")),
                 ("(c) sección con ala", "plano" if ala_c else "—",
                  "nota del plano" if ala_c else
                  "no aplica (sin ala)" if sec.tipo == "I" else
                  "no aplica (el ala está en medio del alma)" if sec.tipo == "II" else
                  "no aplica (el extremo con ala no lo exige)"),
                 ("(d) estribos de §21.6.4.1 y §21.6.4.6 (hx ≤ 350 mm)", "3.3", estado("21.9.7.6(d)")),
                 ("(e) diámetro del estribo", "3.3", estado("21.9.7.6(e)")),
                 ("(f) separación de los estribos", "3.3", estado("21.9.7.6(f)")),
                 ("(g) estribos dentro del apoyo", "plano", "nota del plano"),
                 ("(h) malla horizontal anclada en el núcleo (ℓdg)", "4",
                  si_no(all(f["cumple"] for f in vb.detalle if f["art"] == "21.9.7.6(h)")))]
        for texto, donde, res in filas:
            out(f"    {texto:<56}{donde:<8}{res}")
    else:
        out("  §21.9.7.6: no aplica (ningún extremo exige borde confinado).")
    out("  §21.9.7.7, donde no hay borde confinado"
        + (f" (encima de {max(alto.values()):.2f} m" + ("" if len(exigidos) == len(casos) else
                                                         " y en el otro extremo") + "):" if exigidos else
           " (los dos extremos):"))
    concentrados = resumen.get("estribos concentrado")
    out(f"    {'(a) estribos: diámetro, hx y s ≤ mín(10·db, dim., 250)':<56}{'3.3':<8}"
        + (si_no(all(concentrados)) if concentrados else "no aplica"))
    out(f"    {'(b) malla horizontal con gancho estándar en el borde':<56}{'plano':<8}nota del plano")
    if resumen.get("estribos ala central"):
        out(f"    {'ala central (columna en Y; confinada por el proyecto)':<56}{'3.3':<8}"
            f"{si_no(all(resumen['estribos ala central']))}")


def biaxial_por_nivel(out, dis, base, red, corte: float) -> None:
    """1.4: el D/C biaxial en cada nivel (la base y el tope de cada entrepiso
    y la cota de corte), con el armado que lleva y el estado que gobierna, y
    el mayor de cada entrepiso entre todas sus cotas."""
    dem = dis.demanda
    out("")
    out("  En cada nivel: la base y el tope de cada entrepiso (el momento salta en las losas), con el")
    out("  armado que lleva el entrepiso (el corte cae en una losa); y el mayor D/C dentro de cada")
    out(f"  entrepiso (cada {D.PASO_DE_COTAS:.2f} m como máximo, más los planos del MEF):")
    out(f"    {'entrepiso':<10}{'nivel':<9}{'z (m)':>7}  {'armado':<10}{'estado':<18}{'Pu':>9}{'Mux':>10}"
        f"{'Muy':>9}{'D/C':>8}   resultado")
    cotas = dis.cotas()
    for s in range(len(D.ALTURAS)):
        arm = A.armado_en(dis, base, red, corte, s)
        for nombre, z in (("base", dem.z_base[s]), ("tope", dem.z_tope[s])):
            dc, e = A.dc_biaxial_en(dis, arm, z, s)
            out(f"    {D.PISOS[s]:<10}{nombre:<9}{z:7.2f}  {arm.nombre:<10}{e.nombre:<18}{e.P:9.1f}"
                f"{e.M:10.1f}{e.My:9.2f}{dc:8.3f}   {si_no(dc <= 1.0)}")
        peor = max(((*A.dc_biaxial_en(dis, arm, z, s), z) for z, piso, _ in cotas if piso == s),
                   key=lambda t: t[0])
        out(f"    {'':<10}→ el mayor del entrepiso: D/C = {peor[0]:.3f} en z = {peor[2]:.2f} m "
            f"({peor[1].nombre})  → {si_no(peor[0] <= 1.0)}")
    out("  (kN y kN·m; la figura dc_en_altura lo dibuja en toda la altura)")


def pisos_del_armado(dis, arm, corte: float) -> list[int]:
    """Los entrepisos donde va un armado: el de base, los que empiezan por
    debajo del corte; el reducido, los que acaban por encima (si el corte
    queda en el techo, el último, solo para informar)."""
    zb, zt = dis.demanda.z_base, dis.demanda.z_tope
    pisos = range(len(D.ALTURAS))
    if arm.nombre == "base":
        return [s for s in pisos if zb[s] < corte - A.TOL]
    return [s for s in pisos if zt[s] > corte + A.TOL] or [len(D.ALTURAS) - 1]


def nombre_por_P(e) -> str:
    """El estado nombrado por su axial (c solo depende de P)."""
    if e.paso in ("Max", "Min", ""):
        return e.nombre
    return f"{e.combo} (P de la fila {'Max' if e.paso[0] == '+' else 'Min'})"


def con_signo(v: float, decimales: int = 4) -> str:
    """El número con su signo, sin «−0.0000»."""
    v = round(v, decimales) + 0.0
    return f"{v:+.{decimales}f}"


def texto_de_ejes() -> str:
    """'Vx = V2, Mx = −M3; Vy = V3, My = −M2' según EJE_X y EJE_Y."""
    partes = []
    for nombre, (eje, signo) in zip("xy", mef.ejes()):
        momento = "M3" if eje == "2" else "M2"        # el que comprime la cara +2 / +3
        partes.append(f"V{nombre} = {'−' if signo < 0 else ''}V{eje}, "
                      f"M{nombre} = {'−' if signo > 0 else ''}{momento}")
    return "; ".join(partes)


def pisos_con_cargas_dentro(fuerzas) -> list[str]:
    """Los entrepisos donde ETABS tiene cargas entre losas (su tope no es la
    estática de la base: `mef.cargas_dentro_del_entrepiso`)."""
    return [p for s, p in enumerate(D.PISOS) if mef.cargas_dentro_del_entrepiso(fuerzas, s)]


def nota_de_cargas_dentro(out, fuerzas) -> None:
    """Si ETABS tiene cargas dentro de algún entrepiso, lo dice y cómo se trata."""
    pisos = pisos_con_cargas_dentro(fuerzas)
    if pisos:
        out(f"  OJO: en {', '.join(pisos)} la parte estática del tope de ETABS no es la de la base (más que")
        out("  el redondeo): ETABS tiene cargas dentro del entrepiso (un nivel intermedio, vigas a media")
        out("  altura…). Ahí la demanda es la ENVOLVENTE de dos lecturas, la estática desde la base y la")
        out("  estática desde el tope de ETABS (los estados «(tope)»), criterio del proyecto.")


def contraste_en_y(out, dem, crudas, fuerzas, dy: float) -> None:
    """Dirección Y (fuera del plano): ETABS tal como lo da (V y M del eje
    local que va a Y), pasado a los ejes del programa, frente a lo que usa el
    programa, en la base y el tope de cada entrepiso y para todos los estados
    de ETABS («+++» es la fila Max y «−−−» la Min). My del programa se
    compara en el mismo punto que ETABS (sin el traslado) y se da también
    llevado al centroide real, que es el que entra en el diseño."""
    (ey, sy) = mef.ejes()[1]
    nv, nm = f"V{ey}", ("M2" if ey == "3" else "M3")
    out("")
    out("  Fuerzas fuera del plano (dirección Y): ETABS frente al programa (kN, kN·m)")
    out(f"  EJE_Y = \"{D.EJE_Y}\": Vy = {'−' if sy < 0 else ''}{nv} y My = {'' if sy < 0 else '−'}{nm} (M < 0 "
        f"comprime la cara +Y), así que en ETABS uno de los dos")
    out("  sale con el signo cambiado. En el entrepiso, Vy es constante y My va por estática desde la base")
    out("  (la gravedad) y con la fórmula de arriba (el sismo). «My centroide» = My + P·dy, el que se diseña.")
    out(f"  {'estado':<16}{'piso':>7}{'':>6}{nv + ' ETABS':>10}{'Vy ETABS':>10}{'Vy prog.':>10}"
        f"{nm + ' ETABS':>10}{'My ETABS':>10}{'My prog.':>10}{'My centroide':>14}")
    peor_v = peor_m = 0.0
    for s in range(len(D.PISOS)):
        for donde, z, nombre_ub in (("Bottom", dem.z_base[s], "base"), ("Top", dem.z_tope[s], "tope")):
            est = {e.nombre: e for e in dem.en(z, s)}
            for (combo, paso), f in fuerzas.items():
                c, r = crudas[(combo, paso)][s][donde], f[s][donde]
                # la fila Max es el estado «+++» y la Min el «−−−» (sin
                # PERMUTAR_SIGNOS, los estados se llaman «Max» y «Min»)
                nombre = {"Max": f"{combo} +++", "Min": f"{combo} −−−"}.get(paso, combo)
                if donde == "Top" and dem.con_cargas_dentro[s]:
                    nombre += " (tope)"           # la lectura desde el tope de ETABS
                e = est.get(nombre) or est[f"{combo} {paso}".strip()]
                my_prog = e.My - e.P * dy             # en el punto de ETABS
                peor_v = max(peor_v, abs(e.Vy - r["Vy"]))
                peor_m = max(peor_m, abs(my_prog - r["My"]))
                out(f"  {(combo + ' ' + paso).strip():<16}{D.PISOS[s]:>7}{nombre_ub:>6}{c[nv]:10.2f}"
                    f"{r['Vy']:10.2f}{e.Vy:10.2f}{c[nm]:10.2f}{r['My']:10.2f}{my_prog:10.2f}{e.My:14.2f}")
    if max(peor_v, peor_m) <= 0.05:
        out(f"  Mayor diferencia: Vy {peor_v:.3f} kN, My {peor_m:.3f} kN·m (redondeo de ETABS)  → el programa "
            f"reproduce ETABS en Y.")
    else:
        out(f"  Mayor diferencia: Vy {peor_v:.3f} kN, My {peor_m:.3f} kN·m  ← más que el redondeo de ETABS: "
            f"revisar.")
    if any(dem.con_cargas_dentro):
        out("  (En el tope de los entrepisos con cargas dentro, el estado comparado es su lectura «(tope)».)")


def envolventes_frente_a_etabs(out, dis, fuerzas, dx: float, dy: float) -> None:
    """La envolvente de momentos de las figuras envolventes_X y envolventes_Y
    (el mínimo y el máximo de todos los estados, sin el traslado al centroide)
    frente a la de ETABS (el mínimo y el máximo de sus filas), en la base y el
    tope de cada entrepiso."""
    dem = dis.demanda
    out("")
    out("  Envolventes de momentos de las figuras envolventes_X y envolventes_Y frente a ETABS (kN·m;")
    out("  mínimo / máximo de todos los estados, en el punto de referencia de ETABS, sin el traslado):")
    out(f"    {'piso':<8}{'nivel':<6}{'Mx ETABS':>22}{'Mx figura':>22}{'My ETABS':>22}{'My figura':>22}")
    peor = 0.0
    for s in range(len(D.PISOS)):
        for donde, z, nombre in (("Bottom", dem.z_base[s], "base"), ("Top", dem.z_tope[s], "tope")):
            est = dis.estados_del_analisis(z, s)
            # en un entrepiso con cargas dentro, la figura lleva la envolvente
            # de las dos lecturas: pasa a ETABS del lado seguro
            dos_lecturas = dem.con_cargas_dentro[s]
            fila = []
            for comp, campo, d in (("Mx", "M", dx), ("My", "My", dy)):
                et = [f[s][donde][comp] for f in fuerzas.values()]
                fg = [getattr(e, campo) - e.P * d for e in est]
                if not dos_lecturas:
                    peor = max(peor, abs(min(et) - min(fg)), abs(max(et) - max(fg)))
                fila += [f"{min(et):9.2f} /{max(et):9.2f}", f"{min(fg):9.2f} /{max(fg):9.2f}"]
            out(f"    {D.PISOS[s]:<8}{nombre:<6}" + "".join(f"{t:>22}" for t in fila)
                + ("  *" if dos_lecturas else ""))
    out(f"    mayor diferencia {peor:.3f} kN·m" + (" (redondeo de ETABS)" if peor <= 0.05 else
                                                 "  ← más que el redondeo de ETABS: revisar"))
    if any(dem.con_cargas_dentro):
        out("    * entrepiso con cargas dentro: la figura lleva la envolvente de la lectura desde la base y")
        out("      desde el tope de ETABS, que contiene a la de ETABS (del lado seguro); no entra en la")
        out("      mayor diferencia.")


def seccion_y_ejes(out, sec) -> None:
    """La sección real, el modelo de ETABS, los ejes y la convención de signos."""
    out.titulo("SECCIÓN Y EJES")
    out("  Ejes del programa: x de 0 (extremo izquierdo) a ℓm (extremo derecho); y a través del")
    out("  espesor, hacia el ala; el alma va de y = −b/2 a +b/2.")
    out("  Sección real (rectángulos que no se solapan):")
    for p in sec.piezas:
        out(f"    x {p.x0:7.3f} a {p.x1:7.3f}   y {p.y0:+7.3f} a {p.y1:+7.3f}   "
            f"área {p.area:.4f} m²")
    out(f"    Ag = {sec.area:.4f} m²; centroide x = {sec.xc:.4f} m, y = {sec.yc:+.4f} m;  "
        f"Ix = {sec.Ix:.5f} m⁴, Iy = {sec.Iy:.5f} m⁴")
    if sec.tipo != "I":
        out("  Modelo de ETABS (cada muro, un rectángulo sobre su línea media; ETABS da los")
        out("  momentos respecto de su centro de gravedad):")
        for L, t, x, y in sec.muros_etabs:
            out(f"    muro de {L:.3f} m × {t:.3f} m con centro en ({x:.4f}, {y:+.4f})")
        Ae = sum(L * t for L, t, _, _ in sec.muros_etabs)
        out(f"    área {Ae:.4f} m²; centro de gravedad x = {sec.xe:.4f} m, y = {sec.ye:+.4f} m")
        out(f"  Traslado de los momentos al centroide real: dx = {con_signo(sec.xe - sec.xc)} m, "
            f"dy = {con_signo(sec.ye - sec.yc)} m  (M = M_ETABS + P·d)")
    d = mef.direcciones_globales()
    out("")
    out(f"  AxisAngle = {D.AXIS_ANGLE:g}°: +2 = {mef.nombre_global(d['2'])}, "
        f"+3 = {mef.nombre_global(d['3'])}")
    out(f"  EJE_X = \"{D.EJE_X}\" → +X del programa = {mef.nombre_global(d['X'])}: el extremo "
        f"izquierdo queda hacia {mef.nombre_global(-d['X'])} y el derecho hacia "
        f"{mef.nombre_global(d['X'])}")
    out(f"  EJE_Y = \"{D.EJE_Y}\" → +Y del programa = {mef.nombre_global(d['Y'])}"
        + (" (hacia donde sale el ala)" if sec.tipo != "I" else ""))
    filas = mef.comprobar_convencion()
    if not filas:
        out("  (sin CASOS_ESTATICOS: no se comprueba la convención de signos de ETABS)")
        return
    out("")
    out("  Convención de signos de ETABS, comprobada con los casos estáticos (base del primer piso).")
    out("  Cada caso empuja en + global: el cortante (V2·e2 + V3·e3) y la cara que comprime el")
    out("  momento (M3·e2 + M2·e3) tienen que ir hacia donde empuja la carga, como en un voladizo.")
    out(f"  {'caso':<20}{'paso':<9}{'V2':>8}{'V3':>8}{'M2':>9}{'M3':>9}{'V·empuje':>10}"
        f"{'M·empuje':>10}  resultado")
    for f in filas:
        r = f["registro"]
        out(f"  {f['caso']:<20}{f['paso']:<9}{r['V2']:8.1f}{r['V3']:8.1f}{r['M2']:9.1f}"
            f"{r['M3']:9.1f}{f['cortante']:10.1f}{f['momento']:10.1f}  {si_no(f['cumple'])}")
    if all(f["cumple"] for f in filas):
        out("  → la convención se cumple: V2 y V3 van con +2 y +3; M3 > 0 comprime la cara +2 y")
        out("    M2 > 0 la cara +3. Con EJE_X y EJE_Y, las fuerzas pasan a los ejes del programa.")
    else:
        out("  → LA CONVENCIÓN NO SE CUMPLE: revisa AXIS_ANGLE (y los nombres de CASOS_ESTATICOS).")


def tabla_de_elementos(out, armado) -> None:
    out("")
    out(f"  Elementos confinados del armado {armado.nombre.upper()} (caja en m; barras en el perímetro del")
    out("  estribo; r: del borde de la caja al eje de las barras, en las caras del muro y del lado del alma)")
    out(f"    {'elemento':<10}{'x':>15}{'y':>17}{'barra':>7}{'nx×ny':>7}{'r caras':>9}{'r alma':>8}"
        f"{'estribo':>14}{'grapas':>10}")
    for e in armado.elementos:
        c = e.caja
        est = "—" if e.estribo is None else f"{e.estribo}@{e.separacion_estribo * 100:.0f}"
        if e.separacion_arriba:
            est += f"/@{e.separacion_arriba * 100:.0f}"
        alma = "—" if e.hacia_el_alma is None else f"{e.a_eje(e.hacia_el_alma) * 100:.2f}"
        out(f"    {e.nombre:<10}{c.x0:7.3f}–{c.x1:<7.3f}{c.y0:+8.3f}–{c.y1:<+8.3f}{e.barra:>7}"
            f"{e.nx:>4}×{e.ny:<2}{e.r * 100:9.2f}{alma:>8}{est:>14}{e.grapas:>10}")
    if armado.nombre != "base":
        return                                  # las notas, una vez (valen para los dos armados)
    if any(e.separacion_arriba for e in armado.elementos):
        out("    (estribo «@s1/@s2»: @s1 hasta la altura de cambio y @s2 encima, hasta la cota de corte;")
        out("     la altura de cambio, en 3.3)")
    out("    (en los extremos, del lado del alma no hay recubrimiento: la cara exterior del estribo")
    out("     llega al borde de la zona confinada, la «longitud» declarada; en el reducido, ≤ la de la")
    out("     base)")
    if A.seccion().tipo == "III":
        out("    (tipo III: las barras del elemento derecho que caen dentro del ala no se ponen;")
        out("     allí mandan las del ala)")
        declarada = A.longitud_declarada("derecho")
        if declarada < A.seccion().ancho_ala - 1e-9:
            out(f"    (tipo III: la «longitud» del derecho, {declarada:.3f} m, no llega a b' = "
                f"{A.seccion().ancho_ala:.3f} m: se confina todo el ancho del ala)")


def tabla_de_barras(out, armado, barras) -> None:
    """Cada línea de barras verticales: dónde está, para pasarla a ETABS."""
    sec = A.seccion()
    out("")
    out(f"  Barras verticales del armado {armado.nombre} ({len(barras.x)} barras, "
        f"As = {barras.As * 1e4:.2f} cm²). Coordenadas en cm.")
    out(f"    {'línea':<6}{'grupo':<11}{'x':>8}{'x − xc':>9}{'barra':>7}{'nº':>4}   "
        f"y − yc de cada barra (yc = {sec.yc * 100:+.2f})")
    xs = np.unique(np.round(barras.x, 9))
    for i, x in enumerate(xs, 1):
        aqui = np.abs(barras.x - x) < 1e-9
        ys = sorted(np.round((barras.y[aqui] - sec.yc) * 100, 2))
        grupo = barras.zona[aqui][0]
        out(f"    {i:<6}{grupo:<11}{x * 100:8.2f}{(x - sec.xc) * 100:9.2f}{barras.barra[aqui][0]:>7}"
            f"{int(aqui.sum()):>4}   " + " ".join(f"{v:+.2f}" for v in ys))
    out("    (x desde el extremo izquierdo; x − xc e y − yc desde el centroide de la sección real,")
    out("     como las pide Section Designer de ETABS si el origen se pone en el centroide)")


def puntos_de_control(dis, base, red, corte):
    """Base y cada losa (cara de encima), con el armado de cada una; el corte
    cae en una losa (la que lo dice)."""
    salida = [("BASE z = 0.00 m", base, 0.0, 0)]
    for s in range(1, len(D.ALTURAS)):
        zs = dis.demanda.z_base[s]
        es_corte = red is not None and abs(zs - corte) <= 1e-6
        salida.append((f"{'CORTE' if es_corte else 'LOSA'} z = {zs:.2f} m",
                       A.armado_en(dis, base, red, corte, s), zs, s))
    return salida


def equilibrio(out, dis, arm, e) -> None:
    """El punto del diagrama con φPn = Pu, desarrollado línea a línea de barras."""
    sec = A.seccion()
    b = dis.barras(arm)
    diag = dis.diagrama(arm)
    sentido = 1 if e.M <= 0 else -1
    n = A.direccion(sentido)
    p_lim = diag.p_lim[sentido]

    def phi_de(p):
        return A.phi_por_axial(p.P, p_lim)

    bajo, alto = 1e-3 * D.LW, 10 * D.LW
    for _ in range(80):
        medio = 0.5 * (bajo + alto)
        p = A.estado_ultimo(b, n, medio)
        if phi_de(p) * p.P > e.P:
            bajo = medio
        else:
            alto = medio
    p = A.estado_ultimo(b, n, 0.5 * (bajo + alto), detalle=True)
    phi = phi_de(p)
    d = p.detalle
    out("")
    out(f"  EQUILIBRIO para comprobar a mano: el punto con φPn = Pu = {e.P:.1f} kN "
        f"(sentido {sentido:+d}, comprime el extremo {'derecho' if sentido > 0 else 'izquierdo'})")
    out(f"    c = {p.c:.4f} m   a = β1·c = {d['a']:.4f} m   área comprimida Ac = {d['Ac']:.5f} m²   "
        f"Cc = 0.85·f'c·Ac = 0.85·{D.FC:.3f}·1000·{d['Ac']:.5f} = {d['Cc']:.1f} kN")
    out(f"    centroide de Ac en x = {d['cx']:.4f} m → brazo de Cc = x − xc = {d['cx'] - sec.xc:+.4f} m;"
        f"   ε = 0.003·(1 − prof/c);  σ = Es·ε sin pasar de ±fy")
    out(f"    {'grupo':<11}{'x (m)':>8}{'prof. (m)':>10}{'ε':>10}{'σ (MPa)':>10}"
        f"{'nº':>4}{'As (cm²)':>10}{'F (kN)':>10}{'F·brazo':>10}")
    for x in np.unique(np.round(b.x, 9)):
        aqui = np.abs(b.x - x) < 1e-9
        k = int(np.argmax(aqui))
        As = float(b.A[aqui].sum())
        F = float(d["Fs"][aqui].sum())
        out(f"    {b.zona[k]:<11}{b.x[k]:8.3f}{d['prof'][k]:10.4f}{d['eps'][k]:10.5f}"
            f"{d['sigma'][k] / 1000:10.1f}{int(aqui.sum()):>4}{As * 1e4:10.2f}{F:10.1f}"
            f"{F * (b.x[k] - sec.xc):10.1f}")
    out("    (las barras de una misma línea x tienen la misma deformación; + compresión; σ de las")
    out("     barras dentro del bloque ya descuenta 0.85·f'c)")
    fs = float(d["Fs"].sum())
    mc = d["Cc"] * (d["cx"] - sec.xc)
    ms = float((d["Fs"] * (b.x - sec.xc)).sum())
    out(f"    Fuerzas:  Cc + ΣF = {d['Cc']:.1f} {'+' if fs >= 0 else '−'} {abs(fs):.1f} = "
        f"{-p.P:.1f} kN  → Pn = {p.P:.1f} kN")
    out(f"    Momentos respecto del centroide (x = {sec.xc:.3f} m): Cc·brazo + Σ(F·brazo) = "
        f"{mc:.1f} {'+' if ms >= 0 else '−'} {abs(ms):.1f} = {mc + ms:.1f} kN·m  → "
        f"Mn = {p.Mx:.1f} kN·m (el signo: − comprime el extremo derecho)")
    if p.detalle["tope"] < p.detalle["Cc"] + fs - 1e-6:
        out(f"    (llega al tope 0.80·P0 = {p.detalle['tope']:.1f} kN: P y M se escalan a él)")
    if p.P < 0 and p_lim > 0:
        out(f"    Pn = {-p.P:.1f} kN en compresión, P_lím = {p_lim:.1f} kN → φ = 0.90/(1 + 0.20·"
            f"{-p.P:.1f}/{p_lim:.1f}) = {phi:.3f} (§9.3.2.2, no menos de 0.70)")
    else:
        out(f"    φ = {phi:.3f} (§9.3.2.2)")
    mn_pol = A.phi_Mn(diag, e.P, 1.0 if e.M > 0 else -1.0)
    out(f"    φPn = {phi * p.P:.1f} kN   φMn = {phi * abs(p.Mx):.1f} kN·m  (el polígono de "
        f"{D.PUNTOS_DEL_DIAGRAMA} puntos da {mn_pol:.1f})")
    out(f"    A Pu constante: |Mu|/φMn = {abs(e.M):.1f}/{phi * abs(p.Mx):.1f} = "
        f"{abs(e.M) / (phi * abs(p.Mx)):.3f}  (el D/C radial, que usa el diseño, da "
        f"{A.dc_radial(diag, e.P, e.M):.3f})")


def equilibrio_de_borde(out, dis, arm, caso) -> None:
    """El estado nominal con Pn = Pu del caso de borde: de ahí sale c."""
    b = dis.barras(arm)
    n = np.array(caso["extremo"].n)
    p = A.eje_neutro(b, caso["estado"].P, n)
    p = A.estado_ultimo(b, n, p.c, detalle=True)
    d = p.detalle
    M = p.Mx
    out(f"    Comprobación de c ({nombre_por_P(caso['estado'])}, extremo {caso['zona']}): "
        f"Pn = Pu = {caso['estado'].P:.1f} kN")
    out(f"      c = {p.c:.4f} m, a = {d['a']:.4f} m, Cc = {d['Cc']:.1f} kN, ΣFs = "
        f"{float(d['Fs'].sum()):.1f} kN → ΣF = {d['Cc'] + float(d['Fs'].sum()):.1f} kN = −Pu; "
        f"Mn = {M:.1f} kN·m")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except A.DatoInvalido as error:
        print(f"\nDATO INVÁLIDO en {getattr(D, '__file__', DATOS_POR_DEFECTO)}: {error}", file=sys.stderr)
        sys.exit(2)
