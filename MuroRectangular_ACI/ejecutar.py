"""Diseño del muro con ACI 318-25: MEF + flexocompresión + corte + borde.

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

    for nombre in ("FC", "FY", "ES", "EC", "LW", "ESPESOR", "HN", "DELTA_U_MM", "MALLA_MEF",
                   "PASO_DE_COTAS"):
        problema = numero(nombre)
        if problema:
            return problema
    if d.FC < 17.0:
        return (f"FC = {d.FC:.2f} MPa: el mínimo de la tabla 19.2.1.1 es 17 MPa (y 21 MPa en muros "
                f"especiales)")
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
    if d.HN < sum(alturas) - 1e-6:
        return (f"HN = {d.HN:g} m (altura del edificio) es menor que la del muro, "
                f"{sum(alturas):g} m (la suma de ALTURAS)")
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
import diseno_aci as A  # noqa: E402
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
               f"ACI 318-25, muro estructural especial")
    out(f"  f'c = {D.FC:.2f} MPa ({D.FC / 0.0980665:.0f} kgf/cm²)   fy = {D.FY:.2f} MPa "
        f"({D.FY / 0.0980665:.0f} kgf/cm²)   Es = {D.ES:.0f} MPa   Ec = {D.EC:.0f} MPa (§19.2.2.1)")
    for aviso in A.avisos_de_datos():
        out(f"  AVISO: {aviso}")
    out(f"  β1 = {A.BETA1:.3f} (tabla 22.2.2.4.3)   εcu = {A.EPS_CU}   "
        f"εty = fy/Es = {A.EPS_TY:.5f}")
    out(f"  ℓw = {sec.lw:.3f} m   b = {sec.b:.3f} m   Ag = {sec.area:.4f} m²   "
        f"Acv = ℓw·b = {sec.acv:.4f} m²   H = {A.H:.2f} m "
        f"({' + '.join(f'{h:.2f}' for h in D.ALTURAS)})   hw/ℓw = {A.ESBELTEZ:.2f}")
    if sec.tipo != "I":
        out(f"  ala: b' = {sec.ancho_ala:.3f} m, l'w = {sec.largo_ala:.3f} m (desde la cara exterior "
            f"del alma), de x = {sec.ala.x0:.3f} a {sec.ala.x1:.3f} m")
    out(f"  Recubrimiento libre {D.RECUBRIMIENTO * 100:.1f} cm; malla horizontal "
        f"{'por fuera' if D.HORIZONTAL_POR_FUERA else 'por dentro'} de la vertical; "
        + (f"agregado {D.AGREGADO_MM:.0f} mm" if getattr(D, "AGREGADO_MM", None)
           else "agregado no declarado (§25.2.3 sin el término de 4/3·dagg)"))

    seccion_y_ejes(out, sec)

    out.titulo("ARMADO")
    armados = [x for x in (base, red) if x is not None]
    for arm in armados:
        out(f"  {arm.nombre.upper():<9} {arm.texto()}   → {A.kg_por_metro(arm):.2f} kg/m vertical")
        out(f"  {'':<9} malla horizontal {arm.malla_horizontal.texto()}, ρt = "
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
    separacion = mef.leer_separacion_del_corte(fuerzas)
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
    dem = mef.Demanda(res, fuerzas, traslado, separacion)
    dis = A.Diseno(dem, D.DELTA_U_MM / 1000.0)
    out("")
    if any(abs(t) > 1e-12 for t in traslado):
        out("  Momentos llevados al centroide de la sección real (ver SECCIÓN Y EJES):")
        out(f"    Mx = Mx_ETABS + P·({sec.xe:.4f} − {sec.xc:.4f}) = Mx_ETABS + P·({con_signo(traslado[0])})   "
            f"My = My_ETABS + P·({sec.ye:.4f} − {sec.yc:.4f}) = My_ETABS + P·({con_signo(traslado[1])})")
    contraste_en_y(out, dem, crudas, fuerzas, traslado[1])
    envolventes_frente_a_etabs(out, dis, fuerzas, *traslado)
    out("")
    out(f"  Desplazamiento de diseño del techo en el plano del muro (archivo de datos): δu = "
        f"{D.DELTA_U_MM:.2f} mm, δu/H = {dis.deriva:.5f}")
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
    out("  En el cortante, Vg y VEh salen por separado de ETABS: la combinación de gravedad y el caso")
    out("  de espectro de cada combinación (SEPARACION_DEL_CORTE, ver 2.1).")
    out("")
    out("  Demanda en la BASE (z = 0), con el cortante de diseño Ve del módulo 2:")
    out(f"  {'estado':<18}{'Pu':>9}{'Mux':>10}{'Vux':>9}{'Ve':>9}{'Muy':>10}")
    for e in est0:
        out(f"  {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{e.V:9.1f}{e.Ve:9.1f}{e.My:10.2f}")
    out("  (kN y kN·m; x = en el plano del alma, y = a través del espesor)")

    # =========================================================================
    # MÓDULO 1 · FLEXOCOMPRESIÓN
    # =========================================================================
    out.parte(MODULO_1)
    out.modulo("MÓDULO 1 · DISEÑO A FLEXOCOMPRESIÓN")
    # la región crítica (y φ del corte) se decide aquí, en la base, con el
    # armado de base, y rige en las demás cotas
    vb = dis.verificar(base, 0.0, 0)
    b0 = dis.barras(base)
    z_ext = dis.z_extremo
    # la cota de corte se calcula ya (se explica en 1.6): dice en qué
    # entrepisos va cada armado
    corte = A.H
    cc = None
    if red is not None:
        cc = A.cota_de_corte(dis, base, red, vb.dc_biaxial, z_ext)
        corte = cc.corte
    # ¿pide el cortante refuerzo? (§11.6, §11.7): con el entrepiso de mayor |Ve|
    # de los que lleva cada armado
    refuerzos = {arm.nombre: refuerzo_de(dis, arm, pisos_del_armado(dis, arm, corte))
                 for arm in armados}

    out.titulo("1.1 CORTINAS DE REFUERZO — §18.10.2.2")
    vu = max(abs(e.V) for e in est0)
    umbral = 0.17 * A.LAMBDA * A.RAIZ_FC * mef.MPA * sec.acv
    out("  Dos cortinas si Vu > 0.17·λ·√f'c·Acv  o  hw/ℓw ≥ 2.0 (basta una). Vu es el mayorado,")
    out("  sin amplificar (la norma escribe ωv·Ωv·Vu cuando quiere el amplificado):")
    out(f"    Vu = {vu:.1f} kN  frente a  0.17·1.0·√{A.FC_RAIZ:.2f}·{sec.acv:.3f}·1000 = "
        f"{umbral:.1f} kN  → {'SÍ' if vu > umbral else 'no'}")
    out(f"    hw/ℓw = {A.ESBELTEZ:.2f}  frente a 2.0  → {'SÍ' if A.ESBELTEZ >= 2.0 else 'no'}")
    exige2 = vu > umbral or A.ESBELTEZ >= 2.0
    for arm in armados:
        mv, mh = arm.malla_vertical, arm.malla_horizontal
        out(f"  Declaradas, {arm.nombre:<9}: {mv.cortinas} vertical, {mh.cortinas} horizontal  → "
            f"{si_no(not exige2 or (mv.cortinas >= 2 and mh.cortinas >= 2))}")

    cuantia_longitudinal(out, dis, base, red, refuerzos)

    out.titulo("1.3 DIAGRAMA DE INTERACCIÓN P-Mx EN LA BASE (REFERENCIA) — §22.2, tabla 21.2.2 y "
               "§21.2.2.3")
    out("  La flexocompresión se verifica con el D/C BIAXIAL (1.4), en cada nivel. Este diagrama, en el")
    out("  plano del muro (eje neutro paralelo a Y), queda de referencia: da φ, Pn,bal y el equilibrio")
    out("  para comprobar a mano, y su D/C no entra en el veredicto.")
    diag = dis.diagrama(base)
    p0 = A.p0_nominal(b0)
    out(f"  As = {b0.As * 1e4:.2f} cm²   P0 = 0.85·f'c·(Ag − As) + fy·As = {p0:.1f} kN   "
        f"tope 0.80·P0 = {0.8 * p0:.1f} kN (§22.4.2.1)")
    out(f"  Tracción pura: fy·As = {D.FY * mef.MPA * b0.As:.1f} kN.   "
        f"Tope de φ (§21.2.2.3) entre 0.1·f'c·Ag = {0.1 * D.FC * mef.MPA * sec.area:.1f} kN y "
        f"Pn,bal = {-diag.p_bal[1]:.1f} kN (sentido +1) y {-diag.p_bal[-1]:.1f} kN (sentido −1), "
        f"en compresión,")
    out("  con c_bal = dt·εcu/(εcu + εty), dt la profundidad de la barra más traccionada. La "
        "tracción pura actúa en el centroide del acero.")
    out(f"  c de 0.02·ℓw a 4·ℓw en {D.PUNTOS_DEL_DIAGRAMA} pasos geométricos, por sentido; eje "
        f"neutro paralelo a Y. Sentido +1 comprime el extremo derecho (M < 0). Momentos")
    out(f"  respecto del centroide de la sección real (x = {sec.xc:.4f} m).")
    out(f"  {'#':>3}{'sent.':>6}{'c (m)':>8}{'Pn':>10}{'Mn':>10}{'εt':>9}{'φ':>7}"
        f"{'φPn':>10}{'φMn':>10}")
    for i, p in enumerate(diag.puntos):
        sentido = "" if math.isinf(p.c) else f"{int(p.n[0]):+d}"
        c = "∞" if math.isinf(p.c) else f"{p.c:.4f}"
        out(f"  {i:3d}{sentido:>6}{c:>8}{p.P:10.1f}{p.Mx:10.1f}{p.eps_t:9.5f}{p.phi:7.3f}"
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

    out.titulo("1.4 FLEXOCOMPRESIÓN BIAXIAL EN CADA NIVEL — §22.2 (EL D/C QUE SE VERIFICA)")
    out(f"  Eje neutro girado {D.ANGULOS_BIAXIAL} veces (cada {360 / D.ANGULOS_BIAXIAL:.0f}°) "
        f"con {D.PROFUNDIDADES_BIAXIAL} profundidades; zona comprimida exacta (polígonos).")
    out("  A cada Pu se corta la superficie: D/C = cuánto hay que escalar (Mux, Muy) hasta el contorno.")
    out("  En la base, todos los estados:")
    sup = dis.superficie(base)
    out(f"  {'estado':<18}{'Pu':>9}{'Mux':>10}{'Muy':>9}{'D/C':>8}")
    for e in est0:
        out(f"  {e.nombre:<18}{e.P:9.1f}{e.M:10.1f}{e.My:9.2f}{sup.dc(e.P, e.M, e.My):8.3f}")
    out(f"  GOBIERNA {vb.estado_biaxial.nombre}: D/C = {vb.dc_biaxial:.3f}  → "
        f"{si_no(vb.dc_biaxial <= 1.0)}")
    biaxial_por_nivel(out, dis, base, red, corte)

    out.titulo("1.5 REFUERZO DE EXTREMO — §18.10.2.4")
    req = A.cuantia_exigida_de_extremo()
    out(f"  Aplica: hw/ℓw = {A.ESBELTEZ:.2f} {'≥' if A.ESBELTEZ >= 2 else '<'} 2.0")
    out(f"  (a) dentro de 0.15·ℓw = {0.15 * D.LW:.3f} m de cada extremo y en un ancho igual al "
        f"espesor b, ρ ≥ 0.5·√f'c/fy = 0.5·√{D.FC:.2f}/{D.FY:.0f} = {req:.5f}")
    out("      (se suman las barras cuyo eje cae en esa franja, dentro del espesor del alma)")
    for arm in [x for x in (base, red) if x is not None]:
        bz = dis.barras(arm)
        # el reducido empieza en el corte: (a) rige en él solo si el corte
        # queda por debajo de la altura de (b)
        if A.ESBELTEZ < 2.0:
            no_rige = "no rige: hw/ℓw < 2"
        elif arm is red and corte >= z_ext - A.TOL:
            no_rige = f"no rige: el reducido empieza en z = {corte:.2f} m, encima de {z_ext:.2f} m"
        else:
            no_rige = None
        for nombre in ("izquierdo", "derecho"):
            r = A.cuantia_de_extremo(bz, nombre)
            dist = bz.x if nombre == "izquierdo" else D.LW - bz.x
            dentro = (dist <= 0.15 * D.LW + 1e-9) & (np.abs(bz.y) <= D.ESPESOR / 2 + 1e-9)
            n = int(dentro.sum())
            out(f"      {arm.nombre:<9} {nombre:<10}: {n} barras, "
                f"{bz.A[dentro].sum() * 1e4:.2f} cm²/({0.15 * D.LW * 100:.1f}·"
                f"{D.ESPESOR * 100:.0f}) = {r:.5f}  → {no_rige or si_no(r >= req)}")
    out("  (b) ese acero sube sobre la sección crítica la mayor de ℓw y Mu/3Vu, con el estado que "
        "gobierna la")
    out("      flexocompresión biaxial en la base y su Vu mayorado SIN amplificar:")
    gb = vb.estado_biaxial
    if gb.Vu:
        out(f"      {gb.nombre}: Mu/3Vu = {abs(gb.M):.1f}/(3·{abs(gb.Vu):.1f}) = "
            f"{abs(gb.M) / (3 * abs(gb.Vu)):.2f} m  → máx(ℓw, Mu/3Vu) = {z_ext:.2f} m")
    out(f"      (por eso (a) se comprueba hasta z = {z_ext:.2f} m; encima ya no rige)")

    if cc is not None:
        if A.ESBELTEZ >= 2.0:
            out(f"      el reducido no empieza por debajo de {z_ext:.2f} m (es una de las cotas de 1.6)")
        out(f"  (c) en el corte se termina el {cc.fraccion_terminada:.0%} del acero exigido por "
            f"(a), frente al 50 % permitido  → {si_no(cc.fraccion_terminada <= 0.5 + 1e-12)}")

    out.titulo("1.6 COTA DE CORTE DEL ARMADO REDUCIDO — §18.10.2.3(a), §18.10.2.4(b) y §18.10.6.2(b)(i)")
    if red is None:
        out("  No se declaró armado reducido: el acero de base va en toda la altura.")
    else:
        cotas = dis.cotas()
        out("  El reducido empieza en el entrepiso inmediatamente encima de la MAYOR de tres cotas de la")
        out("  norma (criterio del proyecto) y encima de donde falla por sí mismo. Sus barras verticales")
        out("  continúan barras de la base (criterio del proyecto, ver ARMADO).")
        out("")
        out("  (1) §18.10.2.3(a), ACERO VERTICAL. Se busca z*: la cota MÁS BAJA desde la que el reducido")
        out(f"  cumple la flexocompresión biaxial en todo lo que hay encima. Se baja desde el techo por "
            f"{len(cotas)} cotas (cada {D.PASO_DE_COTAS:.2f} m")
        out("  como máximo, más los planos del MEF) y el cruce se interpola. El acero de base sigue 3.60 m")
        out("  más arriba:")
        filas = [("flexocompresión biaxial D/C ≤ 1 [22.2]", cc.z_biaxial),
                 (f"sección crítica única: D/C biaxial ≤ el de la base = {vb.dc_biaxial:.3f} "
                  f"[18.10.6.2]", cc.z_unica)]
        for texto, z in filas:
            marca = "  <- MANDA" if abs(z - cc.z_estrella) < 1e-9 and z > 0 else ""
            out(f"    {z:6.2f} m  {texto}{marca}")
        out(f"    z* = {cc.z_estrella:.2f} m  +  3.60 m  =  {cc.z_vertical:.2f} m")
        out(f"    (sin la regla de sección crítica única sería {cc.z_biaxial + A.PROLONGACION:.2f} m)")
        out("  (2) §18.10.2.4(b), ACERO DE EXTREMO (1.5): "
            + (f"sube máx(ℓw, Mu/3Vu) = {cc.z_extremo:.2f} m" if A.ESBELTEZ >= 2.0
               else f"no rige (hw/ℓw = {A.ESBELTEZ:.2f} < 2)"))
        if cc.z_borde > 0.0:
            regla = "máx(ℓw, Mu/4Vu)" if cc.art_borde == "18.10.6.2(b)(i)" else "hasta σ < 0.15·f'c"
            out(f"  (3) §{cc.art_borde}, BORDE ESPECIAL (3.1): sube {regla} = {cc.z_borde:.2f} m")
        else:
            out("  (3) §18.10.6.2(b)(i), BORDE ESPECIAL (3.1): no se exige; no condiciona")
        art, z_norma = max(cc.cotas_de_la_norma, key=lambda t: t[1])
        out(f"    la mayor: {z_norma:.2f} m (§{art})")
        out("")
        out("  ACERO HORIZONTAL Y DETALLADO. En todas las cotas, el reducido frente al de base: el cortante")
        out("  (con su malla horizontal), la longitud del borde, los estribos (separación, hx, Ash, diámetro)")
        out("  y las cuantías y separaciones de las mallas. El reducido no puede ir donde falla por él (lo que")
        out("  falla igual con el de base se arregla aparte y sale en la verificación completa); el corte")
        out("  queda encima de la cota más alta donde falla:")
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
            f"{cc.z_detallado:.2f})   (gobierna: {cc.gobierna})")
        out("  El armado cambia en la losa más próxima por encima del corte requerido, al empezar el "
            "entrepiso")
        out("  inmediatamente superior (criterio del proyecto):")
        if cc.corte < A.H - A.TOL:
            out(f"  CORTE A {cc.corte:.2f} m: el reducido va desde el {D.PISOS[dis.piso_de(cc.corte)]} "
                f"hasta el techo; el empalme, en esa losa")
            hsx = min(D.ALTURAS[0], 6.0)
            if cc.corte >= hsx - 1e-9:
                out(f"  El traslape queda fuera de hsx = {hsx:.2f} m sobre la base, donde §18.10.2.3(c) "
                    f"no permite traslapes en las zonas de borde: CUMPLE")
            else:
                out(f"  OJO: la losa de {cc.corte:.2f} m está dentro de hsx = {hsx:.2f} m sobre la base: "
                    f"§18.10.2.3(c) no permite traslapes ahí en las zonas de borde (subir el empalme "
                    f"o usar empalme mecánico, §18.2.7)")
            out("  (§18.10.2.3(a) permite además no pasar de ℓd sobre la losa inmediatamente "
                "encima de z*; ese tope no se")
            out("  aplica —ℓd no se calcula—: del lado seguro)")
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
    out.titulo("2.1 CORTANTE DE DISEÑO — §18.10.3.3")
    amp = A.amplificacion()
    out(f"  hwcs/ℓw = {amp['esbeltez']:.2f} → Ωv = {amp['omega_v']:.2f}   (tabla 18.10.3.3.3)")
    out(f"  hn = {D.HN:.2f} m = {amp['hn_pies']:.1f} pies  → ωv = "
        + (f"0.8 + 0.09·hn^(1/3) = {amp['omega_dyn']:.3f}  (≥ 1.0)" if amp["esbeltez"] >= 2.0
           else "1.0 (hwcs/ℓw < 2)"))
    out(f"  §18.10.3.3.2: se amplifica solo la parte de sismo, VuEh:  Ve = Vg + Ωv·ωv·VEh,  "
        f"Ωv·ωv = {amp['factor']:.3f}")
    out("  Para eso, solo en el corte, cada combinación con sismo se separa con ETABS")
    out("  (SEPARACION_DEL_CORTE): Vg, de su combinación de gravedad; VEh, de su caso de espectro,")
    out("  con el signo del sismo en el estado.")
    tabla_de_amplificacion(out, est0, amp["factor"], dis.demanda.corte)
    sin_sismo = sorted({e.combo for e in est0 if not e.con_sismo})
    if sin_sismo:
        out(f"  {', '.join(sin_sismo)}: sin sismo, no se amplifica (Ve = Vu).")
    out("  (la alternativa Ωv = Mpr/Mu no se usa: pide ASCE/SEI 7 §16.3.2)")

    out.titulo("2.2 RESISTENCIA A CORTANTE EN EL PLANO — §18.10.4 y §21.2.4.1")
    r0 = A.resistencia_en_el_plano(base)
    out(f"  Vn = (αc·λ·√f'c + ρt·fy)·Acv;  Acv = ℓw·b = {sec.acv:.4f} m²;  hw/ℓw = "
        f"{r0['esbeltez']:.2f} → αc = {r0['alpha_c']:.2f}; ρt = {r0['rho_t']:.5f}")
    out(f"  Vc = {r0['alpha_c']:.2f}·1.0·√{A.FC_RAIZ:.2f}·{sec.acv:.3f}·1000 = {r0['Vc']:.1f} kN   "
        f"Vs = {r0['rho_t']:.5f}·{D.FY:.2f}·{sec.acv:.3f}·1000 = {r0['Vs']:.1f} kN   "
        f"Vn = {r0['Vc'] + r0['Vs']:.1f} kN")
    if red is not None and red.malla_horizontal != base.malla_horizontal:
        rr = A.resistencia_en_el_plano(red)
        out(f"  Reducido ({red.malla_horizontal.texto()}): ρt = {rr['rho_t']:.5f}; Vs = "
            f"{rr['Vs']:.1f} kN; Vn = {rr['Vc'] + rr['Vs']:.1f} kN (con el αc de cada entrepiso, abajo)")
    out(f"  Tope §18.10.4.4: 0.83·√f'c·Acv = {r0['tope']:.1f} kN (αsh = 1.0), el de un muro; el de "
        f"0.66·√f'c·ΣAcv de todos los muros")
    out("  que comparten la fuerza lateral no se comprueba (hace falta el edificio entero).")
    out("  αc no depende de la carga axial (§18.10.4.1).")
    fc = dis.phi_corte
    if not fc["aplica"]:
        out(f"  §21.2.4.1: φ = 0.60 no aplica en muros con Ωv ≥ 1.5 (aquí Ωv = {fc['omega_v']:.2f}) "
            f"→ φ = {fc['phi']:.2f}")
    else:
        out(f"  §21.2.4.1: Mn máximo con los axiales de las combinaciones con sismo = {fc['Mn']:.1f} "
            f"kN·m; el cortante que lo desarrolla, Mn·Vu/Mu = {fc['V_flexion']:.1f} kN, frente a "
            f"Vn = {fc['Vn']:.1f} kN → φ = {fc['phi']:.2f}")
    out("  D/C = |Ve|/(φ·Vn) en la base:")
    out(f"    {'estado':<18}{'Pu':>9}{'Ve':>9}{'Vn':>9}{'φVn':>9}{'D/C':>7}")
    for e in est0:
        c = A.corte_en_el_plano(base, e, fc["phi"])
        out(f"    {e.nombre:<18}{e.P:9.1f}{e.Ve:9.1f}{c['Vn']:9.1f}{c['phiVn']:9.1f}{c['dc']:7.3f}"
            + ("" if e.con_sismo else "   (Vu, sin amplificar)"))
    for etiqueta, arm, z, piso in puntos_de_control(dis, base, red, corte):
        v = dis.verificar(arm, z, piso)
        out(f"  {etiqueta:<20} ({arm.nombre:<8}): gobierna {v.estado_corte.nombre}, "
            f"|Ve| = {abs(v.estado_corte.Ve):.1f} kN, φVn = {v.corte['phiVn']:.1f} kN, "
            f"D/C = {v.dc_corte:.3f}  → {si_no(v.dc_corte <= 1)}")

    out("  El corte en Y (fuera del plano del alma) no se verifica: en esa dirección el muro no trabaja")
    out("  como muro" + ("; las alas son columnas de los pórticos especiales y su cortante, §18.7.6, va "
                       "con el pórtico." if sec.tipo != "I" else "."))

    filas_corte = cortante_por_piso(out, dis, base, red, corte, amp["factor"])
    cuantia_transversal(out, dis, base, red, corte, refuerzos)

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
    out(f"  {'cota':<22}{'armado':<10}{'biaxial':>8}{'corte':>7}"
        f"{'borde':>7}{'detallado':>11}   resultado")
    verifs = []
    for etiqueta, arm, z, piso in puntos_de_control(dis, base, red, corte):
        v = dis.verificar(arm, z, piso)
        verifs.append((etiqueta, v))
        det = "ok" if not v.faltas else f"{len(v.faltas)} falla(s)"
        out(f"  {etiqueta:<22}{arm.nombre:<10}{v.dc_biaxial:8.3f}"
            f"{v.dc_corte:7.3f}{v.dc_borde:7.3f}{det:>11}   "
            f"{si_no(v.cumple)}")
        for f in v.faltas:
            out(f"      [{f['art']}] {f['texto']}: {f['valor']:.4g} frente a {A.texto_limite(f['limite'])}")
    out("  (D/C; borde = longitud exigida / confinada donde se exige elemento de borde especial)")
    # entre losas: el biaxial del armado de base en todas las cotas bajo el
    # corte (encima, el del reducido ya lo asegura la cota de corte)
    bajo = A.revisar_bajo_el_corte(dis, base, corte, vb.dc_biaxial)
    entre_losas = not any(bajo[k] for k in ("falla_unica", "falla_biaxial"))
    out(f"  Entre losas (todas las cotas, armado de base{' bajo el corte' if red is not None else ''}): "
        f"mayor D/C biaxial {bajo['peor_dc']:.3f} en z = {bajo['peor_z']:.2f} m  → {si_no(entre_losas)}")
    out("")
    out("  Detallado en la base (todas las comprobaciones de armado y, donde hay borde especial, sus")
    out("  dimensiones):")
    for f in vb.detalle:
        if not f.get("aplica", True):
            continue                         # p. ej. §18.10.6.4(c) con c/ℓw < 3/8
        resultado = si_no(f["cumple"])
        if not f["cumple"] and f["art"] == "18.10.6.2(b)(ii)":
            resultado = "no: rige (iii)"
        out(f"    [{f['art']:<18}] {f['texto']:<68} {f['valor']:10.4g} {f['tipo']:>5} "
            f"{A.texto_limite(f['limite']):<10}  {resultado}")
    out("")
    out("  Para el plano (lo que la norma pide dibujar y no es una cifra que se compruebe):")
    for art, texto in A.detalles_constructivos(dis, base, vb, red):
        out(f"    [{art}] {texto}")

    todo = all(v.cumple for _, v in verifs)
    convencion = mef.comprobar_convencion()
    todo &= all(f["cumple"] for f in convencion)
    todo &= entre_losas
    if cc is not None:
        todo &= cc.fraccion_terminada <= 0.5 + 1e-12
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


REGLA = {"especial": "ESPECIAL §18.10.6.4", "ordinario": "ordinario §18.10.6.5(b)",
         "proyecto": "proyecto (ρ ≤ 2.8/fy)", "ala central": "proyecto (ala central)"}


def modulo_3(out, dis, base, red, corte: float, vb) -> None:
    """Módulo 3: necesidad del borde especial, sus dimensiones, los estribos
    de cada elemento por zonas y el resumen de §18.10.6.4 y §18.10.6.5."""
    sec = A.seccion()
    b0 = dis.barras(base)
    out.modulo("MÓDULO 3 · ELEMENTOS DE BORDE")
    out("  Los extremos se confinan siempre (en la base, longitud declarada ≥ 0.15·ℓw, criterio del")
    out("  proyecto; en el reducido, no más que en la base, sin ese mínimo). Aquí:")
    out("  3.1 decide si la norma EXIGE elemento de borde especial; 3.2 comprueba sus dimensiones (o, sin")
    out("  él, la longitud de §18.10.6.5(b)); 3.3 diseña los estribos de cada elemento por zonas en altura,")
    out("  con borde especial u ordinario; 3.4 resume §18.10.6.4 y §18.10.6.5. Solo en la dirección larga")
    out("  (X): en Y el muro no trabaja como muro"
        + ("; las alas cortas son columnas de los pórticos especiales (§18.7)." if sec.tipo != "I"
           else "."))

    # ---- 3.1 -------------------------------------------------------------------
    casos = vb.casos_borde
    c0 = casos[0]
    por_desplazamiento = c0["via"] == "18.10.6.2"
    out.titulo(f"3.1 ¿SE EXIGE ELEMENTO DE BORDE ESPECIAL? — §{c0['via']} (sección crítica en la base)")
    out(f"  ℓw = {c0['lw']:.3f} m;  hwcs/ℓw = {c0['hwcs'] / c0['lw']:.2f} "
        f"{'≥' if por_desplazamiento else '<'} 2.0 → vía §{c0['via']}")
    if por_desplazamiento:
        out(f"  δu/hwcs = {dis.deriva:.5f} → se toma {c0['deriva']:.4f} (no menor que 0.005)")
        out(f"  Se exige si 1.5·δu/hwcs ≥ ℓw/(600·c), es decir si c ≥ c_lím = "
            f"ℓw/(600·1.5·δu/hwcs) = {c0['c_lim']:.3f} m")
    else:
        out("  Vía por esfuerzo: se exige si la compresión elástica de la fibra extrema, con la "
            "sección bruta y las")
        out("  combinaciones del sismo, llega a 0.2·f'c (índice = σ/(0.2·f'c)); c solo da la longitud.")
    out(f"  c al axial mayorado y Mn, combinaciones del sismo en el plano "
        f"({', '.join(D.COMBINACIONES_DEL_SISMO_EN_X)}):")
    for caso in casos:
        out(f"  extremo {caso['zona']} (comprimido):")
        for e, c in caso["por_estado"]:
            out(f"    {nombre_por_P(e):<26} Pu = {e.P:8.1f} kN   c = {c:.3f} m")
        out(f"    el mayor: c = {caso['c']:.3f} m ({nombre_por_P(caso['estado'])})  →  índice "
            + (f"c/c_lím = {caso['indice']:.3f}" if por_desplazamiento
               else f"σ/(0.2·f'c) = {caso['indice']:.3f}") + "  → "
            + ("SE EXIGE elemento de borde especial" if caso["exige"]
               else "no se exige elemento de borde especial"))
    equilibrio_de_borde(out, dis, base, max(casos, key=lambda cs: cs["indice"]))
    g = c0["gobierna"]
    altura = dis.altura_critica
    alto = {zn: alt for zn, (ex, alt, _) in dis.region.items() if ex}   # altura del borde especial
    out(f"  Región crítica (§18.10.6.2(b)(i)): sube sobre la sección crítica máx(ℓw, Mu/4Vu), con "
        f"{g.nombre} (el de mayor |Mu|")
    out("  y su Vu mayorado sin amplificar): "
        + (f"Mu/4Vu = {abs(g.M):.1f}/(4·{abs(g.V):.1f}) = {abs(g.M) / (4 * abs(g.V)):.2f} m"
           if g.V else "Vu = 0") + f" → {altura:.2f} m.")
    if por_desplazamiento:
        out("  Ahí van los estribos del borde especial donde se exige y, donde no, la fila «región crítica»")
        out("  de la tabla 18.10.6.5(b).")
    else:
        out("  Ahí rige la fila «región crítica» de la tabla 18.10.6.5(b). El borde especial de la vía por")
        out("  esfuerzo sube hasta donde la compresión baja de 0.15·f'c (y no menos que esa altura):")
        for zn, alt in alto.items():
            out(f"    extremo {zn}: hasta {alt:.2f} m")
        out("  Si por encima vuelve a llegar a 0.2·f'c, la verificación completa lo marca como falta (el")
        out("  programa modela el borde especial solo desde la base).")
    if alto:
        out("  Por debajo de la base, los estribos del borde especial entran en el apoyo (§18.10.6.4(j),")
        out("  nota del plano).")

    # ---- 3.2 -------------------------------------------------------------------
    out.titulo("3.2 DIMENSIONES DEL BORDE — §18.10.6.4(a)–(d), §18.10.6.2(b)(ii),(iii) y §18.10.6.5(b)")
    resumen: dict = {}

    def anota(art: str, cumple: bool) -> None:
        resumen.setdefault(art, []).append(cumple)

    limite_rho = 2.8 / D.FY
    for caso in casos:
        ext = caso["extremo"]
        c = caso["c"]
        rho = A.cuantia_de_borde(base, b0, ext)
        if caso["exige"]:
            hasta = alto[ext.nombre]
            pisos_region = [s for s in range(len(D.ALTURAS)) if dis.demanda.z_base[s] < hasta - 1e-9]
            out(f"  extremo {ext.nombre}: BORDE ESPECIAL hasta {hasta:.2f} m sobre la base "
                f"(c = {c:.3f} m en la sección crítica)")
            for s in pisos_region:
                for f in A.dimensiones_de_borde(caso, s):
                    clave = f["art"]
                    if s != pisos_region[0] and clave != "18.10.6.4(b)":
                        continue                # solo hu cambia con el entrepiso
                    anota(clave, f["cumple"])
                    if clave == "18.10.6.4(a)":
                        out(f"    (a) longitud ≥ {f['nota']} = {f['limite']:.3f} m;  confinada "
                            f"{f['valor']:.3f} m  → {si_no(f['cumple'])}")
                    elif clave == "18.10.6.4(b)":
                        out(f"    (b) ancho b de la zona comprimida en esa longitud (el menor, con el ala "
                            f"si la hay) = {f['valor']:.3f} m ≥ hu/16:")
                        out(f"        {f['nota']}  → {si_no(f['cumple'])}")
                    elif clave == "18.10.6.4(c)":
                        out(f"    (c) {f['nota']}"
                            + (f": b = {f['valor']:.3f} m ≥ 0.300 m  → {si_no(f['cumple'])}"
                               if f["aplica"] else ""))
                    elif clave == "18.10.6.2(b)(ii)":
                        out(f"    §18.10.6.2(b)(ii): b medio en c = {f['valor']:.3f} m ≥ {f['nota']}  → "
                            + (si_no(True) + " (no hace falta (iii))" if f["cumple"]
                               else "no se cumple: vale si se cumple (iii)"))
                    elif clave == "18.10.6.2(b)(iii)":
                        out(f"    §18.10.6.2(b)(iii): {f['nota']}:")
                        out(f"      δc/hwcs = {f['valor']:.4f} ≥ 1.5·δu/hwcs = {f['limite']:.4f}  → "
                            f"{si_no(f['cumple'])}")
            if sec.tipo == "III" and ext.nombre == "derecho":
                out("    (d) sección con ala: el borde incluye el ancho efectivo del ala en compresión y entra ≥ "
                    "300 mm en")
                out("        el alma → nota para el plano (VERIFICACIÓN COMPLETA, «Para el plano»)")
            else:
                out("    (d) sin ala en este extremo: no aplica")
            out(f"    Encima de {hasta:.2f} m el extremo ya no lleva borde especial: rige §18.10.6.5(b) "
                f"(3.3; su longitud,")
            out("    con el c de cada cota, la comprueba la verificación completa en las losas y el corte).")
        else:
            out(f"  extremo {ext.nombre}: no exige borde especial (c = {c:.3f} m) → no aplican "
                f"§18.10.6.4(a)–(d)")
            if rho > limite_rho:
                req = A.longitud_de_borde(c)
                ok = ext.provista >= req * (1 - 1e-9)
                anota("18.10.6.5(b) longitud", ok)
                out(f"    §18.10.6.5(b): ρ del borde = {rho:.4f} > 2.8/fy = {limite_rho:.5f} → estribos en "
                    f"la longitud de §18.10.6.4(a):")
                out(f"      máx(c − 0.1ℓw, c/2) = máx({c:.3f} − {0.1 * D.LW:.3f}, {c / 2:.3f}) = "
                    f"{req:.3f} m;  confinada {ext.provista:.3f} m  → {si_no(ok)}")
            else:
                out(f"    ρ del borde = {rho:.4f} ≤ 2.8/fy = {limite_rho:.5f}: §18.10.6.5(b) no pide "
                    f"estribos; se confina igual en {ext.provista:.3f} m (criterio del proyecto)")
            if sec.tipo == "III" and ext.nombre == "derecho":
                out("    esquina alma-ala (§18.10.6.5(b), 300 mm en el alma y en el ala): nota para el plano")
    out("  (ρ del borde: el acero de sus elementos entre el área de sus cajas, fig. R18.10.6.5a, con el "
        "armado de base)")

    # ---- 3.3 -------------------------------------------------------------------
    out.titulo("3.3 ESTRIBOS DE LOS ELEMENTOS POR ZONAS — §18.10.6.4(e)–(g), §18.10.6.5(b), §18.7.5.2–3, "
               "§25.7.2")
    out("  ESPECIAL (§18.10.6.4(e)–(g)): estribos de §18.7.5.2(a)–(d); s ≤ la menor de un tercio de la")
    out("    dimensión menor del elemento, 6·db y so (§18.7.5.3), y de la tabla 18.10.6.5(b); hx ≤ mín(350 mm,")
    out("    2b/3); ramas ≤ 2·bc salvo (i) o (ii); Ash/(s·bc) de la tabla 18.10.6.4(g).")
    out("  ORDINARIO (§18.10.6.5(b), ρ del borde > 2.8/fy): estribos de §18.7.5.2(a)–(e) (hx ≤ 350 mm) en la")
    out("    longitud de §18.10.6.4(a); s de la tabla 18.10.6.5(b): mín(6·db, 150 mm) en la región crítica y")
    out("    mín(8·db, 200 mm) en otras zonas.")
    out("  PROYECTO (ρ ≤ 2.8/fy: la norma no pide estribos): se confina igual, con los límites del ordinario.")
    out("  REPLANTEO (criterio del proyecto): s es múltiplo de 2 o 5 cm, y 6·db y 8·db se llevan al múltiplo")
    out("    de 2 o 5 cm más cercano, por debajo o por encima (p. ej. 6·db = 9.5 cm → 10 cm).")
    if sec.tipo == "II":
        out("  ALA CENTRAL (tipo II): no es borde del muro en X; se confina por criterio del proyecto con los")
        out("    límites del ordinario en «otras zonas» (como columna en Y, su §18.7.5 va con el pórtico).")
    out("  db: la barra vertical principal más pequeña del borde (tabla 18.10.6.5(b), nota [1]; en el tipo")
    out("  III, del derecho y del ala); bc, al exterior del estribo (hacia el alma el estribo llega al borde")
    out("  de la zona confinada). §18.7.5.2(a)–(c) (estribos cerrados, dobleces que abrazan barras, grapas")
    out("  alternadas) van en el plano; (d) es el apoyo lateral de §25.7.2.2 y §25.7.2.3, abajo. Ash cuenta")
    out("  cada grapa con el área del estribo: las grapas van del mismo diámetro que el estribo.")
    out("  Cada elemento lleva un estribo en el armado de base, con una separación hasta la altura de")
    out("  cambio y, si se declara «separacion_estribo_arriba», otra encima (criterio del proyecto), y un")
    out("  estribo y una separación en el reducido. La altura de cambio es la mayor de la región crítica")
    out("  de la tabla 18.10.6.5(b), máx(ℓw, Mu/4Vu), y la del borde especial de cada extremo que lo")
    out(f"  exige, al cm hacia arriba: {dis.altura_de_cambio():.2f} m. «s ≤» da la máxima de cada zona.")
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
        con_estribo = [(a, el) for a, el in elemento_por_armado.items() if el.estribo is not None]
        for a, el in (con_estribo if cajas_distintas else con_estribo[:1]):
            q = A.cuantia_de_ash(el)
            out(f"    núcleo a la cara exterior del estribo{f' ({a})' if cajas_distintas else ''}: "
                f"bc = {q['bcx'] * 100:.1f} cm en X × {q['bcy'] * 100:.1f} cm en Y;  Ag = "
                f"{q['Ag'] * 1e4:.0f} cm², Ach = {q['Ach'] * 1e4:.0f} cm², Ag/Ach = {q['Ag'] / q['Ach']:.3f}")
        out(f"    {'zona z (m)':<14}{'armado':<10}{'barras':<8}{'regla':<26}{'ρ':>8}{'hx':>7}"
            f"{'hx ≤':>7}{'s ≤':>7}{'s':>5}   resultado")
        for f in filas:
            est, el = f["est"], f["elemento"]
            if est is None:
                regla = (A.regla_de_estribos(f["exige"], f["rho"])[0] if suyos else "ala central")
                out(f"    {f['z0']:5.2f}–{f['z1']:<7.2f} {f['armado'].nombre:<10}{el.barra:<8}"
                    f"{REGLA[regla]:<26}{f['rho']:8.4f}   sin estribo declarado   NO CUMPLE")
                anota(f"estribos {regla}", False)
                if regla == "especial":
                    for art in ("18.10.6.4(e)", "18.10.6.4(f)", "18.10.6.4(g)"):
                        anota(art, False)
                continue
            out(f"    {f['z0']:5.2f}–{f['z1']:<7.2f} {f['armado'].nombre:<10}{el.barra:<8}"
                f"{REGLA[est['regla']]:<26}{f['rho']:8.4f}{est['apoyo']['hx'] * 100:7.1f}"
                f"{est['hx_lim'] * 100:7.1f}{est['s_max'] * 100:7.1f}{el.separacion_estribo * 100:5.0f}"
                f"   {si_no(f['cumple'])}")
            anota(f"estribos {est['regla']}", f["cumple"])
        out("    (hx, s en cm; ρ: " + ("la del borde, fig. R18.10.6.5a)" if suyos else
                                     "la de la caja del ala, informativa)"))
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
            if est["regla"] == "especial":
                anota("18.10.6.4(e)", ok_s and ok_d)
                anota("18.10.6.4(f)", ok_hx)
            grupos: dict = {}
            for x in est["limites"]:
                grupos.setdefault(x["art"], []).append(x["expr"])
            partes = [f"{'; '.join(v)} [{k}]" for k, v in grupos.items()]
            if len(partes) == 1:
                out(f"      s ≤ la menor de {partes[0]} → {est['s_max'] * 100:.1f} cm;  s = "
                    f"{el.separacion_estribo * 100:.0f} cm  → {si_no(ok_s)}")
            else:
                out("      s ≤ la menor de:")
                for t in partes:
                    out(f"          {t}")
                out(f"        → s ≤ {est['s_max'] * 100:.1f} cm;  s = {el.separacion_estribo * 100:.0f} cm  → "
                    f"{si_no(ok_s)}")
            out(f"      apoyo lateral (§25.7.2.3 y el tope de hx): {ap['ramas_y']} de {len(cols)} barras por cara en X y "
                f"{ap['ramas_x']} de {len(fils)} por cara en Y, en esquina o grapa")
            out(f"        → hx = {ap['hx'] * 100:.1f} cm ≤ {est['hx_lim'] * 100:.1f} cm [{est['art_hx']}]  → "
                f"{si_no(ok_hx)}")
            minimo = est["estribo_min"]
            out(f"      diámetro del estribo (§25.7.2.2): barras hasta {est['db_max'] * 1000:.1f} mm → "
                f"≥ {minimo * 1000:.1f} mm; {el.estribo} = {A.diametro(el.estribo) * 1000:.1f} mm  → "
                f"{si_no(ok_d)}")
            if est["cuantia_ash"] is not None:
                q = est["cuantia_ash"]
                out("      Ash (tabla 18.10.6.4(g)): Ash/(s·bc) ≥ máx[0.3·(Ag/Ach − 1)·f'c/fyt, "
                    "0.09·f'c/fyt], fyt = fy")
                out(f"        = máx[0.3·({q['Ag'] / q['Ach']:.3f} − 1)·{D.FC:.2f}/{D.FY:.2f}, "
                    f"0.09·{D.FC:.2f}/{D.FY:.2f}] = máx[{q['a']:.5f}, {q['b']:.5f}] = {q['exigida']:.5f}")
                for a in est["ash"]:
                    out(f"        bc {a['donde']} = {a['bc'] * 100:.1f} cm: {a['ramas']} ramas de {el.estribo} = "
                        f"{a['valor'] * 1e4:.2f} cm² ≥ {q['exigida']:.5f}·{el.separacion_estribo * 100:.0f}·"
                        f"{a['bc'] * 100:.1f} = {a['limite'] * 1e4:.2f} cm²  → {si_no(a['cumple'])}")
                    anota("18.10.6.4(g)", a["cumple"])
                largo, corto = max(q["bcx"], q["bcy"]), min(q["bcx"], q["bcy"])
                if largo <= 2 * corto + 1e-9:
                    out(f"      ramas (§18.10.6.4(f)): la más larga {largo * 100:.1f} cm ≤ 2·bc = "
                        f"{2 * corto * 100:.1f} cm  → CUMPLE")
                else:
                    exento = A.rama_larga_permitida(el, vb, dis)
                    out(f"      ramas (§18.10.6.4(f)): {largo * 100:.1f} cm > 2·bc = {2 * corto * 100:.1f} cm: "
                        + (f"vale un solo estribo porque se cumple {exento}" if exento else
                           "usar estribos superpuestos, solapados al menos mín(150 mm, 2b/3) (nota del "
                           "plano)"))

        for arm in [x for x in (base, red) if x is not None and nombre == "derecho"]:
            for y, cerca in A.esquinas_sin_barra(arm):
                out(f"    {arm.nombre}: esquina del estribo dentro del ala en y = {y * 100:+.1f} cm: el modelo "
                    f"no tiene ahí barra del ala (la más")
                out(f"    cercana a {cerca * 100:.1f} cm); en el plano va una fila del ala en esa cara "
                    f"(§18.7.5.2(b): los dobleces")
                out("    abrazan barras; con ella valen el hx y las ramas de Ash de arriba). Nota del plano.")

    # ---- 3.4 -------------------------------------------------------------------
    out.titulo("3.4 RESUMEN DE §18.10.6.4 Y §18.10.6.5")
    exigidos = [cs for cs in casos if cs["exige"]]

    def estado(art: str) -> str:
        v = resumen.get(art)
        return "—" if v is None else si_no(all(v))

    if exigidos:
        out("  §18.10.6.4, borde especial en el extremo "
            + " y ".join(f"{cs['zona']} hasta {alto[cs['zona']]:.2f} m" for cs in exigidos) + ":")
        con_losa = any(dis.demanda.z_base[s] < max(alto.values()) - 1e-9
                       for s in range(1, len(D.ALTURAS)))
        aplica_c = any(f["aplica"] for cs in exigidos for f in A.dimensiones_de_borde(cs, 0)
                       if f["art"] == "18.10.6.4(c)")
        ala_d = sec.tipo == "III" and any(cs["zona"] == "derecho" for cs in exigidos)
        filas = [("(a) longitud horizontal", "3.2", estado("18.10.6.4(a)")),
                 ("(b) ancho ≥ hu/16", "3.2", estado("18.10.6.4(b)")),
                 ("(c) ancho ≥ 300 mm", "3.2", estado("18.10.6.4(c)") if aplica_c else "no aplica"),
                 ("(d) sección con ala", "plano" if ala_d else "—",
                  "nota del plano" if ala_d else
                  "no aplica (sin ala)" if sec.tipo == "I" else
                  "no aplica (el ala está en medio del alma)" if sec.tipo == "II" else
                  "no aplica (el extremo con ala no lo exige)"),
                 ("(e) estribos de §18.7.5.2(a)–(d) y separación", "3.3", estado("18.10.6.4(e)")),
                 ("(f) hx (las ramas > 2·bc, en 3.3 y en el plano)", "3.3", estado("18.10.6.4(f)")),
                 ("(g) Ash", "3.3", estado("18.10.6.4(g)")),
                 ("(h) f'c de la losa ≥ 0.7·f'c", "plano" if con_losa else "—",
                  "nota del plano" if con_losa else "no aplica (ninguna losa dentro del borde especial)"),
                 ("(i) grapas del alma en la región crítica", "plano", "nota del plano"),
                 ("(j) estribos dentro del apoyo", "plano", "nota del plano"),
                 ("(k) malla horizontal anclada en el núcleo (ℓdh)", "4",
                  si_no(all(f["cumple"] for f in vb.detalle if f["art"] == "18.10.6.4(k)")))]
        if por_desplazamiento:
            filas.insert(0, ("§18.10.6.2(b)(ii)/(iii) esbeltez de la zona comprimida", "3.2",
                             estado("18.10.6.2(b)(ii)") if all(resumen.get("18.10.6.2(b)(ii)", [True]))
                             else estado("18.10.6.2(b)(iii)")))
        for texto, donde, res in filas:
            out(f"    {texto:<56}{donde:<8}{res}")
    else:
        out("  §18.10.6.4: no aplica (ningún extremo exige borde especial).")
    vu = max(abs(e.V) for e in dis.estados(0.0, 0))
    factor = A.amplificacion()["factor"]
    lim_a = 0.083 * A.LAMBDA * A.RAIZ_FC * mef.MPA * sec.acv
    out("  §18.10.6.5, donde no hay borde especial"
        + (f" (encima de {max(alto.values()):.2f} m" + ("" if len(exigidos) == len(casos) else
                                                         " y en el otro extremo") + "):" if exigidos else
           " (los dos extremos):"))
    out(f"    {'(a) malla horizontal con gancho o horquillas en U':<56}{'plano':<8}"
        + ("nota del plano" if factor * vu >= lim_a else
           f"no aplica (ωv·Ωv·Vu = {factor * vu:.0f} < {lim_a:.0f} kN)"))
    ordinarios = resumen.get("estribos ordinario")
    out(f"    {'(b) estribos con ρ > 2.8/fy (tabla 18.10.6.5(b))':<56}{'3.3':<8}"
        + (si_no(all(ordinarios)) if ordinarios else "no aplica (ρ ≤ 2.8/fy)"))
    out(f"    {'(b) longitud con estribos ≥ máx(c − 0.1ℓw, c/2)':<56}{'3.2':<8}"
        + (estado("18.10.6.5(b) longitud") if "18.10.6.5(b) longitud" in resumen else
           "se comprueba en las losas y el corte" if exigidos else "no aplica (ρ ≤ 2.8/fy)"))
    out(f"    {'(b) esquina alma-ala (300 mm en el alma y en el ala)':<56}"
        f"{('plano' if sec.ala is not None else '—'):<8}"
        + ("nota del plano" if sec.ala is not None else "no aplica (sin ala)"))
    if resumen.get("estribos proyecto"):
        out(f"    {'con ρ ≤ 2.8/fy (confinados por el proyecto)':<56}{'3.3':<8}"
            f"{si_no(all(resumen['estribos proyecto']))}")
    if resumen.get("estribos ala central"):
        out(f"    {'ala central (columna en Y; confinada por el proyecto)':<56}{'3.3':<8}"
            f"{si_no(all(resumen['estribos ala central']))}")
    sig = [A.esfuerzo_de_fibra_extrema(e) for e in dis.estados(0.0, 0)]
    out(f"  Informativo, vía de esfuerzo en el plano (§18.10.6.3, no se usa): compresión elástica "
        f"máxima en la base {max(sig) / 1000:.2f} MPa = {max(sig) / (D.FC * mef.MPA):.2f}·f'c")


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
    queda sobre el techo, el último, solo para informar)."""
    zb, zt = dis.demanda.z_base, dis.demanda.z_tope
    pisos = range(len(D.ALTURAS))
    if arm.nombre == "base":
        return [s for s in pisos if zb[s] < corte - A.TOL]
    return [s for s in pisos if zt[s] > corte + A.TOL] or [len(D.ALTURAS) - 1]


def refuerzo_de(dis, arm, pisos: list[int]) -> dict:
    """¿Pide el cortante refuerzo? (§11.6, §11.7), con el entrepiso de mayor
    |Ve| de los que lleva el armado (la malla es la misma en todos ellos)."""
    return max((A.refuerzo_de_corte_requerido(arm, dis.estados(dis.demanda.z_base[s], s),
                                              dis.phi_corte["phi"]) for s in pisos),
               key=lambda r: r["Ve"])


def tope_de_separacion(refuerzo: dict, divisor: int, art: str) -> tuple[float, str]:
    """La separación máxima de la malla: 450 mm (§18.10.2.1), 3h (§11.7.2.1 o
    §11.7.3.1) y, si el cortante pide refuerzo, ℓw/divisor; con el texto."""
    partes = [(0.45, "45 cm [18.10.2.1]"), (3 * D.ESPESOR, f"3h = {3 * D.ESPESOR * 100:.0f} cm [{art}]")]
    if refuerzo["requerido"]:
        partes.append((D.LW / divisor, f"ℓw/{divisor} = {D.LW / divisor * 100:.1f} cm [{art}, el "
                                       f"cortante pide refuerzo]"))
    return min(v for v, _ in partes), "mín(" + ", ".join(t for _, t in partes) + ")"


def texto_de_refuerzo(refuerzo: dict) -> str:
    return (f"|Ve| = {refuerzo['Ve']:.1f} kN {'>' if refuerzo['requerido'] else '≤'} φ·Vc = "
            f"{refuerzo['phi']:.2f}·{refuerzo['Vc']:.1f} = {refuerzo['phiVc']:.1f} kN → el cortante "
            + ("PIDE refuerzo (Vs > 0)" if refuerzo["requerido"] else "no pide refuerzo"))


def cuantia_longitudinal(out, dis, base, red, refuerzos: dict) -> None:
    """1.2: cuantía mínima y separación del refuerzo vertical del alma, de
    cada armado con su malla horizontal y el cortante de sus entrepisos."""
    out.titulo("1.2 CUANTÍA MÍNIMA DEL REFUERZO VERTICAL (LONGITUDINAL) — §18.10.2.1, §18.10.4.3, "
               "§11.6.2, §11.7.2")
    armados = [x for x in (base, red) if x is not None]
    if base.malla_vertical.automatica:
        out(f"  Separación calculada: la mayor múltiplo de 2 o 5 cm (≤ 45 cm) con ρℓ ≥ "
            f"{'máx(0.0025, ρt)' if A.ESBELTEZ <= 2.0 else '0.0025'}")
    out("  §18.10.2.1: ρℓ ≥ 0.0025 (la vertical no se reduce nunca; la reducción con Vu bajo es solo")
    out("  de ρt). ρℓ = cortinas·Ab/(s·b), de la malla vertical:")
    for arm in armados:
        mv = arm.malla_vertical
        out(f"    {arm.nombre:<9} {mv.texto():<16} ρℓ = {mv.cortinas}·{A.area(mv.barra) * 1e4:.2f}/"
            f"({mv.separacion * 100:.0f}·{D.ESPESOR * 100:.0f}) = {mv.cuantia:.5f}  ≥ 0.0025  → "
            f"{si_no(mv.cuantia >= 0.0025 * (1 - 1e-9))}")
    if A.ESBELTEZ <= 2.0:
        out(f"  §18.10.4.3: muro robusto (hw/ℓw = {A.ESBELTEZ:.2f} ≤ 2), ρℓ ≥ ρt (la malla horizontal "
            f"del mismo armado):")
        for arm in armados:
            mv, mh = arm.malla_vertical, arm.malla_horizontal
            out(f"    {arm.nombre:<9} ρℓ = {mv.cuantia:.5f} frente a ρt = {mh.cuantia:.5f}  → "
                f"{si_no(mv.cuantia >= mh.cuantia * (1 - 1e-9))}")
    else:
        out(f"  §18.10.4.3 (ρℓ ≥ ρt): solo si hw/ℓw ≤ 2; aquí hw/ℓw = {A.ESBELTEZ:.2f} → no aplica")
    out("  §11.6.2 y §11.7.2.1, con el mayor |Ve| de los entrepisos de cada armado:")
    for arm in armados:
        refuerzo = refuerzos[arm.nombre]
        mh = arm.malla_horizontal
        umbral = refuerzo["umbral_116"]
        if refuerzo["sobre_116"]:
            out(f"    {arm.nombre}: §11.6.2(a): |Ve| = {refuerzo['Ve']:.1f} kN > 0.5·φ·αc·λ·√f'c·Acv = "
                f"0.5·φ·Vc = {umbral:.1f} kN, así que")
            out(f"      ρℓ ≥ 0.0025 + 0.5·(2.5 − hw/ℓw)·(ρt − 0.0025) = 0.0025 + 0.5·(2.5 − "
                f"{A.ESBELTEZ:.2f})·({mh.cuantia:.5f} − 0.0025) = {refuerzo['ecuacion_1162']:.5f},")
            out(f"      no menos de 0.0025 (ni más que el ρt que pide la resistencia, "
                f"{refuerzo['rho_t_resistencia']:.5f}) → {refuerzo['rho_l_1162']:.5f}"
                + ("; con hw/ℓw ≥ 2.5 es la de §18.10.2.1" if A.ESBELTEZ >= 2.5 else ""))
            if A.ESBELTEZ < 2.5:
                mv = arm.malla_vertical
                out(f"      ρℓ = {mv.cuantia:.5f}  → "
                    f"{si_no(mv.cuantia >= refuerzo['rho_l_1162'] * (1 - 1e-9))}")
        else:
            out(f"    {arm.nombre}: §11.6.1: |Ve| = {refuerzo['Ve']:.1f} kN ≤ 0.5·φ·Vc = {umbral:.1f} kN → "
                f"tabla 11.6.1 (ρℓ ≥ 0.0012 o 0.0015), menos que §18.10.2.1")
    out("  Separación de las barras verticales:")
    for arm in armados:
        refuerzo = refuerzos[arm.nombre]
        tope, texto = tope_de_separacion(refuerzo, 3, "11.7.2.1")
        mv = arm.malla_vertical
        real = A.separacion_real_vertical(dis.barras(arm))
        out(f"    {arm.nombre}: ≤ {texto} = {tope * 100:.1f} cm")
        out(f"      ({texto_de_refuerzo(refuerzo)})")
        out(f"      malla s = {mv.separacion * 100:.0f} cm; la mayor separación real entre "
            f"líneas de barras verticales del alma = {real * 100:.1f} cm  → "
            f"{si_no(max(mv.separacion, real) <= tope * (1 + 1e-9))}")
    tramos = A.tramos_libres(base)
    out("  La malla va solo en los tramos libres entre elementos, sin pisar sus barras: "
        + ", ".join(f"{a * 100:.1f}–{b * 100:.1f} cm" for a, b in tramos)
        + f"; líneas a {base.malla_vertical.separacion * 100:.0f} cm, centradas en cada tramo (un tramo "
        f"cuyo hueco no pasa de la separación no lleva líneas).")
    out("  §11.7.5.1 (sin estribos que apoyen la malla): ρℓ ≤ 0.01 → "
        + ", ".join(f"{arm.nombre} {arm.malla_vertical.cuantia:.5f}" for arm in armados)
        + f"  → {si_no(all(arm.malla_vertical.cuantia <= 0.01 for arm in armados))}")


def cortante_por_piso(out, dis, base, red, corte: float, factor: float) -> list[dict]:
    """2.3: el cortante amplificado en cada entrepiso (V es constante entre
    losas), por combinación, con φVn y el D/C. φVn con la malla horizontal del
    armado del entrepiso; si el corte cae dentro de él, con la más débil de
    las dos (las dos llevan el mismo V). Devuelve las filas (para la figura)."""
    out.titulo("2.3 CORTANTE AMPLIFICADO POR ENTREPISO — §18.10.3.3.2 y §18.10.4")
    out(f"  Ve = Vg + Ωv·ωv·VEh, Ωv·ωv = {factor:.3f}; V no cambia dentro del entrepiso (entre losas no")
    out("  entra fuerza horizontal). |Vu| y |Ve|: el mayor de los dos signos del sismo, |Vg| + |VEh| y")
    out(f"  |Vg| + {factor:.3f}·|VEh|. La gravitacional no se amplifica (Ve = Vu). φVn, con la malla")
    out("  horizontal del armado de cada entrepiso (donde está el corte, la más débil de las dos).")
    out(f"    {'entrepiso':<10}{'armado':<10}{'combinación':<15}{'Vg':>8}{'|VEh|':>8}{'|Vu|':>8}{'|Ve|':>8}"
        f"{'Ve/Vu':>7}{'φVn':>8}{'D/C':>7}")
    filas = []
    phi = dis.phi_corte["phi"]
    for s in range(len(D.ALTURAS)):
        z = dis.demanda.z_base[s]
        estados = dis.estados(z, s)
        presentes = [a for a in (base, red) if a is not None and s in pisos_del_armado(dis, a, corte)
                     and (a is base or corte < A.H - A.TOL)]
        arm = min(presentes, key=lambda a: A.corte_en_el_plano(a, estados[0], phi)["phiVn"])
        r = A.corte_en_el_plano(arm, estados[0], phi)
        etiqueta = "base/red." if len(presentes) > 1 else arm.nombre
        vistas = {}
        for e in estados:
            vistas.setdefault(e.combo, e)
        peor = None
        for combo, e in vistas.items():
            if e.con_sismo:
                g, h = e.Vg, abs(e.VEh)
                vu = max(abs(g + h), abs(g - h))
                ve = max(abs(g + factor * h), abs(g - factor * h))
            else:
                g, h, vu, ve = e.V, 0.0, abs(e.V), abs(e.Ve)
            dc = ve / r["phiVn"]
            fila = {"piso": s, "z0": z, "z1": dis.demanda.z_tope[s], "combo": combo, "Vg": g,
                    "VEh": h, "Vu": vu, "Ve": ve, "phiVn": r["phiVn"], "dc": dc, "armado": etiqueta}
            filas.append(fila)
            if peor is None or ve > peor["Ve"]:
                peor = fila
            out(f"    {D.PISOS[s]:<10}{etiqueta:<10}{combo:<15}{g:8.1f}{h:8.1f}{vu:8.1f}{ve:8.1f}"
                f"{(ve / vu if vu else 1.0):7.2f}{r['phiVn']:8.1f}{dc:7.3f}")
        out(f"    {'':<10}→ gobierna {peor['combo']}: |Ve| = {peor['Ve']:.1f} kN ≤ φVn = "
            f"{peor['phiVn']:.1f} kN  → {si_no(peor['dc'] <= 1.0)}")
    out(f"  (kN. Figura: cortante_en_altura_{D.MURO}.png)")
    return filas


def cuantia_transversal(out, dis, base, red, corte: float, refuerzos: dict) -> None:
    """2.4: cuantía mínima y separación del refuerzo horizontal (transversal)
    de cada armado, con el cortante de los entrepisos donde va."""
    sec = A.seccion()
    out.titulo("2.4 CUANTÍA MÍNIMA DEL REFUERZO HORIZONTAL (TRANSVERSAL) — §18.10.2.1, tabla 11.6.1, "
               "§11.6.2(b), §11.7.3")
    umbral = 0.083 * A.LAMBDA * A.RAIZ_FC * mef.MPA * sec.acv
    out("  Cada malla horizontal se comprueba con el mayor cortante de los entrepisos donde va (en cada")
    out("  losa y en el corte lo repite la verificación completa).")
    out("  §18.10.2.1: ρt ≥ 0.0025; si Vu ≤ 0.083·λ·√f'c·Acv puede bajar a la tabla 11.6.1 (Vu, el")
    out(f"  mayorado sin amplificar, como en §18.10.2.2); 0.083·1.0·√{A.FC_RAIZ:.2f}·{sec.acv:.3f}·1000 = "
        f"{umbral:.1f} kN.")
    for arm in [x for x in (base, red) if x is not None]:
        refuerzo = refuerzos[arm.nombre]
        mh = arm.malla_horizontal
        pisos = pisos_del_armado(dis, arm, corte)
        vu = max(abs(e.V) for s in pisos for e in dis.estados(dis.demanda.z_base[s], s))
        donde = ", ".join(D.PISOS[s] for s in pisos)
        out(f"  {arm.nombre.upper()} ({mh.texto()}; entrepisos {donde}"
            + (", no se usa: el corte queda sobre el techo" if arm is red and corte >= A.H - A.TOL
               else "") + "):")
        out(f"    Vu = {vu:.1f} kN  frente a  {umbral:.1f} kN")
        if vu <= umbral:
            grande = D.BARRAS[mh.barra][0] > 16.0
            minimo = 0.0025 if (grande or D.FY < 420.0) else 0.0020
            out(f"    → puede bajar a la tabla 11.6.1: barras {'> No. 16' if grande else '≤ No. 16'}, "
                f"fy = {D.FY:.2f} MPa {'< 420' if D.FY < 420.0 else '≥ 420'} → ρt ≥ {minimo}")
            if refuerzo["sobre_116"]:
                minimo = 0.0025
                out(f"    pero §11.6.2(b): |Ve| = {refuerzo['Ve']:.1f} kN > 0.5·φ·Vc = "
                    f"{refuerzo['umbral_116']:.1f} kN → ρt ≥ 0.0025")
        else:
            minimo = 0.0025
            out("    → Vu pasa del umbral: ρt ≥ 0.0025"
                + (" (lo mismo pide §11.6.2(b), con |Ve| > 0.5·φ·Vc)" if refuerzo["sobre_116"] else ""))
        out(f"    ρt = cortinas·Ab/(s·b) = {mh.cortinas}·{A.area(mh.barra) * 1e4:.2f}/"
            f"({mh.separacion * 100:.0f}·{D.ESPESOR * 100:.0f}) = {mh.cuantia:.5f}  ≥ {minimo}  → "
            f"{si_no(mh.cuantia >= minimo * (1 - 1e-9))}")
        tope, texto = tope_de_separacion(refuerzo, 5, "11.7.3.1")
        out(f"    separación ≤ {texto} = {tope * 100:.1f} cm")
        out(f"      ({texto_de_refuerzo(refuerzo)})")
        out(f"    s = {mh.separacion * 100:.0f} cm  → {si_no(mh.separacion <= tope * (1 + 1e-9))}")
    out("  §18.10.4.3: refuerzo de corte repartido en las dos direcciones del plano del muro (malla")
    out("  vertical y horizontal); el ρt que pide la resistencia lo da el D/C de 2.2 y 2.3.")


def nombre_por_P(e) -> str:
    """El estado nombrado por su axial (c solo depende de P)."""
    if e.paso in ("Max", "Min", ""):
        return e.nombre
    return f"{e.combo} (P de la fila {'Max' if e.paso[0] == '+' else 'Min'})"


def tabla_de_amplificacion(out, estados, factor: float, corte) -> None:
    """Por combinación con sismo: de dónde salen Vg y VEh (SEPARACION_DEL_CORTE),
    sus valores y los dos cortantes de diseño. Las que no llevan sismo no se
    amplifican."""
    out(f"    {'combinación':<15}{'gravedad (ETABS)':<18}{'sismo (ETABS)':<25}{'Vg':>8}{'|VEh|':>8}"
        f"{'+':>9}{'−':>9}")
    vistas = set()
    for e in estados:
        if e.combo in vistas or not e.con_sismo:
            continue
        vistas.add(e.combo)
        grav, sis = corte.partes[e.combo]
        g, h = e.Vg, abs(e.VEh)
        out(f"    {e.combo:<15}{grav:<18}{sis:<25}{g:8.1f}{h:8.1f}{g + factor * h:9.1f}"
            f"{g - factor * h:9.1f}")
    out("    (+ y −: Vg ± Ωv·ωv·|VEh|, los dos cortantes de diseño de cada combinación)")
    out("  Comprobación: las filas Max y Min de cada combinación de ETABS son gravedad + sismo y")
    out("  gravedad − sismo, en todos los pisos, base y tope (mayor diferencia, kN y kN·m):")
    for combo, (en_v, en_todo) in corte.comprobacion.items():
        out(f"    {combo:<15}en Vx {en_v:.3f};  en P, V y M en X e Y {en_todo:.3f}  → "
            f"{si_no(en_v <= mef.TOLERANCIA_DEL_CORTE)}")
    out("    (ETABS redondea a 0.01: Vu = Vg + VEh)")


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
            est = dis.estados(z, s)
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
    out("  Ejes del programa: x de 0 (extremo izquierdo) a ℓw (extremo derecho); y a través del")
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
    out("     llega al borde de la zona confinada, la «longitud» declarada: ≥ 0.15·ℓw en la base;")
    out("     en el reducido, ≤ la de la base y sin ese mínimo)")
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

    def phi_de(p):
        phi = A.phi_por_deformacion(p.eps_t)
        p_umbral = -0.1 * D.FC * mef.MPA * sec.area
        p_bal = diag.p_bal[sentido]
        if p_bal < p_umbral and p_bal <= p.P <= p_umbral:
            avance = (p.P - p_umbral) / (p_bal - p_umbral)
            phi = min(phi, 0.9 + avance * (0.65 - 0.9))
        return phi

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
    out(f"    εt = {p.eps_t:.5f} (barra más traccionada) → φ = {phi:.3f} (tabla 21.2.2 y §21.2.2.3)")
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
