"""DISEÑO DEL MURO CON ACI 318-25 (capítulo 18, muros estructurales especiales).

LA SECCIÓN (TIPO en el archivo de datos)
  TIPO I    rectángulo ℓw × b, con un elemento confinado en cada extremo.
  TIPO II   el rectángulo con un ala corta en medio: b' de ancho y l'w de largo,
            medido desde la cara exterior del alma (incluye toda la intersección).
  TIPO III  el rectángulo con un ala corta en el extremo derecho.
  Los tipos II y III son muros rectangulares especiales: sus alas cortas son,
  en la otra dirección (Y), columnas de los pórticos especiales. La sección se
  diseña como un muro (el alma) con alas confinadas, no como un muro en L o en
  T: el muro trabaja como muro solo en X. Las alas sí entran en la capacidad a
  flexocompresión (diagrama P-M y biaxial con la sección completa).

LOS ELEMENTOS CONFINADOS
  izquierdo  [0, L] × b, con L ≥ 0.15·ℓw en la base (criterio del proyecto;
             en el reducido, L ≤ la de la base, que quepan las barras y
             §18.10.6.5(b); sus barras continúan las de la base).
  derecho    [ℓw − L, ℓw] × b. En el tipo III, L ≥ b' (si 0.15·ℓw < b', se
             confina todo el ancho del ala).
  ala        el ala entera, b' × l'w, con la intersección (tipos II y III).
  Las barras de cada uno van en el perímetro de su estribo: nx en cada cara
  paralela a X y ny en cada cara paralela a Y, con el eje a
      r = recubrimiento + diámetro del estribo + diámetro de la barra / 2
  de cada lado de la caja. En el tipo III, donde el elemento derecho se monta
  sobre el ala, mandan las barras del ala.
  El acero del alma (malla) va solo en los tramos libres entre elementos, sin
  pisar sus barras, con la separación que da la cuantía mínima (o la que se
  declare); si el hueco de un tramo no pasa de esa separación, no lleva
  líneas. En el tipo III, si todas las columnas del derecho caen dentro del ala, el ala lo
  absorbe (su «longitud» no puede pasar de b').

LOS MÓDULOS
  1. FLEXOCOMPRESIÓN  — flexión BIAXIAL en cada nivel (§22.2): su D/C es el
                        que se verifica; el diagrama P-M en el plano (φ de la
                        tabla 21.2.2 con el tope de §21.2.2.3) queda de
                        referencia. Cuantía mínima y separación del acero
                        vertical (§18.10.2.1, §18.10.4.3, §11.6.2, §11.7.2.1),
                        cortinas (§18.10.2.2), acero de extremo (§18.10.2.4) y
                        la COTA DE CORTE del armado reducido (encima de
                        §18.10.2.3(a), §18.10.2.4(b) y §18.10.6.2(b)(i)).
  2. CORTE            — en el plano del muro: amplificación de la parte de
                        sismo (§18.10.3.3.2), por entrepiso, resistencia
                        (§18.10.4) y cuantía mínima y separación del acero
                        horizontal (§18.10.2.1, tabla 11.6.1, §11.6.2,
                        §11.7.3.1). El corte en Y no se verifica: en esa
                        dirección el muro no trabaja como muro.
  3. ELEMENTO DE BORDE — necesidad (§18.10.6.2, o §18.10.6.3 en muros bajos) en
                        la dirección larga (X), aparte del confinamiento de
                        flexocompresión; dimensiones (§18.10.6.4(a)–(c),
                        §18.10.6.2(b)) y estribos por zonas en altura, con
                        borde especial (§18.10.6.4(e)–(g)) u ordinario
                        (§18.10.6.5(b)). §18.10.6.4(d) y la regla de la
                        esquina alma-ala de §18.10.6.5(b), (h)–(k) y
                        §18.10.6.5(a) van como notas del plano.

CONVENIOS
  - Unidades: kN y m. Los esfuerzos se pasan de MPa a kN/m² multiplicando por
    MPA = 1000. Las fórmulas con √f'c usan f'c en MPa, como en la norma.
  - P negativo en compresión.
  - x va de 0 (extremo IZQUIERDO) a ℓw (extremo DERECHO); y atraviesa el
    espesor (el alma va de −b/2 a +b/2) y apunta hacia el ala. Los momentos se
    toman respecto del centroide de la sección real.
  - M negativo comprime la cara +x (+y). En el plano, sentido +1 comprime el
    extremo DERECHO (M < 0) y sentido −1 el IZQUIERDO (M > 0).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache

import numpy as np

import datos as D
from mef import MPA, DatoInvalido, Demanda, Estado

# =============================================================================
# 0. CONSTANTES DE LA NORMA
# =============================================================================

EPS_CU = 0.003                      # §22.2.2.1
EPS_TY = D.FY / D.ES                # §21.2.2.1: εty = fy/Es (0.00206 con 411.88 MPa)
PHI_COMPRESION = 0.65               # tabla 21.2.2, controlada por compresión
PHI_TRACCION = 0.90                 # tabla 21.2.2, controlada por tracción
PHI_CORTE = 0.75                    # tabla 21.2.1(b)
PHI_CORTE_REDUCIDO = 0.60           # §21.2.4.1: si Vn < cortante de Mn
LAMBDA = 1.0                        # concreto de peso normal
FC_RAIZ = min(D.FC, 85.0)           # §18.10.4.1: f'c ≤ 85 MPa dentro de la raíz
RAIZ_FC = math.sqrt(FC_RAIZ)        # √f'c en MPa
H = sum(D.ALTURAS)                  # altura del muro, m
ESBELTEZ = H / D.LW if D.LW > 0 else math.inf     # hw/ℓw del muro entero
PROLONGACION = 3.60                 # §18.10.2.3(a): 3.6 m sobre donde deja de hacer falta
TOL = 1e-9
TIPOS = ("I", "II", "III")
FC_MINIMO_ESPECIAL = 21.0           # tabla 19.2.1.1: muros especiales con grado 420, MPa
DIAMETRO_No32 = 0.0323              # §25.7.2.2: No. 32 (32.3 mm) y menores → estribo No. 10
ESTRIBO_No10 = 0.0095               # No. 10: 9.5 mm (3/8")
ESTRIBO_No13 = 0.0127               # No. 13: 12.7 mm
# Replanteo (criterio del proyecto, no de la norma): las separaciones de los
# estribos y de las dos mallas, en centímetros enteros múltiplos de 2 o de 5
# (8, 10, 12, 14, 15, 16, 18, 20…)
REPLANTEO = "múltiplo de 2 o 5 cm"


def _de_replanteo(cm: int) -> bool:
    return cm > 0 and (cm % 2 == 0 or cm % 5 == 0)


def es_de_replanteo(s: float) -> bool:
    """Si la separación s (m) es un centímetro entero múltiplo de 2 o de 5."""
    cm = round(s * 100)
    return abs(s * 100 - cm) <= 1e-6 and _de_replanteo(cm)


def replanteo_hasta(s: float) -> float:
    """La mayor separación de replanteo que no pasa de s (m); 0 si no hay."""
    cm = math.floor(s * 100 + 1e-6)
    while cm > 0 and not _de_replanteo(cm):
        cm -= 1
    return max(cm, 0) / 100


def texto_limite(limite) -> str:
    """El límite de una fila del detallado: una cifra o un texto («2 o 5 cm»)."""
    return limite if isinstance(limite, str) else f"{limite:.4g}"


def replanteo_cercano(s: float) -> float:
    """La separación de replanteo más cercana a s (m), por debajo o por
    encima; a igual distancia, la de abajo."""
    abajo = replanteo_hasta(s)
    cm = max(math.ceil(s * 100 - 1e-6), 1)
    while not _de_replanteo(cm):
        cm += 1
    arriba = cm / 100
    return arriba if abajo <= 0 or arriba - s < s - abajo - 1e-9 else abajo


def beta1(fc: float) -> float:
    """Tabla 22.2.2.4.3: profundidad del bloque, a = β1·c."""
    if fc <= 28.0:
        return 0.85
    if fc >= 55.0:
        return 0.65
    return 0.85 - 0.05 * (fc - 28.0) / 7.0


BETA1 = beta1(D.FC)


def phi_por_deformacion(eps_t: float) -> float:
    """Tabla 21.2.2: 0.65 hasta εty, 0.90 desde εty + 0.003, lineal en medio."""
    if eps_t <= EPS_TY:
        return PHI_COMPRESION
    if eps_t >= EPS_TY + 0.003:
        return PHI_TRACCION
    return PHI_COMPRESION + (eps_t - EPS_TY) / 0.003 * (PHI_TRACCION - PHI_COMPRESION)


def diametro(barra: str) -> float:
    return D.BARRAS[barra][0] / 1000.0          # m


def area(barra: str) -> float:
    return D.BARRAS[barra][1] / 1e4             # m²


def masa(barra: str) -> float:
    return D.BARRAS[barra][2]                   # kg/m


# =============================================================================
# 1. LA SECCIÓN
# =============================================================================


@dataclass(frozen=True)
class Rect:
    """Rectángulo [x0, x1] × [y0, y1] en planta."""

    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def ancho(self) -> float:          # a lo largo de X
        return self.x1 - self.x0

    @property
    def alto(self) -> float:           # a lo largo de Y
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.ancho * self.alto

    @property
    def centro(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @cached_property
    def esquinas(self) -> np.ndarray:
        """En sentido antihorario (se calcula una vez)."""
        return np.array([[self.x0, self.y0], [self.x1, self.y0],
                         [self.x1, self.y1], [self.x0, self.y1]])

    @cached_property
    def vertices(self) -> tuple:
        """Las mismas esquinas como parejas (x, y) de números: `recortar` va más rápido."""
        return tuple((float(x), float(y)) for x, y in self.esquinas)

    def interseccion(self, otro: "Rect") -> float:
        dx = min(self.x1, otro.x1) - max(self.x0, otro.x0)
        dy = min(self.y1, otro.y1) - max(self.y0, otro.y0)
        return max(dx, 0.0) * max(dy, 0.0)


@dataclass(frozen=True)
class Seccion:
    tipo: str
    lw: float                  # ℓw, longitud total en X
    b: float                   # espesor del alma
    piezas: tuple              # rectángulos que NO se solapan: el alma y lo que el ala sale de él
    ala: Rect | None           # el ala entera, con la intersección (la caja de su elemento)
    ancho_ala: float           # b'
    largo_ala: float           # l'w, desde la cara exterior del alma
    area: float                # Ag de la sección real
    xc: float                  # centroide de la sección real
    yc: float
    Ix: float                  # ∫(x − xc)² dA: flexión en el plano (alrededor de un eje paralelo a Y)
    Iy: float                  # ∫(y − yc)² dA: flexión en Y
    xe: float                  # centro de gravedad del modelo de ETABS (muros por su eje)
    ye: float
    muros_etabs: tuple         # (largo, espesor, xc, yc) de cada muro del modelo de ETABS

    @cached_property
    def esquinas(self) -> np.ndarray:
        return np.vstack([p.esquinas for p in self.piezas])

    @property
    def acv(self) -> float:
        """§18.10.4.1: área del alma en la dirección del cortante, ℓw·b."""
        return self.lw * self.b

    def canto(self, n: np.ndarray) -> float:
        """Extensión de la sección en la dirección n."""
        proy = self.esquinas @ n
        return float(proy.max() - proy.min())

    def anchos_a(self, n: np.ndarray, profs: np.ndarray) -> np.ndarray:
        """Ancho de la sección (largo de la cuerda) a cada profundidad de
        `profs` desde la fibra extrema en la dirección n (solo n = ±X o ±Y)."""
        niveles = float((self.esquinas @ n).max()) - np.asarray(profs, dtype=float)
        total = np.zeros_like(niveles)
        for p in self.piezas:
            proy = p.esquinas @ n
            dentro = (proy.min() - 1e-12 <= niveles) & (niveles <= proy.max() + 1e-12)
            total += np.where(dentro, p.alto if abs(n[0]) > 0.5 else p.ancho, 0.0)
        return total


def _numero(nombre: str, minimo: float = 0.0) -> float:
    valor = getattr(D, nombre, None)
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not valor > minimo:
        raise DatoInvalido(f"{nombre} tiene que ser un número mayor que {minimo:g}; es {valor!r}")
    return float(valor)


def tipo_de_seccion() -> str:
    """El TIPO del muro: el programa admite SOLO los tipos I, II y III."""
    tipo = str(getattr(D, "TIPO", "")).strip().upper()
    if tipo not in TIPOS:
        raise DatoInvalido(f"TIPO es \"I\", \"II\" o \"III\" (el programa admite solo esos tres tipos "
                           f"de muro), no {getattr(D, 'TIPO', None)!r}")
    return tipo


@lru_cache(maxsize=None)
def seccion() -> Seccion:
    """La sección real y la del modelo de ETABS, a partir del archivo de datos."""
    tipo = tipo_de_seccion()
    lw, b = _numero("LW"), _numero("ESPESOR")
    alma = Rect(0.0, lw, -b / 2, b / 2)
    piezas: list[Rect] = [alma]
    ala = None
    b_ala = l_ala = 0.0
    muros = [(lw, b, lw / 2, 0.0)]
    if tipo != "I":
        b_ala, l_ala = _numero("ALA_ANCHO"), _numero("ALA_LARGO")
        if l_ala <= b + 1e-9:
            raise DatoInvalido(f"ALA_LARGO (l'w) se mide desde la cara exterior del alma e incluye "
                               f"su espesor: tiene que pasar de ESPESOR = {b:g} m; es {l_ala:g} m")
        if b_ala >= lw - 1e-9:
            raise DatoInvalido(f"ALA_ANCHO (b') = {b_ala:g} m tiene que ser menor que LW = {lw:g} m")
        tope = -b / 2 + l_ala
        if tipo == "II":
            xa = _numero("ALA_X")
            x0, x1 = xa - b_ala / 2, xa + b_ala / 2
            if x0 <= 1e-9 or x1 >= lw - 1e-9:
                raise DatoInvalido(f"el ala del tipo II (ALA_X = {xa:g} m, b' = {b_ala:g} m) tiene que "
                                   f"quedar dentro del alma, entre 0 y {lw:g} m")
            # ETABS: el alma entera por su eje y el ala desde el eje del alma
            muros.append((l_ala - b / 2, b_ala, xa, (l_ala - b / 2) / 2))
        else:
            x0, x1 = lw - b_ala, lw
            # ETABS: el alma llega al eje del ala; el ala sale del eje del alma
            muros = [(lw - b_ala / 2, b, (lw - b_ala / 2) / 2, 0.0),
                     (l_ala - b / 2, b_ala, lw - b_ala / 2, (l_ala - b / 2) / 2)]
        ala = Rect(x0, x1, -b / 2, tope)
        piezas.append(Rect(x0, x1, b / 2, tope))
    A = sum(p.area for p in piezas)
    xc = sum(p.area * p.centro[0] for p in piezas) / A
    yc = sum(p.area * p.centro[1] for p in piezas) / A
    Ix = sum(p.alto * p.ancho**3 / 12 + p.area * (p.centro[0] - xc) ** 2 for p in piezas)
    Iy = sum(p.ancho * p.alto**3 / 12 + p.area * (p.centro[1] - yc) ** 2 for p in piezas)
    Ae = sum(L * t for L, t, _, _ in muros)
    xe = sum(L * t * x for L, t, x, _ in muros) / Ae
    ye = sum(L * t * y for L, t, _, y in muros) / Ae
    return Seccion(tipo, lw, b, tuple(piezas), ala, b_ala, l_ala, A, xc, yc, Ix, Iy, xe, ye,
                   tuple(muros))


def area_bruta() -> float:
    return seccion().area


# =============================================================================
# 2. EL ARMADO: ELEMENTOS CONFINADOS Y MALLA DEL ALMA
# =============================================================================


@dataclass(frozen=True)
class Elemento:
    """Un elemento confinado: su caja, sus barras en el perímetro del estribo y
    el estribo."""

    nombre: str                       # "izquierdo", "derecho" o "ala"
    caja: Rect
    barra: str
    nx: int                           # barras en cada cara paralela a X
    ny: int                           # barras en cada cara paralela a Y
    estribo: str | None
    separacion_estribo: float | None
    grapas: str = "alternas"          # "alternas" (mínimo de §25.7.2.3) o "todas"
    hacia_el_alma: str | None = None  # lado de la caja que da al alma: "x0", "x1" o None
    # solo en el armado de base: la separación encima de la altura de cambio
    # (`altura_de_cambio`); None = la misma de abajo
    separacion_arriba: float | None = None

    def recubrimiento(self, lado: str) -> float:
        """Recubrimiento libre en un lado de la caja ("x0", "x1", "y0", "y1"):
        el de los datos en las caras del muro; 0 en el lado que da al alma,
        porque allí no hay cara: el estribo llega al borde de la zona confinada
        (0.15·ℓw o la longitud declarada)."""
        return 0.0 if lado == self.hacia_el_alma else D.RECUBRIMIENTO

    def a_eje(self, lado: str) -> float:
        """Del borde de la caja al eje de las barras, en ese lado."""
        est = diametro(self.estribo) if self.estribo else 0.0
        return self.recubrimiento(lado) + est + diametro(self.barra) / 2

    @property
    def r(self) -> float:
        """Del borde de la caja al eje de las barras en las caras del muro."""
        return self.a_eje("y0")

    def columnas(self) -> np.ndarray:
        return np.linspace(self.caja.x0 + self.a_eje("x0"), self.caja.x1 - self.a_eje("x1"), self.nx)

    def filas(self) -> np.ndarray:
        return np.linspace(self.caja.y0 + self.a_eje("y0"), self.caja.y1 - self.a_eje("y1"), self.ny)

    def contorno_del_estribo(self) -> tuple[float, float, float, float]:
        """x0, x1, y0, y1 de la CARA EXTERIOR del estribo."""
        c = self.caja
        return (c.x0 + self.recubrimiento("x0"), c.x1 - self.recubrimiento("x1"),
                c.y0 + self.recubrimiento("y0"), c.y1 - self.recubrimiento("y1"))

    def posiciones(self) -> list[tuple[float, float]]:
        """Las barras del perímetro: las dos caras paralelas a X completas y,
        en las caras paralelas a Y, las intermedias."""
        xs, ys = self.columnas(), self.filas()
        salida = [(float(x), float(ys[0])) for x in xs] + [(float(x), float(ys[-1])) for x in xs]
        for y in ys[1:-1]:
            salida += [(float(xs[0]), float(y)), (float(xs[-1]), float(y))]
        return salida

    @property
    def nucleo(self) -> tuple[float, float]:
        """bc en X y en Y: el núcleo medido a la cara exterior del estribo."""
        x0, x1, y0, y1 = self.contorno_del_estribo()
        return x1 - x0, y1 - y0

    @property
    def espesor(self) -> float:
        """El espesor del muro que confina: b en los extremos del alma, b' en el ala."""
        return self.caja.ancho if self.nombre == "ala" else self.caja.alto

    def texto(self, cantidad: int | None = None) -> str:
        n = cantidad if cantidad is not None else len(self.posiciones())
        return f"{n}x{self.barra} ({self.nx}x{self.ny})"


@dataclass(frozen=True)
class Malla:
    barra: str
    separacion: float
    cortinas: int
    automatica: bool = False          # separación calculada con la cuantía mínima

    @property
    def cuantia(self) -> float:
        """ρ = cortinas·Ab/(s·b), con la separación declarada (o calculada)."""
        return self.cortinas * area(self.barra) / (self.separacion * D.ESPESOR)

    def texto(self) -> str:
        return f"{self.barra}@{self.separacion * 100:.0f} ({self.cortinas}c)"


@dataclass(frozen=True)
class Armado:
    nombre: str
    malla_vertical: Malla
    malla_horizontal: Malla
    elementos: tuple
    debajo: "Armado | None" = None    # en el reducido, la base: sus barras continúan las de ella

    def elemento(self, nombre: str) -> Elemento | None:
        return next((e for e in self.elementos if e.nombre == nombre), None)

    def texto(self) -> str:
        b = colocar_barras(self)
        partes = [f"malla {self.malla_vertical.texto()}"]
        for e in self.elementos:
            partes.append(f"{e.nombre} {e.texto(int((b.zona == e.nombre).sum()))}")
        return " | ".join(partes)


CLAVES_ELEMENTO = {"barra", "nx", "ny", "estribo", "separacion_estribo", "separacion_estribo_arriba",
                   "longitud", "grapas"}
CLAVES_MALLA = {"barra", "separacion", "cortinas"}
BLOQUES = {"izquierdo": "BORDE_IZQUIERDO", "derecho": "BORDE_DERECHO", "ala": "BORDE_ALA"}


def _bloque(nombre: str) -> dict:
    valor = getattr(D, nombre, None)
    if not isinstance(valor, dict):
        raise DatoInvalido(f"falta {nombre} (un diccionario) en el archivo de datos")
    return valor


def validar_longitud(L, donde: str, con_minimo: bool = True) -> float:
    """Una longitud confinada de un extremo: un número positivo y, en el
    armado de base (con_minimo), no menor que 0.15·ℓw (criterio del proyecto,
    solo en la base; en el reducido termina en una columna de barras de la
    base, `longitudes_sobre_la_base`, y cumple §18.10.6.5(b))."""
    minima = 0.15 * D.LW
    if L is None:
        raise DatoInvalido(f"{donde}: falta «longitud», la longitud confinada del extremo en m "
                           f"(no menor que 0.15·ℓw = {minima:.3f} m)")
    if isinstance(L, bool) or not isinstance(L, (int, float)) or L <= 0:
        raise DatoInvalido(f"{donde}: «longitud» tiene que ser un número positivo; es {L!r}")
    if con_minimo and L < minima - 1e-9:
        raise DatoInvalido(f"{donde}: «longitud» = {L:g} m es menor que 0.15·ℓw = {minima:.3f} m "
                           f"(el mínimo confinado del proyecto)")
    return float(L)


def longitud_declarada(nombre: str) -> float:
    """La longitud confinada de un extremo en el armado de base: dato
    obligatorio y no menor que 0.15·ℓw (criterio del proyecto)."""
    donde = BLOQUES[nombre]
    return validar_longitud(_bloque(donde).get("longitud"), donde)


def cajas(Li: float | None = None, Ld: float | None = None, donde: str = "") -> dict:
    """La caja de cada elemento confinado, con las longitudes de los extremos
    (por defecto, las del armado de base). `donde`: para los mensajes."""
    sec = seccion()
    lw, b = sec.lw, sec.b
    if Li is None:
        Li = longitud_declarada("izquierdo")
    if Ld is None:
        Ld = longitud_declarada("derecho")
    previo = f"{donde}: " if donde else ""
    if sec.tipo == "III":
        Ld = max(Ld, sec.ancho_ala)              # si no llega a b', todo el ancho del ala
    salida = {"izquierdo": Rect(0.0, Li, -b / 2, b / 2), "derecho": Rect(lw - Ld, lw, -b / 2, b / 2)}
    if Li + Ld > lw + 1e-9:
        raise DatoInvalido(f"{previo}los elementos de los extremos ({Li:.2f} + {Ld:.2f} m) no caben "
                           f"en ℓw = {lw:.2f} m")
    if sec.ala is not None:
        salida["ala"] = sec.ala
        if sec.tipo == "II" and (sec.ala.x0 < Li - 1e-9 or sec.ala.x1 > lw - Ld + 1e-9):
            raise DatoInvalido(f"{previo}el ala central (de {sec.ala.x0:.3f} a {sec.ala.x1:.3f} m) se "
                               f"monta sobre un elemento de extremo (izquierdo hasta {Li:.3f} m, "
                               f"derecho desde {lw - Ld:.3f} m)")
    return salida


def _elemento(nombre: str, d: dict, donde: str, caja: Rect) -> Elemento:
    desconocidas = set(d) - CLAVES_ELEMENTO
    if desconocidas:
        raise DatoInvalido(f"{donde}: no conozco {sorted(desconocidas)}; las claves son "
                           f"{sorted(CLAVES_ELEMENTO)}")
    if "barra" not in d:
        raise DatoInvalido(f"{donde}: falta «barra», el diámetro de sus barras verticales")
    barra = d.get("barra")
    if barra not in D.BARRAS:
        raise DatoInvalido(f"{donde}: la barra {barra!r} no está en BARRAS")
    for k in ("nx", "ny"):
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 2:
            raise DatoInvalido(f"{donde}: «{k}» es un entero ≥ 2 (barras por cara); es {v!r}")
    estribo, s = d.get("estribo"), d.get("separacion_estribo")
    if estribo is not None and estribo not in D.BARRAS:
        raise DatoInvalido(f"{donde}: el estribo {estribo!r} no está en BARRAS")
    if estribo is not None and (isinstance(s, bool) or not isinstance(s, (int, float)) or s <= 0):
        raise DatoInvalido(f"{donde}: «separacion_estribo» tiene que ser un número positivo; es {s!r}")
    if estribo is None and s is not None:
        raise DatoInvalido(f"{donde}: hay «separacion_estribo» = {s!r} pero no «estribo»: declara el "
                           f"estribo o quita la separación")
    # opcional: la separación encima de la altura de cambio (`altura_de_cambio`)
    arriba = d.get("separacion_estribo_arriba")
    if arriba is not None:
        if estribo is None:
            raise DatoInvalido(f"{donde}: hay «separacion_estribo_arriba» pero no «estribo»")
        if isinstance(arriba, bool) or not isinstance(arriba, (int, float)) or arriba <= 0:
            raise DatoInvalido(f"{donde}: «separacion_estribo_arriba» tiene que ser un número positivo "
                               f"(o no ponerla: la misma de abajo); es {arriba!r}")
    grapas = d.get("grapas", "alternas")
    if grapas not in ("alternas", "todas"):
        raise DatoInvalido(f"{donde}: «grapas» es \"alternas\" o \"todas\", no {grapas!r}")
    # los extremos del alma: el estribo llega al borde de la caja del lado del alma
    alma = {"izquierdo": "x1", "derecho": "x0"}.get(nombre)
    e = Elemento(nombre, caja, barra, d["nx"], d["ny"], estribo,
                 float(s) if estribo is not None else None, grapas, alma,
                 float(arriba) if arriba is not None and abs(arriba - s) > 1e-9 else None)
    if (caja.ancho - e.a_eje("x0") - e.a_eje("x1") <= 1e-6
            or caja.alto - e.a_eje("y0") - e.a_eje("y1") <= 1e-6):
        raise DatoInvalido(f"{donde}: en la caja de {caja.ancho * 100:.1f} × {caja.alto * 100:.1f} cm "
                           f"no caben barras a {e.r * 100:.2f} cm de las caras del muro "
                           f"(recubrimiento + estribo + Ø/2)")
    return e


def _malla(d: dict, donde: str, rho_min: float, tope: float = 0.45) -> Malla:
    desconocidas = set(d) - CLAVES_MALLA
    if desconocidas:
        raise DatoInvalido(f"{donde}: no conozco {sorted(desconocidas)}; las claves son "
                           f"{sorted(CLAVES_MALLA)}")
    if "barra" not in d:
        raise DatoInvalido(f"{donde}: falta «barra», el diámetro de las barras de la malla")
    barra, cortinas = d.get("barra"), d.get("cortinas", 2)
    if barra not in D.BARRAS:
        raise DatoInvalido(f"{donde}: la barra {barra!r} no está en BARRAS")
    if isinstance(cortinas, bool) or cortinas not in (1, 2):
        raise DatoInvalido(f"{donde}: «cortinas» es 1 o 2; es {cortinas!r}")
    s = d.get("separacion")
    if s is None:
        # la mayor separación de replanteo (múltiplo de 2 o 5 cm), no mayor
        # que `tope`, que da la cuantía mínima
        s_rho = cortinas * area(barra) / (rho_min * D.ESPESOR)
        s = replanteo_hasta(min(s_rho, tope))
        if s < 0.05 - 1e-12:
            raise DatoInvalido(f"{donde}: con {cortinas} cortina(s) de {barra} la cuantía {rho_min} "
                               f"pide menos de 5 cm de separación; usa una barra mayor")
        return Malla(barra, round(s, 4), cortinas, True)
    if isinstance(s, bool) or not isinstance(s, (int, float)) or s <= 0:
        raise DatoInvalido(f"{donde}: «separacion» tiene que ser un número positivo o None; es {s!r}")
    return Malla(barra, float(s), cortinas, False)


def _mallas(vertical: dict, donde_v: str, horizontal: dict, donde_h: str) -> tuple[Malla, Malla]:
    """Las dos mallas. Con «separacion»: None, la separación sale de la cuantía
    mínima, sin pasar de 45 cm (§18.10.2.1), de 3h y de ℓw/5 la horizontal y
    ℓw/3 la vertical (§11.7.3.1 y §11.7.2.1, si el cortante pide refuerzo:
    aquí todavía no se conoce el cortante, se toman siempre, del lado seguro)."""
    tope_h = min(0.45, 3 * D.ESPESOR, D.LW / 5)
    tope_v = min(0.45, 3 * D.ESPESOR, D.LW / 3)
    mh = _malla(horizontal, donde_h, 0.0025, tope_h)
    # §18.10.2.1: ρℓ ≥ 0.0025; §18.10.4.3: en muros robustos (hw/ℓw ≤ 2), ρℓ ≥ ρt
    # (con la malla horizontal del mismo armado)
    rho_v = max(0.0025, mh.cuantia) if ESBELTEZ <= 2.0 else 0.0025
    return _malla(vertical, donde_v, rho_v, tope_v), mh


def nombres_de_elementos() -> list[str]:
    tipo = seccion().tipo
    return {"I": ["izquierdo", "derecho"], "II": ["izquierdo", "ala", "derecho"],
            "III": ["izquierdo", "derecho", "ala"]}[tipo]


def absorbido(e: Elemento, armado: Armado) -> bool:
    """Tipo III: el elemento derecho no tiene barras propias porque todas sus
    columnas caen dentro del ala (allí mandan las del ala)."""
    ala = armado.elemento("ala")
    return (seccion().tipo == "III" and e.nombre == "derecho" and ala is not None
            and all(x > ala.caja.x0 - 1e-9 for x in e.columnas()))


def _comprobar_absorcion(armado: Armado, donde: str) -> None:
    """Si el derecho del tipo III queda sin barras propias, su zona confinada
    es la del ala (b'): una «longitud» mayor que b' no estaría confinada por
    ningún estribo."""
    der = armado.elemento("derecho")
    sec = seccion()
    if der is not None and absorbido(der, armado) and der.caja.ancho > sec.ancho_ala + 1e-9:
        raise DatoInvalido(f"{donde}: la «longitud» del derecho, {der.caja.ancho:.3f} m, pasa de b' = "
                           f"{sec.ancho_ala:.3f} m pero todas sus columnas de barras caen dentro del ala, "
                           f"así que su estribo no confinaría nada fuera de ella: usa longitud ≤ b' (lo "
                           f"absorbe el ala) o una que deje una columna fuera del ala")


def armado_de_base() -> Armado:
    return armar({n: _bloque(BLOQUES[n]) for n in nombres_de_elementos()},
                 _bloque("MALLA_VERTICAL"), _bloque("MALLA_HORIZONTAL"))


def armar(bloques: dict, malla_v: dict, malla_h: dict, nombre: str = "base",
          debajo: Armado | None = None) -> Armado:
    """Un armado completo a partir de diccionarios con el formato del archivo
    de datos: {"izquierdo": BORDE_IZQUIERDO, "derecho": …, "ala": …} y las dos
    mallas. Lo usan `armado_de_base` y el optimizador (que arma miles). El
    mínimo de 0.15·ℓw de la «longitud» rige solo en la base. debajo: en un
    reducido, la base (sus barras tienen que continuar las de ella)."""
    base = nombre == "base"
    caja = cajas(validar_longitud(bloques["izquierdo"].get("longitud"), BLOQUES["izquierdo"], base),
                 validar_longitud(bloques["derecho"].get("longitud"), BLOQUES["derecho"], base))
    if "ala" in caja and "longitud" in bloques["ala"]:
        raise DatoInvalido("BORDE_ALA: «longitud» no va en el ala: su caja es el ala entera, b' × l'w")
    mv, mh = _mallas(malla_v, "MALLA_VERTICAL", malla_h, "MALLA_HORIZONTAL")
    elementos = tuple(_elemento(n, bloques[n], BLOQUES[n], caja[n]) for n in nombres_de_elementos())
    armado = Armado(nombre, mv, mh, elementos, debajo)
    _comprobar_absorcion(armado, "BORDE_DERECHO")
    if debajo is not None:
        comprobar_empalmes(armado)
    return armado


# En el reducido cambia el acero vertical (barra, nx, ny de cada elemento y la
# separación de la malla vertical), el horizontal (la separación de la malla
# horizontal y el estribo, su separación y las grapas de cada elemento) y la
# «longitud» confinada de los extremos (no más que la de la base, terminada en
# una columna de barras de la base y sin el mínimo de 0.15·ℓw, que es solo de
# la base: en su lugar, máx(c − 0.1·ℓw, c/2) de §18.10.6.5(b) en cada cota; el
# resto de reglas, como en la base; el ala es siempre el ala entera). La barra
# de cada malla es la de la base.
# Cada barra vertical del reducido continúa (empalma con) una barra de la base,
# en su misma posición y sin ser más gruesa (`comprobar_empalmes`).
CLAVES_REDUCIDO_ELEMENTO = {"barra", "nx", "ny", "estribo", "separacion_estribo", "grapas",
                            "longitud"}


def _sub_bloque(r: dict, clave: str) -> dict:
    """Un bloque de ARMADO_REDUCIDO: un diccionario (vacío si no se declara)."""
    valor = r.get(clave, {})
    if not isinstance(valor, dict):
        raise DatoInvalido(f"ARMADO_REDUCIDO, {clave}: tiene que ser un diccionario (como "
                           f"{{\"barra\": ...}}); es {valor!r}")
    return dict(valor)


def armado_reducido(base: Armado) -> Armado | None:
    """El armado de encima de la cota de corte. Cambia el acero vertical, el
    horizontal y la longitud confinada de los extremos (CLAVES_REDUCIDO_ELEMENTO
    y las dos mallas). Lo que no se declara se toma de la base."""
    r = getattr(D, "ARMADO_REDUCIDO", None)
    if r is None:
        return None
    if not isinstance(r, dict):
        raise DatoInvalido("ARMADO_REDUCIDO es un diccionario o None")
    if not r:
        raise DatoInvalido("ARMADO_REDUCIDO está vacío: no cambia nada del armado de base; para no "
                           "reducir pon ARMADO_REDUCIDO = None")
    claves = {"malla_vertical", "malla_horizontal"} | {f"borde_{n}" for n in nombres_de_elementos()}
    desconocidas = set(r) - claves
    if desconocidas:
        raise DatoInvalido(f"ARMADO_REDUCIDO: no conozco {sorted(desconocidas)}; las claves son "
                           f"{sorted(claves)}")
    for clave in ("malla_vertical", "malla_horizontal"):
        desconocidas = set(_sub_bloque(r, clave)) - CLAVES_MALLA
        if desconocidas:
            raise DatoInvalido(f"ARMADO_REDUCIDO, {clave}: no conozco {sorted(desconocidas)}; las "
                               f"claves son {sorted(CLAVES_MALLA)}")
        # criterio del proyecto: cada malla del alma lleva la misma barra en la
        # base y en el reducido, por el proceso constructivo
        de_base = _bloque(clave.upper()).get("barra")
        barra = _sub_bloque(r, clave).get("barra", de_base)
        if barra != de_base:
            raise DatoInvalido(f"ARMADO_REDUCIDO, {clave}: la barra {barra!r} tiene que ser la de "
                               f"{clave.upper()}, {de_base!r} (criterio del proyecto: cada malla del alma "
                               f"lleva el mismo diámetro en la base y en el reducido, por el proceso "
                               f"constructivo; en el reducido cambia su separación)")
    mv, mh = _mallas({**_bloque("MALLA_VERTICAL"), **_sub_bloque(r, "malla_vertical")},
                     "ARMADO_REDUCIDO, malla_vertical",
                     {**_bloque("MALLA_HORIZONTAL"), **_sub_bloque(r, "malla_horizontal")},
                     "ARMADO_REDUCIDO, malla_horizontal")
    if mv.automatica:
        # la mayor separación sobre las líneas de la base (k veces la suya) que
        # no pasa de la calculada; al menos la de la base
        validas = separaciones_sobre_la_base(base)
        mv = replace(mv, separacion=max([s for s in validas if s <= mv.separacion + 1e-9] or validas[:1]))
    # la longitud confinada de cada extremo (la de la base si no se declara)
    longitudes = {}
    for nombre in ("izquierdo", "derecho"):
        cambio = _sub_bloque(r, f"borde_{nombre}")
        # sin el mínimo de 0.15·ℓw, que es solo de la base; que quepan las
        # barras lo revisa _elemento y que termine en una columna de la base,
        # comprobar_empalmes
        longitudes[nombre] = (validar_longitud(cambio["longitud"], f"ARMADO_REDUCIDO, borde_{nombre}",
                                               con_minimo=False)
                              if "longitud" in cambio else longitud_declarada(nombre))
        # el borde especial se decide en la sección crítica, en la base
        # (§18.10.6.2): encima no hace falta confinar más que abajo
        if longitudes[nombre] > longitud_declarada(nombre) + 1e-9:
            raise DatoInvalido(f"ARMADO_REDUCIDO, borde_{nombre}: «longitud» = {longitudes[nombre]:g} m "
                               f"es mayor que la de la base ({longitud_declarada(nombre):g} m); la "
                               f"necesidad de borde especial se decide en la base (§18.10.6.2) y por "
                               f"encima no hace falta confinar más: usa una longitud ≤ "
                               f"{longitud_declarada(nombre):g} m")
    caja = cajas(longitudes["izquierdo"], longitudes["derecho"], "ARMADO_REDUCIDO")
    elementos = []
    for e in base.elementos:
        donde = f"ARMADO_REDUCIDO, borde_{e.nombre}"
        cambio = _sub_bloque(r, f"borde_{e.nombre}")
        if e.nombre == "ala" and "longitud" in cambio:
            raise DatoInvalido(f"{donde}: «longitud» no va en el ala: su caja es el ala entera, "
                               f"b' × l'w")
        prohibidas = set(cambio) - CLAVES_REDUCIDO_ELEMENTO
        if prohibidas:
            raise DatoInvalido(f"{donde}: no conozco {sorted(prohibidas)}; en el reducido cambian "
                               f"{sorted(CLAVES_REDUCIDO_ELEMENTO)}")
        cambio.pop("longitud", None)            # ya está en la caja
        # lo no declarado, de la base (la separación de estribos, la de arriba
        # si la tiene: el reducido va encima de la altura de cambio)
        d = {"barra": e.barra, "nx": e.nx, "ny": e.ny, "estribo": e.estribo,
             "separacion_estribo": e.separacion_arriba or e.separacion_estribo, "grapas": e.grapas}
        if "estribo" in cambio and cambio["estribo"] is None:
            d["separacion_estribo"] = None          # sin estribo no hay separación que heredar
        d.update(cambio)
        elementos.append(_elemento(e.nombre, d, donde, caja[e.nombre]))
    reducido = Armado("reducido", mv, mh, tuple(elementos), base)
    if replace(reducido, nombre="base", debajo=None) == base:
        raise DatoInvalido("ARMADO_REDUCIDO no cambia nada del armado de base (todo lo que declara "
                           "es igual a la base); para no reducir pon ARMADO_REDUCIDO = None")
    _comprobar_absorcion(reducido, "ARMADO_REDUCIDO, borde_derecho")
    comprobar_empalmes(reducido)
    return reducido


def longitudes_sobre_la_base(eb: Elemento, base: Armado) -> list[float]:
    """Las «longitudes» confinadas que puede tener un extremo en el reducido
    (criterio del proyecto): la zona confinada termina en una
    columna de barras de la base, la última que el reducido conserva, con el
    estribo de la base pegado a ella del lado del alma (sin recubrimiento,
    como en la base). eb: el extremo en la base. Cada una al centímetro (si
    así se sale de medio diámetro de la barra, al milímetro). En el tipo III,
    el derecho puede quedar además dentro del ala (≤ b': lo absorbe el ala).
    El mínimo de §18.10.6.5(b), máx(c − 0.1·ℓw, c/2), lo comprueba el
    detallado en cada cota."""
    sec = seccion()
    ala = base.elemento("ala")
    tipo_iii = sec.tipo == "III" and eb.nombre == "derecho"
    a = eb.a_eje(eb.hacia_el_alma)              # de la columna al borde de la caja, hacia el alma
    columnas = sorted(eb.columnas(), reverse=eb.hacia_el_alma == "x0")[1:]   # sin la de la cara
    salida = [eb.caja.ancho]
    if tipo_iii:
        salida.append(sec.ancho_ala)
    for x in columnas:
        if tipo_iii and x > ala.caja.x0 - 1e-9:
            continue                            # dentro del ala: allí mandan las del ala
        L = x + a - eb.caja.x0 if eb.hacia_el_alma == "x1" else eb.caja.x1 - (x - a)
        redondeada = round(L, 2) if abs(round(L, 2) - L) <= diametro(eb.barra) / 2 else round(L, 3)
        salida.append(redondeada)
    unicas: list[float] = []
    for L in sorted(salida):
        if L <= eb.caja.ancho + 1e-9 and not any(abs(L - u) < 1e-6 for u in unicas):
            unicas.append(round(float(L), 4))
    return unicas


def comprobar_empalmes(red: Armado) -> None:
    """Criterio del proyecto: el acero vertical del reducido continúa el de
    la base. La barra de cada elemento (extremos y ala) no es más gruesa que
    la de ese elemento en la base, la zona confinada de cada extremo termina
    en una columna de barras de la base (`longitudes_sobre_la_base`), y cada
    barra del reducido, de los elementos y de la malla, empalma con una barra
    de la base (`colocar_barras`, que lo comprueba al ponerlas). La malla
    vertical del reducido va sobre una de cada k líneas de la base: su
    separación es k veces la de la base (`separaciones_sobre_la_base`)."""
    validas = separaciones_sobre_la_base(red.debajo)
    sr = red.malla_vertical.separacion
    if not any(abs(sr - s) < 1e-6 for s in validas):
        lista = ", ".join(f"{s * 100:.0f}" for s in validas)
        raise DatoInvalido(f"ARMADO_REDUCIDO, malla_vertical: separación {sr * 100:g} cm; con la malla de la "
                           f"base @{red.debajo.malla_vertical.separacion * 100:g} cm puede ser {lista} cm "
                           f"(criterio del proyecto: la malla del reducido va sobre una de cada k líneas de "
                           f"la base, así que su separación es k veces la de la base)")
    for e in red.elementos:
        eb = red.debajo.elemento(e.nombre)
        if eb is not None and diametro(e.barra) > diametro(eb.barra) + 1e-9:
            raise DatoInvalido(f"ARMADO_REDUCIDO, borde_{e.nombre}: la barra {e.barra} es más gruesa que "
                               f"la de la base, {eb.barra} (criterio del proyecto: las barras del reducido "
                               f"continúan las de la base y son de diámetro igual o menor)")
        if eb is None or e.nombre == "ala":
            continue
        validas = longitudes_sobre_la_base(eb, red.debajo)
        if not any(abs(e.caja.ancho - L) < 1e-6 for L in validas):
            lista = " o ".join(f"{L:g}" for L in validas)
            ala = (f" (en el tipo III, {seccion().ancho_ala:g} m es cualquier longitud ≤ b': la absorbe "
                   f"el ala)") if seccion().tipo == "III" and e.nombre == "derecho" else ""
            raise DatoInvalido(f"ARMADO_REDUCIDO, borde_{e.nombre}: «longitud» = {e.caja.ancho:g} m no "
                               f"termina en una columna de barras de la base; con este armado de base puede "
                               f"ser {lista} m{ala} (criterio del proyecto: la zona confinada del reducido "
                               f"termina sobre una columna de barras de la base, que sus barras continúan)")
    colocar_barras(red)


def con_separacion_de_arriba(a: Armado) -> Armado:
    """El armado con la «separacion_estribo_arriba» de cada elemento que la
    tiene en lugar de la de abajo (el mismo armado si ninguno la tiene)."""
    if not any(e.separacion_arriba for e in a.elementos):
        return a
    return replace(a, elementos=tuple(replace(e, separacion_estribo=e.separacion_arriba)
                                      if e.separacion_arriba else e for e in a.elementos))


def separaciones_sobre_la_base(base: Armado) -> list[float]:
    """Las separaciones posibles de la malla vertical del reducido: k veces
    la de la base (una de cada k líneas), sin pasar de 45 cm ni de 3h
    (§18.10.2.1, §11.7.2.1)."""
    sb = base.malla_vertical.separacion
    tope = min(0.45, 3 * D.ESPESOR)
    return [round(k * sb, 4) for k in range(1, max(int(tope / sb + 1e-9), 1) + 1)]


def avisos_del_reducido(base: Armado, red: Armado) -> list[str]:
    """Lo que el reducido tiene MÁS que la base (no es un error, pero no es
    una reducción): acero vertical, cuantía de cada malla y estribos (área de
    una rama por metro de altura)."""
    avisos = []
    a_base, a_red = colocar_barras(base).As, colocar_barras(red).As
    if a_red > a_base * (1 + 1e-9):
        avisos.append(f"el acero vertical del reducido ({a_red * 1e4:.2f} cm²) es mayor que el de la "
                      f"base ({a_base * 1e4:.2f} cm²)")
    for nombre, mb, mr in (("vertical", base.malla_vertical, red.malla_vertical),
                           ("horizontal", base.malla_horizontal, red.malla_horizontal)):
        if mr.cuantia > mb.cuantia * (1 + 1e-9):
            avisos.append(f"la malla {nombre} del reducido ({mr.texto()}, ρ = {mr.cuantia:.5f}) tiene más "
                          f"cuantía que la de la base ({mb.texto()}, ρ = {mb.cuantia:.5f})")
    for eb in base.elementos:
        er = red.elemento(eb.nombre)
        if eb.estribo is None or er is None or er.estribo is None:
            continue
        # frente a los de la base justo debajo del corte (los de arriba, si los hay)
        sb = eb.separacion_arriba or eb.separacion_estribo
        qb = area(eb.estribo) / sb
        qr = area(er.estribo) / er.separacion_estribo
        if qr > qb * (1 + 1e-9):
            avisos.append(f"los estribos del elemento {eb.nombre} del reducido ({er.estribo}@"
                          f"{er.separacion_estribo * 100:.0f}) dan más área por metro que los de la base "
                          f"({eb.estribo}@{sb * 100:.0f})")
    return avisos


@dataclass
class Barras:
    """Todas las barras verticales: posición (x, y), área, diámetro, a qué
    pertenecen (nombre del elemento o "malla") y qué barra son."""

    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    d: np.ndarray
    zona: np.ndarray
    barra: np.ndarray

    @property
    def As(self) -> float:
        return float(self.A.sum())


def validar_barras() -> None:
    """BARRAS: {nombre: (diámetro mm, área cm², masa kg/m)}, los tres
    positivos. El área y la masa nominales tienen que corresponder al
    diámetro (π·d²/4 y 7850 kg/m³ · área, con un 3 % de margen): así se
    detecta un número cambiado de columna o mal escrito."""
    barras = getattr(D, "BARRAS", None)
    if not isinstance(barras, dict) or not barras:
        raise DatoInvalido("BARRAS es un diccionario {nombre: (diámetro mm, área cm², masa kg/m)}")
    for nombre, valores in barras.items():
        if (not isinstance(valores, (tuple, list)) or len(valores) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
                       for v in valores)):
            raise DatoInvalido(f"BARRAS[{nombre!r}] tiene que ser (diámetro mm, área cm², masa kg/m), "
                               f"tres números positivos; es {valores!r}")
        d, a, m = valores
        a_geo = math.pi * (d / 10.0) ** 2 / 4.0            # cm²
        m_geo = 7850.0 * a * 1e-4                           # kg/m, con el área nominal
        if abs(a / a_geo - 1.0) > 0.03:
            raise DatoInvalido(f"BARRAS[{nombre!r}]: el área {a:g} cm² no corresponde al diámetro "
                               f"{d:g} mm (π·d²/4 = {a_geo:.2f} cm²)")
        if abs(m / m_geo - 1.0) > 0.03:
            raise DatoInvalido(f"BARRAS[{nombre!r}]: la masa {m:g} kg/m no corresponde al área "
                               f"{a:g} cm² (7850 kg/m³ · área = {m_geo:.3f} kg/m)")


def avisos_de_datos() -> list[str]:
    """Lo que no detiene el cálculo pero conviene saber de los datos."""
    avisos = []
    if D.FC < FC_MINIMO_ESPECIAL - 1e-9:
        avisos.append(f"f'c = {D.FC:.2f} MPa ({D.FC / 0.0980665:.0f} kgf/cm²) es menor que {FC_MINIMO_ESPECIAL:g} MPa, "
                      f"el mínimo de la tabla 19.2.1.1 para muros estructurales especiales con acero "
                      f"grado 420 (§19.2.1.1); para llegar hace falta f'c ≥ "
                      f"{FC_MINIMO_ESPECIAL / 0.0980665:.1f} kgf/cm². No cambia el veredicto.")
    return avisos


def validar_datos() -> None:
    """Lo que se puede comprobar del archivo de datos antes de calcular nada."""
    validar_barras()
    sec = seccion()
    if sec.lw / sec.b <= 3.0:
        raise DatoInvalido(f"ℓw/b = {sec.lw:g}/{sec.b:g} = {sec.lw / sec.b:.2f}: un muro tiene ℓw/b > 3 "
                           f"(definición de «wall», capítulo 2); con eso es una columna")
    if sec.tipo == "I" and getattr(D, "BORDE_ALA", None) is not None:
        raise DatoInvalido("BORDE_ALA va solo en los tipos II y III; en el tipo I (sin ala) pon "
                           "BORDE_ALA = None")
    _numero("RECUBRIMIENTO")
    if not D.RECUBRIMIENTO < sec.b / 2:
        raise DatoInvalido(f"RECUBRIMIENTO = {D.RECUBRIMIENTO:g} m no deja sitio en el espesor")
    if _numero("DELTA_U_MM") <= 0.0:
        raise DatoInvalido("DELTA_U_MM tiene que ser positivo")
    combos = getattr(D, "COMBINACIONES", None)
    if not isinstance(combos, (list, tuple)) or not combos or not all(isinstance(c, str) for c in combos):
        raise DatoInvalido(f"COMBINACIONES es una lista con los nombres de las combinaciones de ETABS; "
                           f"es {combos!r}")
    lista = getattr(D, "COMBINACIONES_DEL_SISMO_EN_X", None)
    if not isinstance(lista, (list, tuple)) or not lista or any(c not in combos for c in lista):
        raise DatoInvalido(f"COMBINACIONES_DEL_SISMO_EN_X tiene que ser una lista de "
                           f"combinaciones de COMBINACIONES; es {lista!r}")
    if getattr(D, "AGREGADO_MM", None) is not None:
        _numero("AGREGADO_MM")
    for opcion in ("HORIZONTAL_POR_FUERA", "AXIAL_CON_PESO_PROPIO", "SISMO_ENTRE_BASE_Y_TOPE",
                   "PERMUTAR_SIGNOS"):
        if not isinstance(getattr(D, opcion, None), bool):
            raise DatoInvalido(f"{opcion} es True o False; es {getattr(D, opcion, None)!r}")
    if not 0.0 < D.FY <= 420.0 + 1e-9:
        raise DatoInvalido(f"FY = {D.FY:g} MPa: el programa trae los límites del grado 420 "
                           f"(tabla 18.10.6.5(b), §18.7.5.3(b)), que valen para fy ≤ 420 MPa "
                           f"(4200 kgf/cm² = 411.88 MPa); para un grado mayor hay que cambiarlos")


def y_de_malla(mv: Malla, mh: Malla) -> float:
    """Distancia del eje del alma a las barras verticales de la malla."""
    if mv.cortinas == 1:
        return 0.0
    return (D.ESPESOR / 2 - D.RECUBRIMIENTO
            - (diametro(mh.barra) if D.HORIZONTAL_POR_FUERA else 0.0) - diametro(mv.barra) / 2)


def tramos_libres(armado: Armado) -> list[tuple[float, float]]:
    """Los tramos del alma entre elementos, (desde, hasta) en x."""
    intervalos = sorted((e.caja.x0, e.caja.x1) for e in armado.elementos)
    unidos: list[list[float]] = []
    for a, b in intervalos:
        if unidos and a <= unidos[-1][1] + 1e-9:
            unidos[-1][1] = max(unidos[-1][1], b)
        else:
            unidos.append([a, b])
    return [(unidos[i][1], unidos[i + 1][0]) for i in range(len(unidos) - 1)
            if unidos[i + 1][0] - unidos[i][1] > 1e-9]


def colocar_barras(armado: Armado) -> Barras:
    """Pone cada barra vertical en planta.

    ELEMENTOS: el perímetro de cada uno (ver `Elemento.posiciones`). En el tipo
    III las barras del elemento derecho que caen dentro del ala no se ponen:
    allí mandan las del ala.
    MALLA: en cada tramo libre del alma, entre la última columna de barras del
    elemento de un lado y la primera del otro, líneas exactamente a la
    separación de la malla, centradas (los dos huecos de las puntas iguales,
    entre s/2 y s; si el hueco no pasa de s, ninguna); las que cayeran dentro
    de un elemento se quitan.
    Cada línea lleva una barra por cortina.
    EL REDUCIDO (con `debajo`, la base): las barras de cada elemento se ponen
    con la barra y el estribo que ese elemento tiene en la base (las mismas
    líneas), cada una continúa una barra de la base y toma su posición
    (`empalme`), y la malla va sobre líneas de barras de la base
    (`lineas_sobre_la_base`) en lugar de a distancias iguales.
    """
    sec = seccion()
    ala = armado.elemento("ala")
    base = colocar_barras(armado.debajo) if armado.debajo is not None else None
    usadas: set = set()
    x, y, A, d, zona, nombre = [], [], [], [], [], []

    def poner(xi, yi, barra, z):
        if base is not None:
            xi, yi = empalme(base, xi, yi, barra, z, usadas)
        x.append(xi)
        y.append(yi)
        A.append(area(barra))
        d.append(diametro(barra))
        zona.append(z)
        nombre.append(barra)

    for e in armado.elementos:
        # en el reducido, sobre las líneas de la base: con la barra y el
        # estribo de ese elemento en la base (otros diámetros no las mueven)
        eb = armado.debajo.elemento(e.nombre) if armado.debajo is not None else None
        lineas_de = replace(e, barra=eb.barra, estribo=eb.estribo) if eb is not None else e
        for xi, yi in lineas_de.posiciones():
            if (sec.tipo == "III" and e.nombre == "derecho" and ala is not None
                    and xi > ala.caja.x0 - 1e-9):
                continue
            poner(xi, yi, e.barra, e.nombre)
    mv, mh = armado.malla_vertical, armado.malla_horizontal
    ym = y_de_malla(mv, mh)
    en_alma = [(xi, yi) for xi, yi in zip(x, y) if abs(yi) <= sec.b / 2 + 1e-9]
    for desde, hasta in tramos_libres(armado):
        x_izq = max(xi for xi, _ in en_alma if xi <= desde + 1e-9)
        x_der = min(xi for xi, _ in en_alma if xi >= hasta - 1e-9)
        if base is not None:
            lineas = lineas_sobre_la_base(base, desde, hasta, x_izq, x_der, mv.separacion)
        else:
            # a la separación declarada, centradas en el tramo: los huecos de
            # las puntas quedan iguales, entre s/2 y s
            s = mv.separacion
            hueco = x_der - x_izq
            n = max(math.ceil(hueco / s - 1e-9) - 1, 0)
            inicio = x_izq + (hueco - (n - 1) * s) / 2
            lineas = [inicio + j * s for j in range(n)]
        for xi in lineas:
            if not (desde + 1e-9 < xi < hasta - 1e-9):
                continue                      # caería dentro de un elemento
            for lado in ([0.0] if mv.cortinas == 1 else [-1.0, 1.0]):
                poner(xi, lado * ym, mv.barra, "malla")
    return Barras(np.array(x), np.array(y), np.array(A), np.array(d), np.array(zona),
                  np.array(nombre))


def empalme(base: Barras, xi: float, yi: float, barra: str, zona: str, usadas: set) -> tuple[float, float]:
    """Criterio del proyecto: cada barra del reducido continúa
    (empalma con) una barra de la base. Las dos ocupan la misma posición: el
    reducido se coloca sobre las líneas de la base (`colocar_barras`) y se
    acepta hasta medio diámetro de la barra de la base en cada dirección (p.
    ej. la columna interior de un extremo más corto que cae casi sobre una de
    la base). La de la base no puede ser más delgada, ni continuar dos barras
    del reducido. Devuelve la posición de la barra de la base, que es la que
    se usa."""
    bloque = "malla_vertical" if zona == "malla" else f"borde_{zona}"
    donde = f"ARMADO_REDUCIDO, {bloque}: la barra {barra} en (x, y) = ({xi * 100:.1f}, {yi * 100:+.1f}) cm"
    cerca = (np.abs(base.x - xi) <= base.d / 2 + 1e-9) & (np.abs(base.y - yi) <= base.d / 2 + 1e-9)
    if not cerca.any():
        raise DatoInvalido(f"{donde} no continúa ninguna barra de la base (criterio del proyecto: cada "
                           f"barra del reducido empalma con una barra de la base, en su misma posición); "
                           f"cambia nx, ny, la «longitud» o la separación para que caiga sobre una")
    k = int(np.argmin(np.where(cerca, np.hypot(base.x - xi, base.y - yi), np.inf)))
    if base.d[k] < diametro(barra) - 1e-9:
        raise DatoInvalido(f"{donde} continúa una barra de {base.barra[k]} de la base ({base.zona[k]}), más "
                           f"delgada (criterio del proyecto: la barra del reducido es de diámetro igual o "
                           f"menor que la de la base que continúa)")
    if k in usadas:
        raise DatoInvalido(f"{donde} cae sobre una barra de la base que ya continúa otra del reducido")
    usadas.add(k)
    return float(base.x[k]), float(base.y[k])


def lineas_sobre_la_base(base: Barras, desde: float, hasta: float, x_izq: float, x_der: float,
                         s: float) -> list[float]:
    """La malla vertical del reducido en un tramo libre del alma, entre las
    columnas x_izq y x_der de los elementos (criterio del proyecto): sus
    líneas van sobre líneas de barras de la base (su malla y, donde el
    reducido acorta un elemento, las columnas de él que quedan en el tramo),
    UNA DE CADA k, con k el mayor que no deja pasar ninguna separación de s
    (ninguna si el hueco no pasa de s). Si sobran líneas, el sobrante se
    reparte entre los dos lados del tramo (los huecos de las puntas, lo más
    parecidos)."""
    if x_der - x_izq <= s + 1e-9:
        return []
    en_alma = np.abs(base.y) <= seccion().b / 2 + 1e-9
    candidatas = sorted({round(float(v), 9) for v in base.x[en_alma] if desde + 1e-9 < v < hasta - 1e-9})
    if not candidatas:
        return []
    for k in range(len(candidatas), 0, -1):
        opciones = []
        for inicio in range(k):
            elegidas = candidatas[inicio::k]
            puntos = [x_izq, *elegidas, x_der]
            if all(b - a <= s + 1e-9 for a, b in zip(puntos, puntos[1:])):
                opciones.append((abs((elegidas[0] - x_izq) - (x_der - elegidas[-1])), elegidas))
        if opciones:
            return min(opciones, key=lambda o: o[0])[1]
    puntos = [x_izq, *candidatas, x_der]
    a, b = max(zip(puntos, puntos[1:]), key=lambda p: p[1] - p[0])
    raise DatoInvalido(f"ARMADO_REDUCIDO, malla_vertical: con separación {s * 100:.0f} cm no se puede seguir "
                       f"las líneas de barras de la base: de x = {a * 100:.1f} a {b * 100:.1f} cm no hay "
                       f"ninguna (criterio del proyecto: la malla del reducido continúa barras de la base); "
                       f"usa una separación mayor")


def kg_por_metro(armado: Armado) -> float:
    """Acero vertical por metro de altura (sin ganchos ni traslapes)."""
    b = colocar_barras(armado)
    return float(sum(masa(n) for n in b.barra))


def separacion_libre_exigida(db: float) -> float:
    """§25.2.3: separación libre entre barras longitudinales de elementos de
    borde: la mayor de 40 mm, 1.5·db y (4/3)·dagg."""
    agregado = (getattr(D, "AGREGADO_MM", None) or 0.0) / 1000.0
    return max(0.040, 1.5 * db, 4.0 / 3.0 * agregado)


# =============================================================================
# 3. FLEXOCOMPRESIÓN: EL ESTADO ÚLTIMO PARA UN EJE NEUTRO
# =============================================================================
#
# Hipótesis de §22.2: secciones planas, εcu = 0.003 en la fibra extrema
# comprimida, bloque rectangular de 0.85·f'c en una profundidad a = β1·c, y
# acero elastoplástico (σ = Es·ε, sin pasar de ±fy). Las barras que caen dentro
# del bloque descuentan el concreto que desplazan (σ − 0.85·f'c).
#
# El eje neutro se describe con su DIRECCIÓN n (un vector unitario que apunta
# hacia el borde comprimido) y su profundidad c. En la flexión en el plano,
# n = (1, 0) comprime el extremo derecho y n = (−1, 0) el izquierdo; n = (0, 1)
# comprime la punta del ala; girando n se obtiene la flexión biaxial.
#
# La zona comprimida del concreto es cada rectángulo de la sección RECORTADO
# por la recta que está a una profundidad a de la fibra extrema: polígonos,
# cuya área y centroide se calculan exactos (fórmula del área de Gauss). Con el
# alma y el ala en piezas que no se solapan, la intersección cuenta una vez.


def recortar(poligono, n: np.ndarray, s: float) -> list:
    """La parte del polígono convexo (parejas (x, y)) donde p·n ≥ s (algoritmo
    de Sutherland-Hodgman)."""
    nx, ny = float(n[0]), float(n[1])
    salida = []
    k = len(poligono)
    for i in range(k):
        (px, py), (qx, qy) = poligono[i], poligono[(i + 1) % k]
        dp, dq = px * nx + py * ny - s, qx * nx + qy * ny - s
        if dp >= 0.0:
            salida.append((px, py))
        if dp * dq < 0.0:
            t = dp / (dp - dq)
            salida.append((px + t * (qx - px), py + t * (qy - py)))
    return salida


def area_y_centroide(poligono: list) -> tuple[float, float, float]:
    """Área y centroide (x, y) de un polígono por la fórmula de Gauss."""
    if len(poligono) < 3:
        return 0.0, 0.0, 0.0
    a = sx = sy = 0.0
    for i in range(len(poligono)):
        x0, y0 = poligono[i]
        x1, y1 = poligono[(i + 1) % len(poligono)]
        cruz = x0 * y1 - x1 * y0
        a += cruz
        sx += (x0 + x1) * cruz
        sy += (y0 + y1) * cruz
    a /= 2.0
    if abs(a) < 1e-14:
        return 0.0, 0.0, 0.0
    return abs(a), sx / (6.0 * a), sy / (6.0 * a)


def zona_comprimida(n: np.ndarray, prof: float) -> tuple[float, float, float]:
    """Área y centroide de la parte de la sección a menos de `prof` de la fibra
    extrema en la dirección n."""
    sec = seccion()
    arriba = float((sec.esquinas @ n).max())
    Ac = sx = sy = 0.0
    for p in sec.piezas:
        a, cx, cy = area_y_centroide(recortar(p.vertices, n, arriba - prof))
        Ac += a
        sx += a * cx
        sy += a * cy
    if Ac <= 0.0:
        return 0.0, 0.0, 0.0
    return Ac, sx / Ac, sy / Ac


def p0_nominal(barras: Barras) -> float:
    """Compresión pura nominal: P0 = 0.85·f'c·(Ag − As) + fy·As (§22.4.2.2)."""
    return 0.85 * D.FC * MPA * (area_bruta() - barras.As) + D.FY * MPA * barras.As


@dataclass
class Punto:
    """Estado último de la sección. P y M nominales; φ aparte."""

    c: float
    n: tuple
    P: float          # kN, negativo en compresión
    Mx: float         # kN·m, en el plano
    My: float         # kN·m, en Y
    eps_t: float      # deformación de la barra más traccionada (positiva en tracción)
    phi: float = 0.0
    detalle: dict = field(default_factory=dict, repr=False)


def estado_ultimo(barras: Barras, n: np.ndarray, c: float, detalle: bool = False) -> Punto:
    """Pn, Mn para un eje neutro de dirección n y profundidad c."""
    return estados_ultimos(barras, n, (c,), detalle)[0]


def estados_ultimos(barras: Barras, n: np.ndarray, cs, detalle: bool = False) -> list[Punto]:
    """`estado_ultimo` para cada profundidad c de `cs`, con el mismo n: el
    acero de todas a la vez (una fila por c), que va mucho más rápido en la
    superficie biaxial y el diagrama P-M."""
    sec = seccion()
    cs = [float(c) for c in cs]
    proy = barras.x * n[0] + barras.y * n[1]                # cada barra, proyectada sobre n
    arriba = max(float((sec.esquinas @ n).max()), float(proy.max()))
    sigma_bloque = 0.85 * D.FC * MPA
    # concreto: cada pieza recortada a la profundidad a = β1·c
    concreto = []
    for c in cs:
        a = BETA1 * c
        Ac = sxc = syc = 0.0
        for p in sec.piezas:
            ap, cx, cy = area_y_centroide(recortar(p.vertices, n, arriba - a))
            Ac += ap
            sxc += ap * cx
            syc += ap * cy
        concreto.append((c, a, Ac, *((sxc / Ac, syc / Ac) if Ac > 0 else (0.0, 0.0))))
    c_, a, Ac, cx, cy = np.array(concreto).T                # un valor por c
    Cc = sigma_bloque * Ac
    # acero: una fila por c, una columna por barra
    prof = arriba - proy                                    # profundidad de cada barra
    eps = EPS_CU * (1.0 - prof / c_[:, None])               # + compresión
    sigma = np.clip(D.ES * MPA * eps, -D.FY * MPA, D.FY * MPA)
    sigma = np.where(prof <= a[:, None], sigma - sigma_bloque, sigma)
    Fs = sigma * barras.A
    fuerzas = (Cc + Fs.sum(axis=1)).tolist()                # + compresión
    mxs = (Cc * (cx - sec.xc) + (Fs * (barras.x - sec.xc)).sum(axis=1)).tolist()
    mys = (Cc * (cy - sec.yc) + (Fs * (barras.y - sec.yc)).sum(axis=1)).tolist()
    eps_t = (-eps.min(axis=1)).tolist()
    # §22.4.2.1: la compresión no pasa de 0.80·P0
    tope = 0.80 * p0_nominal(barras)
    puntos = []
    for i, c in enumerate(cs):
        fuerza, mx, my = fuerzas[i], mxs[i], mys[i]
        if fuerza > tope:
            mx, my = mx * tope / fuerza, my * tope / fuerza
            fuerza = tope
        p = Punto(c, tuple(n), -fuerza, -mx, -my, eps_t[i])
        if detalle:
            p.detalle = {"a": float(a[i]), "Ac": float(Ac[i]), "Cc": float(Cc[i]), "cx": float(cx[i]),
                         "cy": float(cy[i]), "prof": prof, "eps": eps[i], "sigma": sigma[i],
                         "Fs": Fs[i], "tope": tope}
        puntos.append(p)
    return puntos


def direccion(sentido: int) -> np.ndarray:
    """Eje neutro de la flexión en el plano: +1 comprime el extremo derecho."""
    return np.array([float(sentido), 0.0])


# =============================================================================
# 4. EL DIAGRAMA DE INTERACCIÓN EN EL PLANO
# =============================================================================


def traccion_pura(barras: Barras, n=(1.0, 0.0)) -> Punto:
    """Todo el acero en fluencia a tracción: T = fy·As, aplicada en el
    centroide del acero. Respecto del centroide de la sección, con barras
    asimétricas (tipos II y III), deja momento: Mx = fy·ΣA·(x − xc), My igual
    (el convenio de `estado_ultimo`)."""
    sec = seccion()
    fy = D.FY * MPA
    return Punto(math.inf, tuple(n), fy * barras.As,
                 fy * float((barras.A * (barras.x - sec.xc)).sum()),
                 fy * float((barras.A * (barras.y - sec.yc)).sum()), EPS_TY + 0.003, PHI_TRACCION)


def p_balanceado(barras: Barras, n: np.ndarray) -> float:
    """Pn,bal en la dirección n: el estado con la barra más traccionada justo
    en εty (§21.2.2.3), c_bal = dt·εcu/(εcu + εty)."""
    sec = seccion()
    proy = barras.x * n[0] + barras.y * n[1]
    arriba = max(float((sec.esquinas @ n).max()), float(proy.max()))
    dt = arriba - float(proy.min())
    return estado_ultimo(barras, n, dt * EPS_CU / (EPS_CU + EPS_TY)).P


def tope_de_phi(puntos: list[Punto], grupos: list[list[int]], p_bals: list[float]) -> None:
    """§21.2.2.3: entre 0.1·f'c·Ag y Pbal, φ no pasa de la recta 0.90 → 0.65.

    Un Pbal por grupo (cada sentido de flexión, o cada giro del eje neutro en
    el biaxial), de `p_balanceado`. Modifica φ en su sitio.
    """
    p_umbral = -0.1 * D.FC * MPA * area_bruta()
    for grupo, p_bal in zip(grupos, p_bals):
        if p_bal >= p_umbral:
            continue
        for i in grupo:
            p = puntos[i]
            if p_bal <= p.P <= p_umbral:
                avance = (p.P - p_umbral) / (p_bal - p_umbral)
                p.phi = min(p.phi, PHI_TRACCION + avance * (PHI_COMPRESION - PHI_TRACCION))


@dataclass
class Diagrama:
    """Curva cerrada de capacidad, nominal y de diseño, en el plano del muro."""

    puntos: list[Punto]
    p_bal: dict

    @property
    def diseno(self) -> np.ndarray:
        return np.array([[p.phi * p.P, p.phi * p.Mx] for p in self.puntos])

    @property
    def nominal(self) -> np.ndarray:
        return np.array([[p.P, p.Mx] for p in self.puntos])


def diagrama_PM(barras: Barras, n_puntos: int = D.PUNTOS_DEL_DIAGRAMA) -> Diagrama:
    """c de 0.02·ℓw (casi tracción pura) a 4·ℓw (compresión pura), en los dos sentidos.

    El eje neutro es paralelo a Y (n = ±X). Orden del polígono: tracción pura,
    sentido +1 de tracción a compresión, y sentido −1 de compresión a tracción.
    """
    cs = np.geomspace(0.02 * D.LW, 4.0 * D.LW, n_puntos)
    traccion = traccion_pura(barras)
    mas = estados_ultimos(barras, direccion(1), cs)
    menos = estados_ultimos(barras, direccion(-1), cs)
    puntos = [traccion, *mas, *reversed(menos)]
    for p in puntos[1:]:
        p.phi = phi_por_deformacion(p.eps_t)
    # la tracción pura cuenta en el grupo del sentido +1
    g_mas = list(range(0, n_puntos + 1))
    g_menos = list(range(n_puntos + 1, 2 * n_puntos + 1))
    p_bal = {+1: p_balanceado(barras, direccion(1)), -1: p_balanceado(barras, direccion(-1))}
    tope_de_phi(puntos, [g_mas, g_menos], [p_bal[1], p_bal[-1]])
    return Diagrama(puntos, p_bal)


def dc_radial(diag: Diagrama, P: float, M: float) -> float:
    """D/C de flexocompresión: cuánto hay que escalar (P, M) desde el origen
    para tocar la curva de diseño. Con P y M normalizados por sus máximos,
    para que la dirección no dependa de las unidades. Se toma el corte más
    exterior."""
    if P == 0.0 and M == 0.0:
        return 0.0
    curva = diag.diseno
    ep = float(np.abs(curva[:, 0]).max())
    em = float(np.abs(curva[:, 1]).max())
    a = np.column_stack((curva[:, 0] / ep, curva[:, 1] / em))   # cada lado del polígono, de a a b
    e = np.concatenate((a[1:], a[:1])) - a
    dx, dy = P / ep, M / em
    det = dx * (-e[:, 1]) - dy * (-e[:, 0])
    corta = np.abs(det) >= 1e-15
    det = np.where(corta, det, 1.0)
    t = (a[:, 0] * (-e[:, 1]) - a[:, 1] * (-e[:, 0])) / det
    u = (dx * a[:, 1] - dy * a[:, 0]) / det
    valido = corta & (t > 0.0) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    if not valido.any():
        raise ValueError(f"la demanda (P={P:.1f}, M={M:.1f}) no corta el diagrama")
    return 1.0 / float(t[valido].max())                    # el corte más exterior


def phi_Mn(diag: Diagrama, P: float, signo: float) -> float:
    """φMn a un axial P, del lado del momento de signo `signo` (la curva de
    diseño cortada a P constante)."""
    curva = diag.diseno
    mejor = 0.0
    for i in range(len(curva)):
        a, b = curva[i], curva[(i + 1) % len(curva)]
        dp = b[0] - a[0]
        if abs(dp) < 1e-12:
            continue
        u = (P - a[0]) / dp
        if -1e-9 <= u <= 1.0 + 1e-9:
            m = a[1] + u * (b[1] - a[1])
            if m * signo > 0.0:
                mejor = max(mejor, abs(m))
    return mejor


def eje_neutro(barras: Barras, P: float, n: np.ndarray) -> Punto:
    """El estado último con axial P y eje neutro de dirección n, por bisección
    sobre c (§18.10.6.2: «c para la carga axial mayorada y la resistencia
    nominal a momento»)."""
    canto = seccion().canto(n)
    bajo, alto = 1e-3 * canto, 10.0 * canto
    p_bajo, p_alto = estado_ultimo(barras, n, bajo).P, estado_ultimo(barras, n, alto).P
    if not (p_alto - 1e-9 <= P <= p_bajo + 1e-9):
        raise ValueError(f"P = {P:.1f} kN fuera del rango de la sección "
                         f"[{p_alto:.1f}, {p_bajo:.1f}]")
    escala = max(abs(p_bajo), abs(p_alto), 1e-3)
    for _ in range(60):
        medio = 0.5 * (bajo + alto)
        punto = estado_ultimo(barras, n, medio)
        if abs(punto.P - P) <= 1e-4 * escala:
            return punto
        if punto.P > P:
            bajo = medio
        else:
            alto = medio
    return estado_ultimo(barras, n, 0.5 * (bajo + alto))


# =============================================================================
# 5. FLEXIÓN BIAXIAL
# =============================================================================
#
# Se gira el eje neutro (72 ángulos, cada 5°) y en cada giro se barre su
# profundidad (100 valores). Cada (ángulo, c) da un punto (φPn, φMnx, φMny)
# con la misma física y el mismo φ que el diagrama en el plano. A un axial
# Pu dado se corta esa superficie: queda un contorno de momentos (φMnx, φMny),
# y el D/C es cuánto hay que escalar (Mux, Muy) para llegar a él.


class SuperficieBiaxial:
    def __init__(self, barras: Barras, resolucion: tuple[int, int] | None = None) -> None:
        """resolucion = (giros, profundidades); por defecto, los de los datos
        (el optimizador usa una más gruesa en la búsqueda)."""
        n_ang, n_prof = resolucion or (D.ANGULOS_BIAXIAL, D.PROFUNDIDADES_BIAXIAL)
        sec = seccion()
        self.P = np.empty((n_ang, n_prof + 1))
        self.Mx = np.empty_like(self.P)
        self.My = np.empty_like(self.P)
        for k in range(n_ang):
            th = 2.0 * math.pi * k / n_ang
            n = np.array([math.cos(th), math.sin(th)])
            canto = sec.canto(n)
            cs = np.geomspace(0.02 * canto, 4.0 * canto, n_prof)
            fila = [traccion_pura(barras, n)]
            for p in estados_ultimos(barras, n, cs):
                p.phi = phi_por_deformacion(p.eps_t)
                fila.append(p)
            tope_de_phi(fila, [list(range(len(fila)))], [p_balanceado(barras, n)])
            self.P[k] = [p.phi * p.P for p in fila]
            self.Mx[k] = [p.phi * p.Mx for p in fila]
            self.My[k] = [p.phi * p.My for p in fila]
        self._contornos: dict = {}

    def contorno(self, P: float) -> np.ndarray | None:
        """(φMnx, φMny) a φPn = P, uno por ángulo, ordenados por ángulo.
        None si no hay capacidad a ese axial."""
        if P in self._contornos:
            return self._contornos[P]
        a = self.P[:, :-1] - P
        b = self.P[:, 1:] - P
        cruza = (a * b <= 0.0) & (self.P[:, :-1] != self.P[:, 1:])
        u = np.where(cruza, -a / np.where(cruza, b - a, 1.0), 0.0)
        x = self.Mx[:, :-1] + u * (self.Mx[:, 1:] - self.Mx[:, :-1])
        y = self.My[:, :-1] + u * (self.My[:, 1:] - self.My[:, :-1])
        r = np.where(cruza, np.hypot(x, y), -1.0)
        j = r.argmax(axis=1)
        filas = np.arange(len(j))
        ok = r[filas, j] >= 0.0
        borde = np.column_stack((x[filas, j], y[filas, j]))[ok]
        resultado = None
        if len(borde) >= 3:
            borde = borde[np.argsort(np.arctan2(borde[:, 1], borde[:, 0]))]
            if _contiene_al_origen(borde):
                resultado = borde
        self._contornos[P] = resultado
        return resultado

    def dc(self, P: float, Mx: float, My: float) -> float:
        borde = self.contorno(P)
        if borde is None:
            return math.inf
        if Mx == 0.0 and My == 0.0:
            return 0.0
        # el rayo t·(Mx, My) contra cada lado del contorno, de a a b
        a = borde
        e = np.concatenate((borde[1:], borde[:1])) - a
        den = Mx * e[:, 1] - My * e[:, 0]
        corta = np.abs(den) > 1e-12
        t = (a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0]) / np.where(corta, den, 1.0)
        u = (((t * Mx - a[:, 0]) * e[:, 0] + (t * My - a[:, 1]) * e[:, 1])
             / np.maximum((e * e).sum(axis=1), 1e-30))
        valido = corta & (t > 0.0) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
        return 1.0 / float(t[valido].min()) if valido.any() else math.inf


def _contiene_al_origen(poligono: np.ndarray) -> bool:
    puntos = poligono.tolist()           # con números de Python el recorrido va más rápido
    dentro = False
    for i in range(len(puntos)):
        x1, y1 = puntos[i]
        x2, y2 = puntos[(i + 1) % len(puntos)]
        if (y1 > 0.0) != (y2 > 0.0) and 0.0 < x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1):
            dentro = not dentro
    return dentro


# =============================================================================
# 6. CORTE
# =============================================================================


def amplificacion() -> dict:
    """§18.10.3.3, tabla 18.10.3.3.3: Ωv y ωv, con hwcs/ℓw (la sección crítica
    en la base, hwcs = H)."""
    esb = H / D.LW
    if esb <= 1.0:
        omega = 1.0
    elif esb >= 2.0:
        omega = 1.5
    else:
        omega = 1.0 + (esb - 1.0) * 0.5
    hn_pies = D.HN / 0.3048
    w = max(1.0, 0.8 + 0.09 * hn_pies ** (1.0 / 3.0)) if esb >= 2.0 else 1.0
    return {"esbeltez": esb, "omega_v": omega, "hn_pies": hn_pies, "omega_dyn": w,
            "factor": omega * w}


def amplificar(estados: list[Estado]) -> list[Estado]:
    """§18.10.3.3.2: Ve = Vg + Ωv·ωv·VEh, con Vg la parte de gravedad del
    cortante (la combinación de gravedad de ETABS) y VEh la de sismo, con su
    signo (el caso de espectro), de SEPARACION_DEL_CORTE: se amplifica solo
    VuEh. Una combinación sin sismo (la gravitacional) no se amplifica: Ve = Vu.

    Solo el cortante EN EL PLANO: en Y el muro no trabaja como muro (en los
    tipos II y III las alas son columnas de los pórticos especiales)."""
    factor = amplificacion()["factor"]
    return [replace(e, Ve=e.Vg + factor * e.VEh) if e.con_sismo else replace(e, Ve=e.V)
            for e in estados]


def alpha_c(esbeltez: float) -> float:
    """§18.10.4.1: 0.25 si hw/ℓw ≤ 1.5, 0.17 si ≥ 2.0, lineal en medio."""
    if esbeltez <= 1.5:
        return 0.25
    if esbeltez >= 2.0:
        return 0.17
    return 0.25 + (esbeltez - 1.5) / 0.5 * (0.17 - 0.25)


def resistencia_en_el_plano(armado: Armado) -> dict:
    """Vn = (αc·λ·√f'c + ρt·fy)·Acv  (§18.10.4.1), con su tope (§18.10.4.4).

    Acv = ℓw·b (el alma). αc con hw/ℓw del muro entero: §18.10.4.2 pide la
    mayor del muro y del tramo considerado, y un muro sin aberturas es un solo
    tramo (un entrepiso no es un «wall segment», cap. 2). En §18.10.4.1 αc no
    depende de la carga axial.
    El tope es el de UN muro, 0.83·√f'c·Acv; el de 0.66·√f'c·ΣAcv para todos
    los muros que comparten la fuerza lateral no se comprueba (hace falta el
    edificio entero).
    """
    sec = seccion()
    esb = ESBELTEZ
    ac = alpha_c(esb)
    rho_t = armado.malla_horizontal.cuantia
    Vc = ac * LAMBDA * RAIZ_FC * MPA * sec.acv
    Vs = rho_t * D.FY * MPA * sec.acv
    tope = 0.83 * RAIZ_FC * MPA * sec.acv        # §18.10.4.4, αsh = 1
    return {"esbeltez": esb, "alpha_c": ac, "rho_t": rho_t, "Vc": Vc, "Vs": Vs,
            "Vn": min(Vc + Vs, tope), "tope": tope, "recortado": Vc + Vs > tope}


def phi_de_corte(barras: Barras, armado: Armado, estados: list[Estado]) -> dict:
    """§21.2.4.1: φ del cortante en el plano, una vez para el muro.

    Es 0.75 en los muros con Ωv ≥ 1.5 (la excepción del artículo). Si no, baja a
    0.60 cuando Vn es menor que el cortante que desarrolla la resistencia
    nominal a flexión: Mn el MAYOR con los axiales de TODAS las combinaciones
    con sismo E (como dice el artículo; en la base, los dos sentidos), y su
    cortante Mn·|Vu|/|Mu| con la relación Vu/Mu de cada una (la mayor).
    """
    amp = amplificacion()
    if amp["omega_v"] >= 1.5:
        return {"phi": PHI_CORTE, "aplica": False, "omega_v": amp["omega_v"]}
    sismo = [e for e in estados if e.con_sismo]
    Mn = 0.0
    for e in sismo:
        for sentido in (1, -1):
            try:
                Mn = max(Mn, abs(eje_neutro(barras, e.P, direccion(sentido)).Mx))
            except ValueError:
                continue
    v_flexion = max((Mn * abs(e.V) / abs(e.M) for e in sismo if e.M != 0.0), default=0.0)
    Vn = resistencia_en_el_plano(armado)["Vn"]
    phi = PHI_CORTE_REDUCIDO if Vn < v_flexion else PHI_CORTE
    return {"phi": phi, "aplica": True, "omega_v": amp["omega_v"], "Mn": Mn,
            "V_flexion": v_flexion, "Vn": Vn}


def corte_en_el_plano(armado: Armado, e: Estado, phi: float = PHI_CORTE) -> dict:
    """D/C = |Ve|/(φ·Vn), con el φ de `phi_de_corte`."""
    r = resistencia_en_el_plano(armado)
    r.update(phi=phi, phiVn=phi * r["Vn"], dc=abs(e.Ve) / (phi * r["Vn"]))
    return r


def refuerzo_de_corte_requerido(armado: Armado, estados: list[Estado], phi: float) -> dict:
    """Lo que el capítulo 11 decide con el cortante (§18.2.1.2: los muros
    especiales cumplen también los demás capítulos):

      ¿pide refuerzo de corte?  |Ve| > φ·Vc  → §11.7.2.1 y §11.7.3.1 limitan
          la separación de las barras verticales a ℓw/3 y la de las
          horizontales a ℓw/5;
      §11.6.2: si |Ve| > 0.5·φ·αc·λ·√f'c·Acv, ρℓ ≥ 0.0025 + 0.5·(2.5 −
          hw/ℓw)·(ρt − 0.0025) (y ≥ 0.0025), sin pasar del ρt que pide la
          resistencia, y ρt ≥ 0.0025.

    Con el mayor |Ve| de la cota: los artículos dicen Vu, y en el muro
    especial el cortante de diseño es Ve (del lado seguro).
    """
    r = resistencia_en_el_plano(armado)
    sec = seccion()
    ve = max(abs(e.Ve) for e in estados)
    phi_vc = phi * r["Vc"]
    rho_t = armado.malla_horizontal.cuantia
    rho_t_resistencia = max(0.0, (ve / phi - r["Vc"]) / (D.FY * MPA * sec.acv))
    ecuacion = 0.0025 + 0.5 * (2.5 - ESBELTEZ) * (rho_t - 0.0025)
    return {"Ve": ve, "phi": phi, "Vc": r["Vc"], "phiVc": phi_vc, "requerido": ve > phi_vc,
            "umbral_116": 0.5 * phi_vc, "sobre_116": ve > 0.5 * phi_vc,
            "rho_t_resistencia": rho_t_resistencia, "ecuacion_1162": ecuacion,
            "rho_l_1162": max(0.0025, min(ecuacion, max(rho_t_resistencia, 0.0025)))}


# =============================================================================
# 7. ELEMENTO DE BORDE ESPECIAL (§18.10.6.2, vía por desplazamiento)
# =============================================================================
#
# Independiente del confinamiento de flexocompresión (los elementos de los
# extremos van siempre, de 0.15·ℓw o más en la base): aquí se decide si la
# norma EXIGE elemento de borde especial y, si lo exige, si los elementos
# declarados lo cubren.
#
# Solo en la dirección larga (X, el plano del alma), en sus dos extremos. En Y
# el muro no trabaja como muro: en el tipo I es su espesor, y en los tipos II y
# III las alas cortas son columnas de los pórticos especiales de esa dirección
# (§18.10.6 es de muros; las columnas son del §18.7).


@dataclass(frozen=True)
class Extremo:
    nombre: str              # "izquierdo" o "derecho"
    n: tuple                 # dirección del eje neutro que lo comprime
    elementos: tuple         # los elementos que forman su borde confinado
    provista: float          # longitud confinada desde la fibra extrema


def extremos(armado: Armado | None = None) -> list[Extremo]:
    """Los dos extremos del alma. En el tipo III el derecho es el del ala: su
    borde son el elemento derecho y el ala. La longitud confinada es la del
    armado (por defecto, la de la base: el reducido puede tener otra)."""
    sec = seccion()
    c = cajas() if armado is None else {e.nombre: e.caja for e in armado.elementos}
    derecho = ("derecho", "ala") if sec.tipo == "III" else ("derecho",)
    return [Extremo("izquierdo", (-1.0, 0.0), ("izquierdo",), c["izquierdo"].ancho),
            Extremo("derecho", (1.0, 0.0), derecho, c["derecho"].ancho)]


def longitud_de_borde(c: float) -> float:
    """§18.10.6.4(a): la mayor de c − 0.1·ℓw y c/2."""
    return max(c - 0.1 * D.LW, c / 2)


def necesidad_de_borde(barras: Barras, estados: list[Estado], deriva: float,
                       armado: Armado | None = None) -> list[dict]:
    """¿Hace falta elemento de borde especial? Un caso por extremo.

    VÍA POR DESPLAZAMIENTO (§18.10.6.2), la que rige si hwcs/ℓw ≥ 2: se exige
    si  1.5·δu/hwcs ≥ ℓw/(600·c),  con δu/hwcs no menor que 0.005; es decir,
    si c ≥ c_lim = ℓw/(600·1.5·δu/hwcs). Con c el MAYOR de las combinaciones
    del sismo en el plano, al axial mayorado y Mn, comprimiendo ese extremo.

    Si hwcs/ℓw < 2 no aplica y rige la VÍA POR ESFUERZO (§18.10.6.3): se exige
    si la compresión elástica de la fibra extrema (sección bruta) llega a 0.2·f'c.

    La región crítica sube la mayor de ℓw y Mu/4Vu (§18.10.6.2(b)(i)), con la
    combinación de mayor |Mu| y su Vu mayorado SIN amplificar.

    En otras cotas se repite con el axial y el momento de esa cota (para el
    ancho y la esbeltez); hwcs sigue siendo la altura sobre la base, porque la
    sección crítica es una sola, en la base.
    """
    sec = seccion()
    lw = D.LW
    hwcs = H                     # altura sobre la sección crítica, que está en la base
    propios = [e for e in estados if e.combo in D.COMBINACIONES_DEL_SISMO_EN_X]
    gobierna = max(propios, key=lambda e: abs(e.M))
    altura = max(lw, abs(gobierna.M) / (4.0 * abs(gobierna.V))) if gobierna.V else lw
    der = max(deriva, 0.005)
    por_desplazamiento = hwcs / lw >= 2.0
    casos = []
    for ext in extremos(armado):
        n = np.array(ext.n)
        por_estado = []                 # c solo depende de P: uno por axial distinto
        vistos = set()
        for e in propios:
            if round(e.P, 6) in vistos:
                continue
            vistos.add(round(e.P, 6))
            try:
                por_estado.append((e, eje_neutro(barras, e.P, n).c))
            except ValueError:
                continue                # el axial no cabe en la sección: lo marca la flexión
        if por_estado:
            e, c = max(por_estado, key=lambda t: t[1])
        else:
            e, c = gobierna, sec.canto(n)     # toda la sección comprimida
        if por_desplazamiento:
            indice = 1.5 * der / (lw / (600.0 * c))
        else:
            indice = esfuerzo_en_extremo(ext, estados) / (0.2 * D.FC * MPA)
        exige = indice >= 1.0
        # §18.10.6.4(d) (con ala, el borde la incluye y entra 300 mm en el alma)
        # no se comprueba: en el tipo III el extremo con ala es una columna del
        # pórtico (criterio del proyecto); queda como nota del plano
        # (`detalles_constructivos`).
        requerida = longitud_de_borde(c)
        Ac, _, _ = zona_comprimida(n, c)
        b_medio = Ac / c if c > 0 else sec.b
        caso = {"extremo": ext, "zona": ext.nombre, "estado": e, "c": c,
                "por_estado": por_estado, "lw": lw,
                "c_lim": lw / (600.0 * 1.5 * der), "indice": indice, "exige": exige,
                "via": "18.10.6.2" if por_desplazamiento else "18.10.6.3",
                "hwcs": hwcs, "altura": altura, "deriva": der, "gobierna": gobierna,
                "longitud": requerida if exige else 0.0, "requerida": requerida,
                "b_medio": b_medio,
                # §18.10.6.4(b),(c): el ancho de la zona comprimida en la
                # longitud de (a), con el ala si la hay (el menor)
                "ancho_zona": float(sec.anchos_a(n, np.linspace(0.0, max(requerida, 1e-6), 201)).min())}
        caso.update(esbeltez_de_la_zona_comprimida(c, b_medio, propios, der))
        casos.append(caso)
    return casos


def esbeltez_de_la_zona_comprimida(c: float, b: float, estados: list[Estado], deriva: float) -> dict:
    """§18.10.6.2(b): (ii) b ≥ √(c·ℓw/40), o si no, (iii) capacidad de deriva

        δc/hwcs = (1/100)·[4 − (1/50)·(ℓw/b)·(c/b) − Ve/(0.66·√f'c·Acv)] ≥ 0.015

    que debe llegar a 1.5·δu/hwcs. b: el ancho medio de la zona comprimida en
    la profundidad c (R18.10.6.2: con ala, un promedio ponderado). Ve: el mayor
    |Ve| de las combinaciones del sismo en el plano (del lado seguro).
    Devuelve cuánto se incumple (0 si cumple).
    """
    sec = seccion()
    lw = D.LW
    b_ii = math.sqrt(c * lw / 40.0)
    capacidad, incumple, ve = None, 0.0, 0.0
    if b < b_ii:
        ve = max(abs(e.Ve) for e in estados)
        cap = (4.0 - (lw / b) * (c / b) / 50.0 - ve / (0.66 * RAIZ_FC * MPA * sec.acv)) / 100.0
        capacidad = max(cap, 0.015)
        demanda = 1.5 * deriva
        incumple = demanda / capacidad if demanda > capacidad else 0.0
    return {"b_ii": b_ii, "capacidad_deriva": capacidad, "incumple_esbeltez": incumple,
            "Ve_iii": ve}


def dimensiones_de_borde(caso: dict, piso: int, provista: float | None = None) -> list[dict]:
    """Las dimensiones del elemento de borde especial de un extremo, con el
    caso de la sección crítica (c, la longitud y el ancho no cambian en la
    región crítica) y la altura del entrepiso `piso` (hu):

      §18.10.6.4(a): longitud ≥ máx(c − 0.1·ℓw, c/2);
      §18.10.6.4(b): ancho b de la zona comprimida en esa longitud, con el ala
                     si la hay, ≥ hu/16 (hu: la altura del entrepiso, del lado
                     seguro frente a la libre);
      §18.10.6.4(c): si hw/ℓw ≥ 2 (muro continuo con una sola sección crítica)
                     y c/ℓw ≥ 3/8, b ≥ 300 mm;
      §18.10.6.2(b): (ii) b ≥ √(c·ℓw/40) o, si no, (iii) δc/hwcs ≥ 1.5·δu/hwcs.

    Cada fila: art, texto, valor, limite, tipo, cumple, aplica y en_faltas (la
    longitud ya entra en el D/C de borde y la (ii) que no se cumple la cubre
    la (iii): esas no van a las faltas). provista: la longitud confinada del
    armado de la cota (por defecto, la del caso).
    """
    ext, c, lw = caso["extremo"], caso["c"], caso["lw"]
    if provista is None:
        provista = ext.provista
    b = caso["ancho_zona"]
    filas = []

    def fila(art, texto, valor, limite, cumple, aplica=True, en_faltas=True, tipo="≥", nota=""):
        filas.append({"art": art, "texto": texto, "valor": valor, "limite": limite, "tipo": tipo,
                      "cumple": cumple, "aplica": aplica, "en_faltas": en_faltas, "nota": nota})

    req = longitud_de_borde(c)
    fila("18.10.6.4(a)", f"longitud del borde especial, extremo {ext.nombre} (m)", provista,
         req, provista >= req * (1 - 1e-9), en_faltas=False,
         nota=f"máx(c − 0.1·ℓw, c/2) = máx({c:.3f} − {0.1 * lw:.3f}, {c / 2:.3f})")
    hu = D.ALTURAS[piso]
    fila("18.10.6.4(b)", f"ancho de la zona comprimida ≥ hu/16, extremo {ext.nombre} (m)", b,
         hu / 16.0, b >= hu / 16.0 * (1 - 1e-9),
         nota=f"hu = {hu:.2f} m (entrepiso {D.PISOS[piso]}) → hu/16 = {hu / 16:.3f} m")
    aplica_c = ESBELTEZ >= 2.0 and c / lw >= 0.375
    fila("18.10.6.4(c)", f"ancho de la zona comprimida ≥ 300 mm, extremo {ext.nombre} (m)", b, 0.300,
         (not aplica_c) or b >= 0.300 * (1 - 1e-9), aplica=aplica_c,
         nota=f"aplica si hw/ℓw ≥ 2 y c/ℓw ≥ 3/8: hw/ℓw = {ESBELTEZ:.2f} "
              f"{'≥' if ESBELTEZ >= 2.0 else '<'} 2 y c/ℓw = {c / lw:.3f} "
              f"{'≥' if c / lw >= 0.375 else '<'} 0.375 → {'APLICA' if aplica_c else 'no aplica'}")
    if caso["via"] == "18.10.6.2":
        ok_ii = caso["b_medio"] >= caso["b_ii"] * (1 - 1e-9)
        fila("18.10.6.2(b)(ii)", f"b medio en c ≥ √(c·ℓw/40), extremo {ext.nombre} (m)",
             caso["b_medio"], caso["b_ii"], ok_ii, en_faltas=False,
             nota=f"√({c:.3f}·{lw:.3f}/40) = {caso['b_ii']:.3f} m")
        if not ok_ii:
            cap = caso["capacidad_deriva"]
            fila("18.10.6.2(b)(iii)", f"δc/hwcs ≥ 1.5·δu/hwcs, extremo {ext.nombre}", cap,
                 1.5 * caso["deriva"], cap >= 1.5 * caso["deriva"] * (1 - 1e-9),
                 nota=f"δc/hwcs = [4 − (ℓw/b)(c/b)/50 − Ve/(0.66·√f'c·Acv)]/100 ≥ 0.015, "
                      f"con Ve = {caso['Ve_iii']:.1f} kN")
    return filas


def esfuerzo_en_extremo(ext: Extremo, estados: list[Estado]) -> float:
    """§18.10.6.3: la mayor compresión elástica (kN/m², positiva en compresión)
    de la fibra extrema de ese extremo, con la sección bruta y TODAS las
    combinaciones con sismo («load combinations including earthquake effects
    E»); sin ellas, todas."""
    sec = seccion()
    fibra = 0.0 if ext.n[0] < 0 else sec.lw          # x de la fibra extrema
    con_sismo = [e for e in estados if e.con_sismo] or estados
    return max(max(0.0, -(e.P / sec.area + e.M * (fibra - sec.xc) / sec.Ix)) for e in con_sismo)


def esfuerzo_de_fibra_extrema(e: Estado) -> float:
    """§18.10.6.3 (vía de esfuerzo, solo informativa en el plano): compresión
    máxima con la sección bruta, σ = −P/A − M·(x − xc)/I, positiva en compresión."""
    sec = seccion()
    return max(0.0, max(-(e.P / sec.area + e.M * (x - sec.xc) / sec.Ix) for x in (0.0, sec.lw)))


# =============================================================================
# 8. DETALLADO (cuantías, separaciones, estribos, acero de extremo)
# =============================================================================


def cuantia_de_extremo(barras: Barras, lado: str) -> float:
    """§18.10.2.4(a): cuantía dentro de 0.15·ℓw del extremo «y en un ancho igual
    al espesor del muro». Se suman las barras (de elemento o de malla) cuyo eje
    cae en esa franja y dentro del espesor del alma, y se dividen entre
    0.15·ℓw·b."""
    largo = 0.15 * D.LW
    distancia = barras.x if lado == "izquierdo" else D.LW - barras.x
    dentro = (distancia <= largo + 1e-9) & (np.abs(barras.y) <= D.ESPESOR / 2 + 1e-9)
    return float(barras.A[dentro].sum()) / (largo * D.ESPESOR)


def cuantia_exigida_de_extremo() -> float:
    return 0.5 * math.sqrt(D.FC) / D.FY      # §18.10.2.4(a)


def cuantia_de_borde(armado: Armado, barras: Barras, ext: Extremo) -> float:
    """§18.10.6.5(b), fig. R18.10.6.5a: el acero del borde entre el área del
    borde (la unión de las cajas de sus elementos)."""
    elems = [armado.elemento(n) for n in ext.elementos]
    elems = [e for e in elems if e is not None]
    if not elems:
        return 0.0
    As = float(sum(barras.A[barras.zona == e.nombre].sum() for e in elems))
    area_ = sum(e.caja.area for e in elems)
    if len(elems) == 2:
        area_ -= elems[0].caja.interseccion(elems[1].caja)
    return As / area_


def longitud_con_gancho(barra: str) -> float:
    """§25.4.3.1: ℓdh de una barra en tracción con gancho estándar, la mayor de
    fy·ψe·ψs·ψcc·ψr·db/(21·λ·√f'c), 8·db y 150 mm. ψe = 1 (sin epóxico),
    ψcc = ψr = 1 (§25.4.3.2 lo permite; criterio del proyecto) y ψs de la
    tabla 25.4.3.2 (1.15 para las barras No. 32 y No. 36)."""
    db = diametro(barra)
    psi_s = 1.0 if db < 0.030 else 1.15
    return max(D.FY * psi_s * db / (21 * LAMBDA * RAIZ_FC), 8 * db, 0.150)


def largo_para_anclar(armado: Armado, ext: Extremo) -> float:
    """El largo que tiene la malla horizontal dentro del núcleo confinado de un
    extremo: desde la cara del núcleo hacia el alma (el estribo llega a la
    longitud confinada) hasta el recubrimiento del extremo del muro. En el tipo
    III con el derecho absorbido, el largo se cuenta desde la cara del ala
    hacia el alma (criterio del proyecto)."""
    der = armado.elemento("derecho")
    if ext.nombre == "derecho" and der is not None and absorbido(der, armado):
        return seccion().ancho_ala - D.RECUBRIMIENTO
    return ext.provista - D.RECUBRIMIENTO


def caras_del_estribo(e: Elemento, armado: Armado | None) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones de las barras a lo largo de las caras del estribo: las
    columnas (en X) y las filas (en Y). En el tipo III, el estribo del elemento
    derecho abraza también las barras del ala que caen dentro de su caja (la
    intersección, como en la figura del tipo III): sus columnas son las suyas
    fuera del ala más las del ala."""
    xs, ys = e.columnas(), e.filas()
    ala = armado.elemento("ala") if armado is not None else None
    if seccion().tipo == "III" and e.nombre == "derecho" and ala is not None:
        propias = [float(x) for x in xs if x < ala.caja.x0 - 1e-9]
        xs = np.array(sorted(propias + [float(x) for x in ala.columnas()]))
    return xs, ys


def apoyo_lateral(e: Elemento, hx_limite: float, armado: Armado | None = None) -> dict:
    """Qué barras de cada cara llevan apoyo lateral (esquina de estribo o grapa).

    §25.7.2.3 (por §18.7.5.2(d)): todas las esquinas y una sí y otra no, y
    ninguna barra suelta a más de 150 mm libres, a cada lado, de una apoyada
    (con separaciones desiguales manda el lado más largo). Si con eso la
    separación entre barras apoyadas pasa de hx_limite (§18.10.6.4(f) o
    §18.7.5.2(e)), se apoyan también las de en medio. Con grapas = "todas",
    todas las barras.

    Devuelve las ramas que cruzan el núcleo en cada dirección (las que cuentan
    para Ash) y hx, la mayor separación entre barras apoyadas.
    """
    db = diametro(e.barra)
    columnas, filas = caras_del_estribo(e, armado)

    def cara(pos: np.ndarray) -> tuple[int, float]:
        k = len(pos)
        if e.grapas == "todas":
            apoyadas = set(range(k))
        else:
            apoyadas = {0, k - 1} | set(range(0, k, 2))
            for i in range(k):
                if i not in apoyadas:
                    libre = max(pos[i] - pos[i - 1], pos[i + 1] - pos[i]) - db
                    if libre > 0.150 + 1e-9:
                        apoyadas.add(i)
            while True:
                orden = sorted(apoyadas)
                largos = [(orden[j], orden[j + 1]) for j in range(len(orden) - 1)
                          if pos[orden[j + 1]] - pos[orden[j]] > hx_limite + 1e-9
                          and orden[j + 1] - orden[j] > 1]
                if not largos:
                    break
                for a, b in largos:
                    apoyadas.update(range(a + 1, b))
        orden = sorted(apoyadas)
        hx = max(pos[orden[j + 1]] - pos[orden[j]] for j in range(len(orden) - 1))
        return len(apoyadas), float(hx)

    ramas_y, hx_x = cara(columnas)           # ramas paralelas a Y, una por columna apoyada
    ramas_x, hx_y = cara(filas)              # ramas paralelas a X, una por fila apoyada
    return {"ramas_y": ramas_y, "ramas_x": ramas_x, "hx": max(hx_x, hx_y)}


def limite_hx(e: Elemento, exige: bool) -> tuple[float, str]:
    """hx máximo: con elemento de borde especial, mín(350 mm, 2b/3), con b el
    espesor del muro en ese elemento (§18.10.6.4(f)); si no, 350 mm
    (§18.7.5.2(e))."""
    if exige:
        return min(0.350, 2.0 / 3.0 * e.espesor), "18.10.6.4(f)"
    return 0.350, "18.7.5.2(e)"


def diametros_abrazados(e: Elemento, armado: Armado | None) -> tuple[float, float]:
    """(db, db_max) para los estribos del elemento.

    db, el que dan los límites en db: la barra longitudinal principal más
    pequeña DEL BORDE (tabla 18.10.6.5(b), nota [1]; §18.7.5.3(b)): la menor de
    los elementos de su extremo que tienen barras (en el tipo III, el derecho y
    el ala). En el ala central del tipo II, que no es borde, la suya.
    db_max, el que da el diámetro mínimo del estribo (§25.7.2.2): la mayor que
    abraza su estribo (en el tipo III el del derecho abraza también barras del
    ala, `caras_del_estribo`)."""
    abraza = [diametro(e.barra)]
    ala = armado.elemento("ala") if armado is not None else None
    if seccion().tipo == "III" and e.nombre == "derecho" and ala is not None:
        abraza.append(diametro(ala.barra))
    del_borde = [diametro(e.barra)]
    if armado is not None:
        for ext in extremos():
            if e.nombre not in ext.elementos:
                continue
            for nombre in ext.elementos:
                otro = armado.elemento(nombre)
                if otro is not None and not absorbido(otro, armado):
                    del_borde.append(diametro(otro.barra))
    return min(del_borde), max(abraza)


def limites_de_separacion(e: Elemento, exige: bool, cerca: bool, hx: float,
                          db: float) -> tuple[list[dict], str]:
    """Las expresiones que limitan la separación vertical de los estribos de
    un elemento (manda la menor), cada una {"expr", "valor" (m), "art"}, y el
    artículo que manda.

    Sin elemento de borde especial (§18.10.6.5(b), tabla 18.10.6.5(b), grado
    420): dentro de máx(ℓw, Mu/4Vu) sobre la sección crítica, la menor de 6·db
    y 150 mm; en otras zonas, la menor de 8·db y 200 mm. Con elemento de borde
    especial (§18.10.6.4(e)): §18.7.5.3 con un tercio de la dimensión menor
    del elemento en lugar de un cuarto, 6·db y so = 100 + (350 − hx)/3 mm
    (entre 100 y 150 mm), y tampoco más que la tabla 18.10.6.5(b). db: la
    barra vertical principal más pequeña del borde (`diametros_abrazados`).
    6·db y 8·db se llevan a la separación de replanteo más cercana, por debajo
    o por encima (criterio del proyecto, `replanteo_cercano`).
    """
    def por_db(k):
        r = replanteo_cercano(k * db)
        return f"{k}·db = {k * db * 100:.1f} cm → {r * 100:.0f} cm (replanteo, el más cercano)", r

    zona = "región crítica" if cerca else "otras zonas"
    art_tabla = f"tabla 18.10.6.5(b), {zona}"
    k, tope = (6, 0.150) if cerca else (8, 0.200)
    expr, valor = por_db(k)
    tabla = [{"expr": expr, "valor": valor, "art": art_tabla},
             {"expr": f"{tope * 100:.1f} cm", "valor": tope, "art": art_tabla}]
    if not exige:
        return tabla, "tabla 18.10.6.5(b)"
    crudo = 100 + (350 - hx * 1000) / 3
    so = min(max(crudo / 1000, 0.100), 0.150)
    menor = min(e.caja.ancho, e.caja.alto)
    propios = [{"expr": f"dimensión menor/3 = {menor * 100:.1f}/3 = {menor / 3 * 100:.1f} cm",
                "valor": menor / 3.0, "art": "18.10.6.4(e)"},
               {"expr": por_db(6)[0], "valor": por_db(6)[1], "art": "18.7.5.3(b)"},
               {"expr": f"so = 100 + (350 − {hx * 1000:.0f})/3 = {crudo:.0f} mm → {so * 100:.1f} cm "
                        f"(entre 10 y 15 cm)", "valor": so, "art": "18.7.5.3(d)"}]
    return propios + tabla, "18.10.6.4(e)"


def cuantia_de_ash(e: Elemento) -> dict:
    """Tabla 18.10.6.4(g): Ash/(s·bc) ≥ la mayor de (a) 0.3·(Ag/Ach − 1)·f'c/fyt
    y (b) 0.09·f'c/fyt. Ag es la caja del elemento y Ach el núcleo medido a la
    cara exterior del estribo (`Elemento.nucleo`); fyt = fy (estribos del mismo
    acero, ≤ 420 MPa)."""
    bcx, bcy = e.nucleo
    Ag, Ach = e.caja.area, bcx * bcy
    a = 0.3 * (Ag / Ach - 1.0) * D.FC / D.FY
    b = 0.09 * D.FC / D.FY
    return {"bcx": bcx, "bcy": bcy, "Ag": Ag, "Ach": Ach, "a": a, "b": b, "exigida": max(a, b)}


def _ash(e: Elemento, apoyo: dict) -> list[dict]:
    """Ash provisto frente al exigido (`cuantia_de_ash`), en las dos
    direcciones del núcleo: las ramas paralelas a Y (una por columna apoyada)
    cruzan bc en X, y las paralelas a X, bc en Y. Ramas: las de `apoyo_lateral`."""
    q = cuantia_de_ash(e)
    if q["bcx"] <= 0 or q["bcy"] <= 0:
        return []
    filas = []
    for ramas, bc, donde in ((apoyo["ramas_y"], q["bcx"], "en X"),
                             (apoyo["ramas_x"], q["bcy"], "en Y")):
        provisto = ramas * area(e.estribo)
        necesario = q["exigida"] * e.separacion_estribo * bc
        filas.append({"art": "18.10.6.4(g)",
                      "texto": f"Ash con bc {donde} ({ramas} ramas), elemento {e.nombre} (m²)",
                      "valor": provisto, "limite": necesario, "tipo": "≥",
                      "cumple": provisto >= necesario * (1 - 1e-9),
                      "ramas": ramas, "bc": bc, "donde": donde})
    return filas


def regla_de_estribos(exige: bool, rho: float) -> tuple[str, str]:
    """Qué pide la norma a los estribos de un borde: «especial» (§18.10.6.4),
    «ordinario» (§18.10.6.5(b): ρ del borde > 2.8/fy) o «proyecto» (la norma
    no los pide; se confina igual, con la tabla 18.10.6.5(b) como límite)."""
    if exige:
        return "especial", "18.10.6.4(e)"
    if rho > 2.8 / D.FY:
        return "ordinario", "18.10.6.5(b)"
    return "proyecto", "proyecto"


def estribos(e: Elemento, armado: Armado, exige: bool, cerca: bool, rho: float,
             borde: bool = True) -> dict:
    """El diseño de los estribos de un elemento en una zona: la regla que
    manda, el apoyo lateral y hx, la separación máxima con cada expresión, el
    diámetro mínimo y, con borde especial, Ash. exige: borde especial en esta
    zona; cerca: dentro de máx(ℓw, Mu/4Vu) sobre la sección crítica; rho: la
    cuantía del borde (§18.10.6.5(b)); borde: False para el ala central del
    tipo II, que no es borde del muro en X (se confina por criterio del
    proyecto, con los límites del borde ordinario en «otras zonas»)."""
    regla, art = regla_de_estribos(exige, rho) if borde else ("ala central", "proyecto")
    db, db_max = diametros_abrazados(e, armado)
    hx_lim, art_hx = limite_hx(e, exige)
    apoyo = apoyo_lateral(e, hx_lim, armado)
    limites, art_s = limites_de_separacion(e, exige, cerca, apoyo["hx"], db)
    salida = {"regla": regla, "art": art, "db": db, "db_max": db_max, "hx_lim": hx_lim,
              "art_hx": art_hx, "apoyo": apoyo, "limites": limites, "art_s": art_s,
              "s_max": min(x["valor"] for x in limites),
              "estribo_min": ESTRIBO_No10 if db_max <= DIAMETRO_No32 else ESTRIBO_No13,
              "ash": [], "cuantia_ash": None}
    if exige and e.estribo is not None:
        salida["cuantia_ash"] = cuantia_de_ash(e)
        salida["ash"] = _ash(e, apoyo)
    return salida


def separacion_real_vertical(barras: Barras) -> float:
    """La mayor separación entre líneas de barras verticales a lo largo del
    alma (malla y elementos, las que caen dentro del espesor del alma)."""
    en_alma = np.abs(barras.y) <= seccion().b / 2 + 1e-9
    lineas = np.unique(np.round(barras.x[en_alma], 9))
    return float(np.diff(lineas).max()) if len(lineas) > 1 else 0.0


def elementos_de(ext_nombres: set) -> set:
    """Los elementos que forman parte de los extremos dados."""
    return {n for ext in extremos() if ext.nombre in ext_nombres for n in ext.elementos}


def detallado(armado: Armado, barras: Barras, Vu: float, piso: int, exigidas: set,
              cercanas: set, con_extremo: bool = True, casos: list | None = None,
              cortante: dict | None = None) -> list[dict]:
    """Todas las comprobaciones de armado de una cota. Cada una devuelve
    artículo, qué se comprueba, valor, límite y si cumple. Las separaciones
    y la cuantía de extremo se miden sobre las barras colocadas.

    exigidas, cercanas: los extremos que exigen elemento de borde especial en
    esta cota, y los que están dentro de su región crítica. con_extremo: si la
    cota está dentro de la altura de §18.10.2.4(b), donde rige (a). casos: los
    de `necesidad_de_borde` en esta cota (su c da la longitud de
    §18.10.6.5(b)). cortante: el de `refuerzo_de_corte_requerido` (si el
    cortante pide refuerzo, §11.7.2.1 y §11.7.3.1 limitan las separaciones a
    ℓw/3 y ℓw/5)."""
    sec = seccion()
    filas = []

    def minimo(art, texto, valor, limite):
        filas.append({"art": art, "texto": texto, "valor": valor, "limite": limite,
                      "tipo": "≥", "cumple": limite <= 0 or valor >= limite * (1 - 1e-9)})

    def maximo(art, texto, valor, limite):
        filas.append({"art": art, "texto": texto, "valor": valor, "limite": limite,
                      "tipo": "≤", "cumple": limite <= 0 or valor <= limite * (1 + 1e-9)})

    mv, mh = armado.malla_vertical, armado.malla_horizontal
    # --- la malla del alma ---------------------------------------------------
    # §18.10.2.1: 0.0025 en las dos direcciones. Si Vu ≤ 0.083·λ·√f'c·Acv, la
    # HORIZONTAL (ρt) puede bajar a la tabla 11.6.1; la vertical no baja nunca.
    rho_h_min = 0.0025
    if abs(Vu) <= 0.083 * LAMBDA * RAIZ_FC * MPA * sec.acv:
        grande = D.BARRAS[mh.barra][0] > 16.0
        # tabla 11.6.1: 0.0020 solo con barras ≤ No. 16 y fy ≥ 420 MPa (al pie de la
        # letra: con fy = 4200 kgf/cm² = 411.88 MPa rige 0.0025)
        rho_h_min = 0.0025 if (grande or D.FY < 420.0) else 0.0020
        # §11.6.2(b): si el cortante pasa de 0.5·φ·Vc, ρt ≥ 0.0025 igual
        if cortante is not None and cortante["sobre_116"]:
            rho_h_min = 0.0025
    minimo("18.10.2.1", "cuantía vertical de la malla ρℓ", mv.cuantia, 0.0025)
    minimo("18.10.2.1", "cuantía horizontal de la malla ρt", mh.cuantia, rho_h_min)
    # §11.6.2(a), si el cortante pasa de 0.5·φ·Vc; con hw/ℓw ≥ 2.5 da 0.0025 y
    # repite la de §18.10.2.1. (§11.6.2(b), ρt ≥ 0.0025, ya está en §18.10.2.1.)
    if cortante is not None and cortante["sobre_116"] and ESBELTEZ < 2.5:
        minimo("11.6.2(a)", "cuantía vertical ρℓ (ec. 11.6.2)", mv.cuantia, cortante["rho_l_1162"])
    maximo("11.7.5.1", "ρℓ sin estribos de apoyo", mv.cuantia, 0.01)
    # separaciones: 450 mm (§18.10.2.1); 3h (§11.7.2.1 y §11.7.3.1); si el
    # cortante pide refuerzo, ℓw/3 la vertical y ℓw/5 la horizontal
    pide = cortante is not None and cortante["requerido"]
    s_v = min(0.45, 3 * D.ESPESOR, D.LW / 3 if pide else math.inf)
    s_h = min(0.45, 3 * D.ESPESOR, D.LW / 5 if pide else math.inf)
    art_v = "18.10.2.1" if s_v >= 0.45 - 1e-12 else "11.7.2.1"
    art_h = "18.10.2.1" if s_h >= 0.45 - 1e-12 else "11.7.3.1"
    maximo(art_v, "separación vertical (m)", mv.separacion, s_v)
    maximo(art_h, "separación horizontal (m)", mh.separacion, s_h)
    maximo(art_v, "separación real máxima entre barras verticales (m)",
           separacion_real_vertical(barras), s_v)
    # §18.10.2.2: dos cortinas si Vu > 0.17·λ·√f'c·Acv o hw/ℓw ≥ 2;
    # §11.7.2.3: también en todo muro de más de 250 mm de espesor
    if abs(Vu) > 0.17 * LAMBDA * RAIZ_FC * MPA * sec.acv or ESBELTEZ >= 2.0:
        minimo("18.10.2.2", "cortinas de la malla vertical", mv.cortinas, 2)
        minimo("18.10.2.2", "cortinas de la malla horizontal", mh.cortinas, 2)
    if D.ESPESOR > 0.250 + 1e-9:
        minimo("11.7.2.3", "cortinas de la malla vertical (muro de más de 250 mm)", mv.cortinas, 2)
        minimo("11.7.2.3", "cortinas de la malla horizontal (muro de más de 250 mm)", mh.cortinas, 2)
    # espesor ≥ máx(100 mm, menor de ℓw y altura / 25): la tabla 11.3.1.1 solo
    # obliga con el método simplificado de §11.5.3 (su nota [1]); aquí, que
    # se diseña con §22.4, es un criterio del proyecto
    minimo("proyecto", "espesor ≥ máx(100 mm, mín(ℓw, h)/25), como la tabla 11.3.1.1 (m)", D.ESPESOR,
           max(0.100, min(D.ALTURAS[piso], D.LW) / 25.0))
    # §18.10.4.3: en muros robustos (hw/ℓw ≤ 2) ρℓ ≥ ρt
    if ESBELTEZ <= 2.0:
        minimo("18.10.4.3", "ρℓ frente a ρt (muro robusto)", mv.cuantia, mh.cuantia)
    # §18.10.2.4(a): cuantía en los extremos, hasta la altura de (b)
    if ESBELTEZ >= 2.0 and con_extremo:
        for lado in ("izquierdo", "derecho"):
            minimo("18.10.2.4(a)", f"cuantía en 0.15·ℓw, extremo {lado}",
                   cuantia_de_extremo(barras, lado), cuantia_exigida_de_extremo())
    # §18.10.5.2: el ala entera es efectiva si no sale más del 25 % de la altura
    if sec.ala is not None:
        maximo("18.10.5.2", "vuelo del ala fuera del alma frente a 0.25·hw (m)",
               sec.largo_ala - sec.b, 0.25 * (H - sum(D.ALTURAS[:piso])))
        # §18.7.4.1: en Y el ala es columna del pórtico especial; su acero
        # vertical, entre 0.01 y 0.06 de su área b'·l'w, en toda la altura
        rho_ala = float(barras.A[barras.zona == "ala"].sum()) / sec.ala.area
        minimo("18.7.4.1", "cuantía vertical del ala (b'·l'w), columna del pórtico especial", rho_ala, 0.01)
        maximo("18.7.4.1", "cuantía vertical del ala (b'·l'w), columna del pórtico especial", rho_ala, 0.06)

    # --- los elementos confinados --------------------------------------------
    con_borde = elementos_de(exigidas)
    cerca_de = elementos_de(cercanas)
    rho_borde = {ext.nombre: cuantia_de_borde(armado, barras, ext) for ext in extremos(armado)}
    c_de = {caso["zona"]: caso["c"] for caso in (casos or [])}
    for ext in extremos(armado):              # con la longitud confinada de ESTE armado
        # criterio del proyecto: los extremos se confinan en 0.15·ℓw como
        # mínimo, solo en la base (en el reducido, §18.10.6.5(b) de abajo)
        if armado.nombre == "base":
            minimo("proyecto", f"longitud confinada del extremo {ext.nombre} ≥ 0.15·ℓw (m)",
                   ext.provista, 0.15 * D.LW)
        # §18.10.6.5(b): sin borde especial y con ρ > 2.8/fy, los estribos van
        # en la longitud de §18.10.6.4(a), con el c de esta cota (con borde
        # especial, la longitud la comprueba el D/C de borde)
        if ext.nombre not in exigidas and rho_borde[ext.nombre] > 2.8 / D.FY and ext.nombre in c_de:
            minimo("18.10.6.5(b)", f"longitud con estribos del extremo {ext.nombre} ≥ máx(c − 0.1ℓw, "
                   f"c/2) (m)", ext.provista, longitud_de_borde(c_de[ext.nombre]))
        # §18.10.6.4(k): la malla horizontal desarrolla fy con gancho estándar
        # dentro del núcleo confinado; sin borde especial, criterio del proyecto
        minimo("18.10.6.4(k)" if ext.nombre in exigidas else "proyecto",
               f"largo para anclar la malla horizontal con gancho, extremo {ext.nombre} ≥ ℓdh (m)",
               largo_para_anclar(armado, ext), longitud_con_gancho(mh.barra))
    for e in armado.elementos:
        propias = barras.zona == e.nombre
        if not propias.any():
            continue                        # el derecho del tipo III, absorbido por el ala
        exige, cerca = e.nombre in con_borde, e.nombre in cerca_de
        db = diametro(e.barra)
        suyos = [x for x in extremos() if e.nombre in x.elementos]
        rho = max(rho_borde[x.nombre] for x in suyos) if suyos else 0.0
        if e.estribo is None:
            minimo(regla_de_estribos(exige, rho)[1] if suyos else "proyecto",
                   f"estribo en el elemento {e.nombre}", 0, 1)
        else:
            est = estribos(e, armado, exige, cerca, rho, borde=bool(suyos))
            # con la regla «proyecto» (la norma no pide estribos) y en el ala
            # central los límites se exigen igual, por criterio del proyecto
            propio = est["regla"] in ("proyecto", "ala central")
            maximo("proyecto" if propio else est["art_hx"],
                   f"separación entre barras apoyadas hx, elemento {e.nombre} (m)",
                   est["apoyo"]["hx"], est["hx_lim"])
            maximo("proyecto" if propio else est["art_s"],
                   f"separación de estribos, elemento {e.nombre} (m)", e.separacion_estribo,
                   est["s_max"])
            minimo("25.7.2.2", f"diámetro del estribo, elemento {e.nombre} (m)",
                   diametro(e.estribo), est["estribo_min"])
            # §25.7.2.1(a): separación libre entre estribos ≥ (4/3)·dagg
            agregado = (getattr(D, "AGREGADO_MM", None) or 0.0) / 1000.0
            if agregado > 0.0:
                minimo("25.7.2.1(a)", f"separación libre entre estribos ≥ 4/3·dagg, elemento {e.nombre} (m)",
                       e.separacion_estribo - diametro(e.estribo), 4.0 / 3.0 * agregado)
            # §25.7.2.1(b): separación de estribos ≤ la menor de 16·db de la
            # barra vertical, 48·db del estribo y la dimensión menor del
            # elemento (su espesor); casi nunca manda frente a la tabla 18.10.6.5(b)
            maximo("25.7.2.1(b)", f"separación de estribos ≤ mín(16·db, 48·dest, espesor), elemento "
                   f"{e.nombre} (m)", e.separacion_estribo,
                   min(16 * est["db"], 48 * diametro(e.estribo), e.espesor))
            filas.extend(est["ash"])
            filas.append({"art": "proyecto",
                          "texto": f"separación de estribos del elemento {e.nombre}, {REPLANTEO} (m)",
                          "valor": e.separacion_estribo, "limite": "2 o 5 cm", "tipo": "múlt.",
                          "cumple": es_de_replanteo(e.separacion_estribo)})
        # §25.2.3: separación libre entre las barras del elemento
        px, py = barras.x[propias], barras.y[propias]
        dist = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
        np.fill_diagonal(dist, np.inf)
        minimo("25.2.3", f"separación libre entre barras, elemento {e.nombre} (m)",
               float(dist.min()) - db, separacion_libre_exigida(db))
    # §18.10.6.4(d) (sección con ala y borde especial: el borde incluye el ala
    # y entra 300 mm en el alma) y la regla de la esquina de §18.10.6.5(b) (en
    # la intersección alma-ala los estribos entran 300 mm en el alma y en el
    # ala, fig. R18.10.6.5b) no se comprueban aquí: en los tipos II y III el
    # ala es columna del pórtico especial (criterio del proyecto). Quedan como
    # notas para detallar en el plano (`detalles_constructivos`). El resto de
    # §18.10.6.5(b) (longitud y estribos de la tabla 18.10.6.5(b)) sí, arriba.
    # §25.2.3 entre grupos: entre barras de elementos distintos, y entre la
    # malla y los elementos
    grupos = sorted(set(barras.zona.tolist()))
    peor = None
    for i, g1 in enumerate(grupos):
        for g2 in grupos[i + 1:]:
            a, b = barras.zona == g1, barras.zona == g2
            libre = (np.hypot(barras.x[a][:, None] - barras.x[b][None, :],
                              barras.y[a][:, None] - barras.y[b][None, :])
                     - (barras.d[a][:, None] + barras.d[b][None, :]) / 2)
            pedida = np.vectorize(separacion_libre_exigida)(
                np.maximum(barras.d[a][:, None], barras.d[b][None, :]))
            k = np.unravel_index(np.argmin(libre - pedida), libre.shape)
            holgura = float(libre[k] - pedida[k])
            if peor is None or holgura < peor[0]:
                peor = (holgura, float(libre[k]), float(pedida[k]), g1, g2)
    if peor is not None:
        minimo("25.2.3", f"separación libre entre grupos ({peor[3]} y {peor[4]}) (m)",
               peor[1], peor[2])
    # Regla de REPLANTEO del proyecto (no es de la norma): las dos mallas (y
    # los estribos, arriba) cada múltiplo de 2 o 5 cm.
    for nombre, s in (("malla vertical", mv.separacion), ("malla horizontal", mh.separacion)):
        filas.append({"art": "proyecto", "texto": f"separación de la {nombre}, {REPLANTEO} (m)",
                      "valor": s, "limite": "2 o 5 cm", "tipo": "múlt.", "cumple": es_de_replanteo(s)})
    return filas


# =============================================================================
# 9. VERIFICACIÓN COMPLETA EN UNA COTA
# =============================================================================


@dataclass
class Verificacion:
    z: float
    piso: int
    armado: Armado
    estados: list
    dc_flexion: float               # P-Mx (uniaxial): solo de referencia, no entra al veredicto
    estado_flexion: Estado
    dc_biaxial: float               # flexocompresión biaxial: el D/C que se verifica
    estado_biaxial: Estado
    dc_corte: float
    corte: dict
    estado_corte: Estado
    casos_borde: list
    exigidas: set
    cercanas: set
    dc_borde: float
    faltas: list
    detalle: list

    @property
    def peor(self) -> float:
        # la flexocompresión se verifica con el D/C biaxial (criterio del
        # proyecto); el de P-Mx solo se informa
        return max(self.dc_biaxial, self.dc_corte, self.dc_borde)

    @property
    def cumple(self) -> bool:
        return self.peor <= 1.0 and not self.faltas


class Diseno:
    """Reúne la demanda y lo que se decide una vez (la región crítica)."""

    def __init__(self, demanda: Demanda, delta_u: float,
                 resolucion: tuple[int, int] | None = None, estados: dict | None = None) -> None:
        """delta_u: desplazamiento de diseño del techo en el plano del muro, m
        (archivo de datos). La razón δu/hwcs se toma con la altura del muro (la sección
        crítica está en la base). resolucion: la de la superficie biaxial (por
        defecto, la de los datos). estados: un diccionario para compartir la
        demanda de cada cota entre varios Diseno (no depende del armado: el
        optimizador la calcula una vez para todos los candidatos)."""
        if delta_u <= 0.0:
            raise DatoInvalido(f"DELTA_U_MM tiene que ser positivo, es {delta_u * 1000:g}")
        self.demanda = demanda
        self.resolucion = resolucion
        self.delta_u = delta_u
        self.deriva = delta_u / H
        self.region: dict | None = None     # {extremo: (exige, altura del borde especial, longitud)}
        self.casos_region: dict = {}        # {extremo: caso de la sección crítica}
        self.altura_critica = 0.0           # máx(ℓw, Mu/4Vu): fila «región crítica» de la tabla 18.10.6.5(b)
        self.z_extremo: float | None = None  # §18.10.2.4(b): máx(ℓw, Mu/3Vu)
        self.phi_corte: dict | None = None   # §21.2.4.1, ver `phi_de_corte`
        self._barras: dict = {}
        self._diag: dict = {}
        self._sup: dict = {}
        self._estados: dict = {} if estados is None else estados
        self._verif: dict = {}
        self._dcs: dict = {}

    # -- capacidad de un armado, calculada una vez --------------------------
    def barras(self, a: Armado) -> Barras:
        if a not in self._barras:
            self._barras[a] = colocar_barras(a)
        return self._barras[a]

    def diagrama(self, a: Armado) -> Diagrama:
        if a not in self._diag:
            self._diag[a] = diagrama_PM(self.barras(a))
        return self._diag[a]

    def superficie(self, a: Armado) -> SuperficieBiaxial:
        if a not in self._sup:
            self._sup[a] = SuperficieBiaxial(self.barras(a), self.resolucion)
        return self._sup[a]

    def dcs_biaxiales(self, a: Armado, z: float, piso: int) -> list[float]:
        """El D/C biaxial de cada estado de la cota (una vez por armado y cota:
        lo piden la verificación, la cota de corte, el informe y las figuras)."""
        clave = (a, round(z, 9), piso)
        if clave not in self._dcs:
            sup = self.superficie(a)
            self._dcs[clave] = [sup.dc(e.P, e.M, e.My) for e in self.estados(z, piso)]
        return self._dcs[clave]

    def olvidar(self, a: Armado) -> None:
        """Borra lo calculado para un armado (el optimizador prueba miles de
        reducidos sobre el mismo Diseno y no quiere guardarlos todos)."""
        for cache in (self._barras, self._diag, self._sup):
            cache.pop(a, None)
        for cache in (self._verif, self._dcs):
            for clave in [k for k in cache if k[0] == a]:
                del cache[clave]

    # -- demanda --------------------------------------------------------------
    def estados(self, z: float, piso: int) -> list[Estado]:
        clave = (round(z, 9), piso)
        if clave not in self._estados:
            self._estados[clave] = amplificar(self.demanda.en(z, piso))
        return self._estados[clave]

    def altura_del_borde(self, caso: dict) -> float:
        """Hasta dónde sube el borde especial de un extremo que lo exige en la
        base. Vía por desplazamiento: la región crítica, máx(ℓw, Mu/4Vu)
        (§18.10.6.2(b)(i)). Vía por esfuerzo (§18.10.6.3): se puede cortar
        donde la compresión de la fibra extrema baja de 0.15·f'c; se sube por
        las cotas hasta la primera en que pasa (y no menos de la región
        crítica, del lado seguro)."""
        if not caso["exige"] or caso["via"] == "18.10.6.2":
            return caso["altura"]
        for z, s, _ in self.cotas():
            if esfuerzo_en_extremo(caso["extremo"], self.estados(z, s)) < 0.15 * D.FC * MPA:
                return max(z, caso["altura"])
        return H

    def altura_de_cambio(self) -> float:
        """Hasta dónde va la separación de estribos de abajo en el armado de
        base (encima, la «separacion_estribo_arriba» de cada elemento, criterio
        del proyecto): la mayor de la región crítica de la tabla 18.10.6.5(b),
        máx(ℓw, Mu/4Vu), y la altura del borde especial de cada extremo que lo
        exige (§18.10.6.2(b)(i) o §18.10.6.3), al cm hacia arriba y sin pasar
        de H. Antes de verificar la base en z = 0, H (todo con la de abajo)."""
        if self.region is None:
            return H
        alturas = [self.altura_critica] + [alt for ex, alt, _ in self.region.values() if ex]
        return min(math.ceil(max(alturas) * 100 - 1e-6) / 100, H)

    def estribos_en(self, a: Armado, z: float) -> Armado:
        """El armado con la separación de estribos que rige en la cota z: desde
        la altura de cambio, la «de arriba» de cada elemento que la tiene."""
        return a if z < self.altura_de_cambio() - 1e-9 else con_separacion_de_arriba(a)

    def piso_de(self, z: float) -> int:
        return max(s for s in range(len(D.ALTURAS)) if self.demanda.z_base[s] <= z + TOL)

    def cotas(self) -> list[tuple[float, int, str]]:
        """Cotas de cada entrepiso, cada PASO_DE_COTAS como máximo, más los
        planos de nudos del MEF. Las losas salen dos veces (tope y base)."""
        salida = []
        for s in range(len(D.ALTURAS)):
            zb, zt = self.demanda.z_base[s], self.demanda.z_tope[s]
            n = max(1, math.ceil((zt - zb) / D.PASO_DE_COTAS - 1e-9))
            rejilla = {round(zb + (zt - zb) * j / n, 9) for j in range(n + 1)}
            planos = {round(z, 9) for z in self.demanda.planos_interiores(s)}
            for z in sorted(rejilla | planos):
                cara = "base" if abs(z - zb) < 1e-9 else "tope" if abs(z - zt) < 1e-9 else ""
                salida.append((z, s, cara))
        return salida

    # -- verificación ---------------------------------------------------------
    def verificar(self, a: Armado, z: float, piso: int) -> Verificacion:
        """La verificación completa de un armado en una cota (una vez por
        armado y cota: la cota de corte la pide en todas las cotas)."""
        clave = (a, round(z, 9), piso)
        if clave not in self._verif:
            self._verif[clave] = self._verificar(a, z, piso)
        return self._verif[clave]

    def _verificar(self, a: Armado, z: float, piso: int) -> Verificacion:
        b = self.barras(a)
        diag = self.diagrama(a)
        est = self.estados(z, piso)

        flex = [dc_radial(diag, e.P, e.M) for e in est]
        i_f = int(np.argmax(flex))
        bia = self.dcs_biaxiales(a, z, piso)
        i_b = int(np.argmax(bia))
        # φ del cortante (§21.2.4.1): se decide una vez, en la base, con el
        # armado que se verifica primero en z = 0 (el de base).
        if self.phi_corte is None and z <= TOL:
            self.phi_corte = phi_de_corte(b, a, est)
        phi = self.phi_corte["phi"] if self.phi_corte else PHI_CORTE
        cortes = [corte_en_el_plano(a, e, phi) for e in est]
        i_c = int(np.argmax([c["dc"] for c in cortes]))

        # Elemento de borde: se decide en la sección crítica (z = 0), con el
        # armado que se verifica primero en z = 0 (el de base); por encima
        # rige esa decisión hasta la altura de la región crítica.
        casos = necesidad_de_borde(b, est, self.deriva, a)
        if self.region is None and z <= TOL:
            self.region = {c["zona"]: (c["exige"], self.altura_del_borde(c), c["longitud"])
                           for c in casos}
            self.casos_region = {c["zona"]: c for c in casos}
            self.altura_critica = casos[0]["altura"]
        if self.region is None:
            region = {c["zona"]: (c["exige"], c["altura"], c["longitud"]) for c in casos}
            casos_region = {c["zona"]: c for c in casos}
            critica = casos[0]["altura"]
        else:
            region, casos_region, critica = self.region, self.casos_region, self.altura_critica
        exigidas = {zn for zn, (ex, alt, _) in region.items() if ex and z < alt - 1e-9}
        cercanas = {zn for zn in region if z < critica - 1e-9}
        dc_borde = 0.0
        faltas = []
        # la longitud confinada del armado que se verifica (el reducido puede
        # tener otra)
        provista = {ext.nombre: ext.provista for ext in extremos(a)}
        for zn in exigidas:
            dc_borde = max(dc_borde, region[zn][2] / provista[zn])
        # Ancho y esbeltez del elemento de borde (§18.10.6.4(b),(c) y
        # §18.10.6.2(b)): geometría, que el refuerzo no arregla. Con el caso
        # de la sección crítica (su c y su longitud) y la altura de ESTE
        # entrepiso, en los extremos donde rige el borde especial.
        filas_borde = []
        for zn in [x.nombre for x in extremos(a) if x.nombre in exigidas]:
            filas_borde += dimensiones_de_borde(casos_region[zn], piso, provista[zn])
        # Vía por esfuerzo (§18.10.6.3): el borde especial se exige donde σ
        # llega a 0.2·f'c. El programa lo modela desde la base; si por encima
        # de él vuelve a llegar, se avisa como falta.
        for caso in casos:
            zn = caso["zona"]
            if caso["via"] == "18.10.6.3" and zn not in exigidas:
                sigma = esfuerzo_en_extremo(caso["extremo"], est)
                filas_borde.append({
                    "art": "18.10.6.3", "texto": f"compresión de la fibra extrema fuera del borde especial "
                    f"de la base, extremo {zn} (MPa)", "valor": sigma / 1000, "limite": 0.2 * D.FC,
                    "tipo": "<", "cumple": sigma < 0.2 * D.FC * MPA, "aplica": True,
                    "en_faltas": True, "nota": ""})
        faltas += [f for f in filas_borde if f["en_faltas"] and not f["cumple"]]
        # §18.10.2.4(b): el acero de (a) sube máx(ℓw, Mu/3Vu) sobre la sección
        # crítica, con el estado que gobierna la flexocompresión (biaxial) en la
        # base y su Vu mayorado sin amplificar. Se fija en la base, como la
        # región crítica.
        if self.z_extremo is None and z <= TOL:
            g = est[i_b]
            self.z_extremo = max(D.LW, abs(g.M) / (3.0 * abs(g.Vu))) if g.Vu else D.LW
        con_extremo = self.z_extremo is None or z < self.z_extremo - 1e-9
        # §18.10.2.1 y §18.10.2.2 hablan de Vu: el mayorado, sin amplificar
        # (§18.10.6.5(a), en cambio, dice expresamente ωv·Ωv·Vu).
        vu_gob = max(est, key=lambda e: abs(e.V)).V
        cortante = refuerzo_de_corte_requerido(a, est, phi)
        # los estribos, con la separación que rige en esta cota (`estribos_en`)
        detalle = detallado(self.estribos_en(a, z), b, vu_gob, piso, exigidas, cercanas, con_extremo,
                            casos, cortante)
        faltas += [f for f in detalle if not f["cumple"]]
        detalle = filas_borde + detalle
        return Verificacion(
            z, piso, a, est, flex[i_f], est[i_f], bia[i_b], est[i_b],
            cortes[i_c]["dc"], cortes[i_c], est[i_c],
            casos, exigidas, cercanas, dc_borde, faltas, detalle)


def zonas_de_estribos(dis: Diseno, base: Armado, red: Armado | None, corte: float) -> dict:
    """Los estribos de cada elemento por zonas en altura (el diseño que va al
    plano). Las zonas se parten en la altura de la región crítica, máx(ℓw,
    Mu/4Vu) sobre la base (borde especial o, sin él, la fila «región crítica»
    de la tabla 18.10.6.5(b)), y en la cota de corte (encima, el armado
    reducido: otras barras y otra cuantía del borde) y, si algún elemento de
    la base tiene «separacion_estribo_arriba», en la altura de cambio (desde
    ella, esa separación: `Diseno.estribos_en`). Por elemento, una lista de
    {z0, z1, armado, elemento, rho, exige, cerca, est (de `estribos`) y
    cumple}. Los elementos sin barras propias (el derecho del tipo III
    absorbido por el ala) no salen."""
    region = dis.region or {}
    alturas = {min(alt, H) for ex, alt, _ in region.values() if ex} | {min(dis.altura_critica, H)}
    if any(e.separacion_arriba for e in base.elementos):
        alturas.add(dis.altura_de_cambio())
    hay_corte = red is not None and corte < H - TOL
    marcas = sorted({0.0, *alturas, corte if hay_corte else H, H})
    salida = {}
    for nombre in nombres_de_elementos():
        suyos = [x for x in extremos() if nombre in x.elementos]
        filas = []
        for z0, z1 in zip(marcas[:-1], marcas[1:]):
            if z1 - z0 <= TOL:
                continue
            arm = red if (hay_corte and z0 >= corte - TOL) else base
            b = dis.barras(arm)
            e = dis.estribos_en(arm, z0).elemento(nombre)
            if e is None or not (b.zona == nombre).any():
                continue
            if suyos:
                rho = max(cuantia_de_borde(arm, b, x) for x in suyos)
            else:                       # el ala central: la cuantía de su caja, informativa
                rho = float(b.A[b.zona == nombre].sum()) / e.caja.area
            cerca = bool(suyos) and z0 < dis.altura_critica - 1e-9
            exige = any(region[x.nombre][0] and z0 < region[x.nombre][1] - 1e-9
                        for x in suyos if x.nombre in region)
            est = estribos(e, arm, exige, cerca, rho, borde=bool(suyos)) if e.estribo is not None else None
            cumple = est is not None and (
                est["apoyo"]["hx"] <= est["hx_lim"] * (1 + 1e-9)
                and e.separacion_estribo <= est["s_max"] * (1 + 1e-9)
                and diametro(e.estribo) >= est["estribo_min"] * (1 - 1e-9)
                and all(f["cumple"] for f in est["ash"]))
            filas.append({"z0": z0, "z1": z1, "armado": arm, "elemento": e, "rho": rho,
                          "exige": exige, "cerca": cerca, "est": est, "cumple": cumple})
        if filas:
            salida[nombre] = filas
    return salida


def rama_larga_permitida(e: Elemento, v: Verificacion, dis: Diseno) -> str:
    """§18.10.6.4(f): las ramas del estribo pueden pasar de 2·bc si, en todos
    los extremos que exigen borde especial en este elemento, se cumple
      (i)  b ≥ √(ℓw·c/40) y δu/hwcs < 0.012, o
      (ii) hay un ala dentro de c de ancho total ≥ 2·bw y espesor ≥ bw/2.
    Con el caso de la sección crítica de cada extremo (su c y su b).
    Devuelve el texto de la razón, o "" si no se cumple."""
    sec = seccion()
    razones = []
    for ext in extremos():
        if ext.nombre not in v.exigidas or e.nombre not in ext.elementos:
            continue
        caso = dis.casos_region[ext.nombre]
        if caso["b_medio"] >= caso["b_ii"] and dis.deriva < 0.012:
            razones.append(f"(i) en el extremo {ext.nombre}: b = {caso['b_medio']:.3f} ≥ "
                           f"√(ℓw·c/40) = {caso['b_ii']:.3f} m y δu/hwcs = "
                           f"{dis.deriva:.4f} < 0.012")
        elif (sec.tipo == "III" and ext.nombre == "derecho" and sec.largo_ala >= 2 * sec.b
              and sec.ancho_ala >= sec.b / 2):
            razones.append(f"(ii) en el extremo {ext.nombre}: ala de {sec.largo_ala:.2f} m ≥ 2·bw "
                           f"y {sec.ancho_ala:.2f} m ≥ bw/2")
        else:
            return ""
    return "; ".join(razones)


def esquinas_sin_barra(armado: Armado) -> list[tuple[float, float]]:
    """Tipo III: las esquinas del estribo del elemento derecho que caen dentro
    del ala sin una fila de barras del ala a menos de 1 cm: (y, distancia a la
    fila más cercana)."""
    der, ala = armado.elemento("derecho"), armado.elemento("ala")
    if seccion().tipo != "III" or der is None or ala is None or absorbido(der, armado):
        return []
    salida = []
    for y in (der.filas()[0], der.filas()[-1]):
        cerca = float(np.abs(ala.filas() - y).min())
        if cerca > 0.01:
            salida.append((float(y), cerca))
    return salida


def notas_de_la_interseccion(dis: Diseno, armado: Armado,
                             reducido: Armado | None = None) -> list[tuple[str, str]]:
    """§18.10.6.4(d) y la regla de la esquina de §18.10.6.5(b), en la
    intersección alma-ala, como NOTAS para detallar en el plano (criterio del
    proyecto: el ala es columna del pórtico especial y estas reglas no se
    comprueban con cifras). Dan la medida que pide la norma y hasta dónde
    llega hoy el confinamiento declarado.

      §18.10.6.4(d), sección con ala y borde especial: el elemento de borde
          incluye el ancho efectivo del ala en compresión y entra al menos
          300 mm en el alma;
      §18.10.6.5(b), sin borde especial y con ρ del borde > 2.8/fy: en la
          esquina alma-ala, los estribos del borde entran al menos 300 mm en
          el alma y en el ala (fig. R18.10.6.5b: a lo largo del alma b' +
          300 mm desde la cara exterior del ala; a lo largo del ala bw +
          300 mm desde la cara exterior del alma).
    """
    sec = seccion()
    if sec.ala is None:
        return []
    region = dis.region or {}
    limite = 2.8 / D.FY
    salida = []

    def rho_con(arm: Armado, ext: Extremo | None) -> float:
        b = dis.barras(arm)
        if ext is not None:
            return cuantia_de_borde(arm, b, ext)
        ala = arm.elemento("ala")                      # tipo II: el ala central
        return float(b.A[b.zona == "ala"].sum()) / ala.caja.area

    def arriba(ext: Extremo | None) -> str:
        """Qué pasa por encima del corte, con el armado reducido."""
        if reducido is None:
            return ""
        r = rho_con(reducido, ext)
        texto = (f"; con el armado reducido, ρ = {r:.4f}: "
                 + ("también la pide" if r > limite else "ya no la pide"))
        if ext is not None:
            otra = next(x for x in extremos(reducido) if x.nombre == ext.nombre).provista
            if abs(otra - ext.provista) > 1e-9:
                texto += f" (su zona confinada llega a {otra:.2f} m)"
        return texto

    if sec.tipo == "III":
        ext = next(x for x in extremos(armado) if x.nombre == "derecho")
        en_alma = sec.ancho_ala + 0.300                  # desde la cara exterior del ala
        en_ala = min(sec.b + 0.300, sec.largo_ala)       # desde la cara exterior del alma
        exige, altura, _ = region.get("derecho", (False, 0.0, 0.0))
        rho = rho_con(armado, ext)
        if exige:
            salida.append(("18.10.6.4(d)",
                           f"extremo derecho, sección con ala y borde especial (hasta {altura:.2f} m "
                           f"sobre la base): el elemento de borde incluye el ancho efectivo del ala en "
                           f"compresión (el ala confinada entera, {sec.largo_ala:.2f} m) y entra al "
                           f"menos 300 mm en el alma, hasta {en_alma:.2f} m desde la cara exterior del "
                           f"ala (la zona confinada declarada llega a {ext.provista:.2f} m): detallar "
                           f"en el plano"))
        if rho > limite:
            salida.append(("18.10.6.5(b)",
                           "esquina alma-ala del extremo derecho"
                           + (f", encima de {altura:.2f} m" if exige else "")
                           + f" (ρ del borde = {rho:.4f} > 2.8/fy = {limite:.5f}"
                           + arriba(ext) + "): los estribos del borde entran al menos 300 mm "
                           f"en el alma y en el ala (fig. R18.10.6.5b): en el alma hasta "
                           f"{en_alma:.2f} m desde la cara exterior del ala (la zona confinada "
                           f"declarada llega a {ext.provista:.2f} m) y en el ala hasta {en_ala:.2f} m "
                           f"desde la cara exterior del alma (el ala va confinada entera, "
                           f"{sec.largo_ala:.2f} m): detallar en el plano"))
        else:
            salida.append(("18.10.6.5(b)",
                           f"esquina alma-ala del extremo derecho: ρ del borde = {rho:.4f} ≤ 2.8/fy "
                           f"= {limite:.5f}, la norma no pide estribos de borde en la esquina"))
    else:
        # tipo II: el ala está en medio del alma; su intersección no es borde
        # del muro en X, pero en Y es la esquina alma-ala de la columna
        rho = rho_con(armado, None)
        x0, x1 = sec.ala.x0, sec.ala.x1
        if rho > limite:
            salida.append(("18.10.6.5(b)",
                           f"intersección alma-ala (ala central, ρ del ala = {rho:.4f} > 2.8/fy = "
                           f"{limite:.5f}" + arriba(None) + "): los estribos entran al menos "
                           f"300 mm en el alma a cada lado del ala, de x = {x0 - 0.3:.2f} a "
                           f"{x1 + 0.3:.2f} m (el elemento del ala confina de {x0:.2f} a {x1:.2f} m), "
                           f"y en el ala, que va confinada entera ({sec.largo_ala:.2f} m) "
                           f"(fig. R18.10.6.5b): detallar en el plano"))
        else:
            salida.append(("18.10.6.5(b)",
                           f"intersección alma-ala (ala central): ρ del ala = {rho:.4f} ≤ 2.8/fy = "
                           f"{limite:.5f}, la norma no pide estribos de borde en la esquina"))
    return salida


def detalles_constructivos(dis: Diseno, armado: Armado, v: Verificacion,
                           reducido: Armado | None = None) -> list[tuple[str, str]]:
    """Lo que la norma pide dibujar en el plano y que no es una cifra que se
    pueda comprobar con los datos: (artículo, texto). reducido: el armado de
    encima del corte, para decir si la regla de la esquina sigue rigiendo."""
    sec = seccion()
    salida = []
    region = dis.region or {}
    barras = dis.barras(armado)
    for e in armado.elementos:
        if e.estribo is None or not (barras.zona == e.nombre).any():
            continue
        exige = e.nombre in elementos_de(v.exigidas)
        apoyo = apoyo_lateral(e, limite_hx(e, exige)[0], armado)
        separacion = f"@ {e.separacion_estribo * 100:.0f} cm"
        if e.separacion_arriba:
            separacion += (f" hasta {dis.altura_de_cambio():.2f} m sobre la base y @ "
                           f"{e.separacion_arriba * 100:.0f} cm encima")
        salida.append(("18.7.5.2, 25.7.2.3",
                       f"elemento {e.nombre}: estribo cerrado {e.estribo} con ganchos sísmicos "
                       f"(135°) {separacion}; barras apoyadas en esquina o "
                       f"grapa: {apoyo['ramas_y']} por cara en X y {apoyo['ramas_x']} por cara en Y "
                       f"({'todas' if e.grapas == 'todas' else 'esquinas, alternas y las que pide hx'}); "
                       f"hx = {apoyo['hx'] * 100:.1f} cm; grapas {e.estribo} (del mismo diámetro que el "
                       f"estribo: así se cuentan en Ash), con gancho sísmico y alternadas"))
        if exige:
            bcx, bcy = e.nucleo
            largo, corto = max(bcx, bcy), min(bcx, bcy)
            if largo > 2 * corto + 1e-9:
                exento = rama_larga_permitida(e, v, dis)
                if exento:
                    salida.append(("18.10.6.4(f)",
                                   f"elemento {e.nombre}: un solo estribo de {largo * 100:.0f} cm "
                                   f"(> 2·bc = {2 * corto * 100:.0f} cm) vale porque se cumple "
                                   f"{exento}"))
                else:
                    solape = min(0.150, 2.0 / 3.0 * e.espesor)
                    salida.append(("18.10.6.4(f)",
                                   f"elemento {e.nombre}: el estribo tendría una rama de "
                                   f"{largo * 100:.0f} cm > 2·bc = {2 * corto * 100:.0f} cm y no se "
                                   f"cumplen (i) ni (ii): usar estribos superpuestos, solapados al "
                                   f"menos mín(150 mm, 2b/3) = {solape * 100:.0f} cm"))
    # Tipo III: el estribo del elemento derecho abraza las barras del ala en la
    # intersección; sus esquinas (§18.7.5.2(b): los dobleces abrazan barras) y
    # las ramas que se cuentan en hx y Ash necesitan una barra del ala en cada
    # cara del estribo. El modelo pone las filas del ala a distancias iguales
    # a lo largo de l'w; si una no cae en la cara del estribo, en el plano va
    # una fila del ala en esa cara (mover una fila 1 o 2 cm en Y no cambia la
    # flexión en X y apenas la biaxial).
    for arm in [x for x in (armado, reducido) if x is not None]:
        for y, cerca in esquinas_sin_barra(arm):
            salida.append(("18.7.5.2(b), 25.7.2.3",
                           f"armado {arm.nombre}: el estribo del elemento derecho tiene su esquina dentro "
                           f"del ala en y = "
                           f"{y * 100:+.1f} cm y el modelo no tiene ahí barra del ala (la fila más "
                           f"cercana está a {cerca * 100:.1f} cm): en el plano, una fila del ala va en y = "
                           f"{y * 100:+.1f} cm (barras en sus dos columnas), para que la esquina y las "
                           f"ramas que cuentan hx y Ash abracen barras; o cambiar «ny» del ala"))
    if sec.ala is not None:
        salida.append(("18.7",
                       "en Y el ala es columna de los pórticos especiales: su diseño como columna "
                       "(confinamiento en lo, §18.7.5; cortante por capacidad con Mpr, §18.7.6; "
                       "columna fuerte–viga débil, §18.7.3) va con el pórtico. Este programa la "
                       "incluye en la flexocompresión biaxial y la confina como elemento del muro. "
                       "Por ser columna, el confinamiento de la intersección alma-ala "
                       "(§18.10.6.4(d) y la regla de la esquina de §18.10.6.5(b)) no se comprueba "
                       "con cifras: va en las notas siguientes, para detallarlo en el plano"))
    salida += notas_de_la_interseccion(dis, armado, reducido)
    exigidos = [zn for zn, (ex, _, _) in region.items() if ex]
    if exigidos:
        altura = max(region[zn][1] for zn in exigidos)
        salida.append(("18.10.6.4(i)",
                       f"en {altura:.2f} m sobre la base, las barras verticales del alma con apoyo "
                       f"lateral: grapas con ganchos sísmicos en los dos extremos a ≤ 300 mm en "
                       f"altura y de diámetro según §25.7.2.2 (o grapas con gancho de 90° en un "
                       f"extremo y sísmico en el otro, alternadas en el largo y en la altura, a ≤ "
                       f"230 mm)"))
        salida.append(("18.10.6.4(j)",
                       "los estribos del elemento de borde especial entran en el apoyo al menos ℓd "
                       "de la barra vertical mayor del borde; si el borde termina en una zapata, "
                       "losa de cimentación o cabezal, al menos 300 mm (salvo que §18.13.2.4 pida "
                       "más)"))
        salida.append(("18.10.6.4(k)",
                       "la malla horizontal llega a menos de 150 mm del extremo y termina con gancho "
                       "estándar dentro del núcleo confinado (su ℓdh se comprueba en el detallado)"))
        losas = [dis.demanda.z_base[s] for s in range(1, len(D.ALTURAS))
                 if dis.demanda.z_base[s] < altura - 1e-9]
        if losas:
            salida.append(("18.10.6.4(h)",
                           f"el concreto de la losa de {', '.join(f'{z:.2f}' for z in losas)} m, en el "
                           f"espesor del entrepiso y donde está el elemento de borde especial, tiene "
                           f"f'c ≥ 0.7·{D.FC:.1f} = {0.7 * D.FC:.1f} MPa"))
    # §18.10.6.5(a), con ωv·Ωv·Vu como escribe el artículo (el Vu mayorado, sin
    # amplificar, por el factor; del lado seguro frente a amplificar solo VEh).
    # Rige donde no hay borde especial: en los extremos que no lo exigen y,
    # en los que sí, encima de la región crítica (fig. R18.10.6.4c).
    vu = max(abs(e.V) for e in dis.estados(0.0, 0))
    factor = amplificacion()["factor"]
    limite = 0.083 * LAMBDA * RAIZ_FC * MPA * sec.acv
    if factor * vu >= limite:
        donde = ("en todo el muro" if not exigidos else
                 f"encima de {max(region[zn][1] for zn in exigidos):.2f} m"
                 + ("" if len(exigidos) == len(region) else
                    " y, en toda la altura, en el extremo "
                    + " y ".join(zn for zn in region if zn not in exigidos)))
        salida.append(("18.10.6.5(a)",
                       f"donde no hay borde especial ({donde}), la malla horizontal termina con "
                       f"gancho estándar abrazando el acero del extremo, o con horquillas en U del "
                       f"mismo diámetro y separación empalmadas con ella (ωv·Ωv·Vu = "
                       f"{factor:.3f}·{vu:.0f} = {factor * vu:.0f} kN ≥ 0.083·√f'c·Acv = "
                       f"{limite:.0f} kN)"))
    hsx = min(D.ALTURAS[0], 6.0)
    salida.append(("18.10.2.3(b)",
                   "el acero vertical de los extremos desarrolla 1.25·fy en tracción donde puede "
                   "fluir (la región crítica de la base)"))
    zonas = ("las cajas confinadas, y un espesor más allá de la intersección con el ala"
             if sec.ala is not None else "las cajas confinadas")
    salida.append(("18.10.2.3(c)",
                   f"sin traslapes del acero vertical en las zonas de borde ({zonas}) en "
                   f"hsx = {hsx:.2f} m sobre la base ni ℓd por debajo"))
    return salida


# =============================================================================
# 10. LA COTA DE CORTE DEL ARMADO REDUCIDO (§18.10.2.3(a), §18.10.2.4(b), §18.10.6.2(b)(i))
# =============================================================================


def dc_biaxial_en(dis: Diseno, a: Armado, z: float, piso: int) -> tuple[float, Estado]:
    """El D/C de flexocompresión biaxial en una cota y el estado que lo da."""
    dcs = dis.dcs_biaxiales(a, z, piso)
    i = int(np.argmax(dcs))
    return dcs[i], dis.estados(z, piso)[i]


def margen_biaxial(dis: Diseno, a: Armado, z: float, piso: int, k: float = 1.0) -> float:
    """1 − D/C biaxial / k, el peor de los estados. Positivo: cumple. Con
    k = 1 es D/C ≤ 1; con k = D/C biaxial de la base, la «sección crítica
    única»: más arriba ningún armado puede ir más apretado que la base, para
    que la rótula se forme abajo."""
    peor, _ = dc_biaxial_en(dis, a, z, piso)
    return 1.0 - min(peor, 100.0) / k


def losa_encima(dis: Diseno, z: float) -> float:
    """La losa en z o la primera por encima (el techo, si no hay otra).
    Criterio del proyecto: el armado cambia en la losa más próxima por encima
    de donde ya no hace falta el de base, así que la cota de corte es siempre
    una losa (o el techo: sin reducido)."""
    return min((zt for zt in dis.demanda.z_tope if zt >= z - 1e-6), default=H)


def armado_en(dis: Diseno, base: Armado, red: Armado | None, corte: float, piso: int) -> Armado:
    """El armado de un entrepiso: el reducido en los que empiezan en la losa
    del corte o más arriba (en esa losa, la cara de abajo es del entrepiso de
    abajo y lleva el de base)."""
    return base if (red is None or dis.demanda.z_base[piso] < corte - 1e-6) else red


def cota_minima(cotas: list, margen) -> tuple[float, tuple | None]:
    """z*: la cota más baja desde la que el margen es ≥ 0 en TODO lo de arriba.

    Se baja desde el techo y se para en el primer fallo; el cruce se
    interpola linealmente entre la cota que falla y la de encima. Bajar desde
    arriba garantiza que no queda ninguna zona alta sin cumplir (el momento
    salta en cada losa y puede crecer cerca del techo).
    """
    encima = None
    for c in reversed(cotas):
        m = margen(c)
        if m < -1e-6:
            if encima is None:
                return H, c
            cu, mu = encima
            if cu[0] - c[0] <= TOL:
                return cu[0], c
            if math.isfinite(m) and math.isfinite(mu) and mu > m:
                return c[0] + (cu[0] - c[0]) * (-m) / (mu - m), c
            return cu[0], c
        encima = (c, m)
    return 0.0, None


def clave_de_falta(f: dict) -> str:
    """Lo que identifica una comprobación entre el armado de base y el
    reducido: su texto, sin el número de ramas (cambia con las barras). El
    artículo no sirve: con otra cuantía del borde, la misma comprobación de
    estribos sale por §18.10.6.5(b) en uno y por «proyecto» en el otro."""
    return re.sub(r" \(\d+ ramas\)", "", f["texto"])


@dataclass
class CotaDeCorte:
    z_unica: float
    z_biaxial: float
    z_estrella: float               # desde aquí el reducido basta a flexocompresión
    z_vertical: float               # §18.10.2.3(a): z* + 3.60 m
    z_extremo: float                # §18.10.2.4(b): máx(ℓw, Mu/3Vu) (0 si hw/ℓw < 2)
    z_borde: float                  # §18.10.6.2(b)(i): la altura del borde especial (0 si no se exige)
    art_borde: str                  # el artículo de esa altura (§18.10.6.3 en la vía por esfuerzo)
    fraccion_terminada: float
    z_detallado: float              # del resto (corte, borde, malla horizontal, estribos…): la losa
                                    # encima del entrepiso más alto donde falla el reducido
    fallas_del_reducido: list       # [(clave, z más alta, piso, texto)], lo que falla por el reducido
    corte_requerido: float          # la mayor de las cuatro cotas: desde ahí ya no hace falta la base
    corte: float                    # la losa en el corte requerido o la primera encima
    gobierna: str
    verificacion: Verificacion | None

    @property
    def cotas_de_la_norma(self) -> list[tuple[str, float]]:
        """Las tres cotas que el reducido no puede pisar (criterio del
        proyecto): empieza en el entrepiso inmediatamente encima de la mayor."""
        return [("18.10.2.3(a)", self.z_vertical), ("18.10.2.4(b)", self.z_extremo),
                (self.art_borde, self.z_borde)]


def fallas_del_reducido(dis: Diseno, base: Armado, red: Armado, z: float, piso: int) -> list:
    """Lo que falla en una cota POR EL ARMADO REDUCIDO: el cortante, la
    longitud del borde y todo el detallado (cuantías y separaciones de las
    mallas, estribos, Ash, hx…), comparado con el de base en la misma cota. Lo
    que falla igual con el de base no depende del reducido: se arregla aparte
    y sale en la verificación completa. Lista de (clave, texto)."""
    v = dis.verificar(red, z, piso)
    vb = dis.verificar(base, z, piso)
    motivos = []
    for nombre, dr, db in (("cortante D/C", v.dc_corte, vb.dc_corte),
                           ("longitud de borde (exigida/confinada)", v.dc_borde, vb.dc_borde)):
        if dr > 1.0 and dr > db + 1e-9:
            motivos.append((nombre, f"{nombre} = {dr:.3f}"))
    de_la_base = {clave_de_falta(f) for f in vb.faltas}
    motivos += [(clave_de_falta(f), f"[{f['art']}] {f['texto']}: {f['valor']:.4g} frente a "
                                    f"{texto_limite(f['limite'])}")
                for f in v.faltas if clave_de_falta(f) not in de_la_base]
    return motivos


def cota_de_corte(dis: Diseno, base: Armado, red: Armado, dc_base: float,
                  z_ext: float, cotas: list | None = None,
                  cotas_de_detallado: list | None = None) -> CotaDeCorte:
    """La cota de corte del armado reducido. dc_base: el D/C biaxial de la
    base (la sección crítica única); z_ext: la altura de §18.10.2.4(b).

    LAS TRES COTAS DE LA NORMA (criterio del proyecto: el reducido empieza en
    el entrepiso inmediatamente encima de la mayor):
      §18.10.2.3(a): z* + 3.60 m, con z* la cota más baja desde la que el
          reducido cumple la flexocompresión biaxial en todo lo de arriba
          (D/C ≤ 1 y no más que en la base);
      §18.10.2.4(b): máx(ℓw, Mu/3Vu), hasta donde sube el acero de extremo
          (si hw/ℓw ≥ 2, cuando rige §18.10.2.4);
      §18.10.6.2(b)(i): máx(ℓw, Mu/4Vu), hasta donde sube el borde especial,
          si se exige (en la vía por esfuerzo, §18.10.6.3, hasta donde la
          compresión baja de 0.15·f'c).

    LO DEMÁS (el cortante con la malla horizontal del reducido, la longitud
    del borde, los estribos y todo el detallado): el reducido no puede estar
    en un entrepiso donde falla por él, así que el corte queda en la losa de
    encima del entrepiso más alto donde falla. Se comprueba en todas las cotas.

    EN UNA LOSA (criterio del proyecto): el armado cambia en la losa más
    próxima por encima del corte requerido (`losa_encima`); si es el techo, el
    reducido no se usa.

    Para la búsqueda del optimizador (por defecto, todo como arriba): `cotas`,
    menos cotas para el biaxial; `cotas_de_detallado`, las únicas donde se
    busca lo que falla por el reducido, y solo en los entrepisos por encima
    del corte del acero vertical, de arriba abajo, hasta la primera que falla
    (la más alta es la única que mueve el corte). La lista de fallas queda
    entonces incompleta."""
    cotas = dis.cotas() if cotas is None else cotas
    zscu, _ = cota_minima(cotas, lambda c: margen_biaxial(dis, red, c[0], c[1], dc_base))
    zb, _ = cota_minima(cotas, lambda c: margen_biaxial(dis, red, c[0], c[1]))
    z_est = max(zb, zscu)
    # §18.10.2.4(c): qué parte del acero exigido por (a) se termina en el corte
    br = dis.barras(red)
    exigida = cuantia_exigida_de_extremo()
    queda = min(cuantia_de_extremo(br, "izquierdo"), cuantia_de_extremo(br, "derecho"), exigida)
    frac = max(0.0, (exigida - queda) / exigida)

    z_extremo = z_ext if ESBELTEZ >= 2.0 else 0.0
    exigidos = [zn for zn, (ex, _, _) in (dis.region or {}).items() if ex]
    z_borde = max((dis.region[zn][1] for zn in exigidos), default=0.0)
    via = dis.casos_region[exigidos[0]]["via"] if exigidos else "18.10.6.2"
    art_borde = "18.10.6.2(b)(i)" if via == "18.10.6.2" else "18.10.6.3"
    normas = {f"§18.10.2.3(a), z* + 3.60 m ({'sección crítica única' if zscu > zb + TOL else 'biaxial'})":
              z_est + PROLONGACION,
              "§18.10.2.4(b), acero de extremo": z_extremo,
              f"§{art_borde}, borde especial": z_borde}
    gobierna = max(normas, key=normas.get)
    z_norma = normas[gobierna]
    corte = losa_encima(dis, z_norma)
    fallas: dict = {}
    if cotas_de_detallado is None:
        for z, s, _ in dis.cotas():
            for clave, texto in fallas_del_reducido(dis, base, red, z, s):
                if clave not in fallas or (s, z) > (fallas[clave][2], fallas[clave][1] + TOL):
                    fallas[clave] = (clave, z, s, texto)
    else:
        for z, s, _ in sorted(cotas_de_detallado, key=lambda c: (-c[0], -c[1])):
            if z < corte - TOL:
                break
            if dis.demanda.z_base[s] < corte - 1e-6:     # la cara de abajo de la losa del corte
                continue
            motivos = fallas_del_reducido(dis, base, red, z, s)
            if motivos:
                clave, texto = motivos[0]
                fallas[clave] = (clave, z, s, texto)
                break
    lista = sorted(fallas.values(), key=lambda f: (-f[2], -f[1]))
    z_det = dis.demanda.z_tope[lista[0][2]] if lista else 0.0
    if z_det > corte + TOL:
        corte, gobierna = z_det, "el reducido falla más arriba (corte, borde o detallado)"
    verif = None
    if corte < H - TOL and cotas_de_detallado is None:
        verif = dis.verificar(red, corte, dis.piso_de(corte))
    return CotaDeCorte(zscu, zb, z_est, z_est + PROLONGACION, z_extremo, z_borde, art_borde, frac,
                       z_det, lista, max(z_norma, z_det), corte, gobierna, verif)


def revisar_bajo_el_corte(dis: Diseno, base: Armado, corte: float, dc_base: float) -> dict:
    """El acero de base tiene que cumplir en todo lo que queda por debajo del
    corte: la flexocompresión biaxial (D/C ≤ 1) y la sección crítica única
    (D/C ≤ el de la base), en los entrepisos que lo llevan. dc_base: el D/C
    biaxial de la base."""
    bajo = [c for c in dis.cotas() if dis.demanda.z_base[c[1]] < corte - 1e-6]
    _, fs = cota_minima(bajo, lambda c: margen_biaxial(dis, base, c[0], c[1], dc_base))
    _, fb = cota_minima(bajo, lambda c: margen_biaxial(dis, base, c[0], c[1]))
    peor = max(((dc_biaxial_en(dis, base, z, s)[0], z, cara) for z, s, cara in bajo),
               key=lambda t: t[0])
    return {"peor_dc": peor[0], "peor_z": peor[1], "peor_cara": peor[2],
            "falla_unica": None if (fs is None or fs[0] <= 0.0) else fs[0],
            "falla_biaxial": None if fb is None else fb[0]}


# =============================================================================
# 11. PESO DE ACERO Y EL ARMADO EN UNA LÍNEA
# =============================================================================

def gancho(d: float) -> float:
    """Largo aproximado de un gancho sísmico de 135° (extensión de 6·db, no
    menos de 75 mm, más el doblez)."""
    return max(6.0 * d, 0.075) + 4.0 * d


def peso_de_acero(dis: Diseno, base: Armado, red: Armado | None = None,
                  corte: float | None = None) -> dict:
    """kg de acero en toda la altura H, sin traslapes: vertical (cada barra de
    los elementos y de la malla), malla horizontal (cada cortina a lo largo del
    alma, entre recubrimientos) y estribos (perímetro de cada estribo cerrado
    más sus grapas, con los ganchos, por zonas en altura: en cada zona las
    grapas que pide su apoyo lateral). Con reducido, la base va hasta la cota
    de corte y el reducido encima. Pide la base ya verificada en z = 0
    (`Diseno.verificar`), que decide las zonas de estribos."""
    if red is None or corte is None or corte >= H - TOL:
        red, corte = None, H
    tramos = [(base, corte)] + ([(red, H - corte)] if red is not None else [])
    vertical = sum(kg_por_metro(a) * h for a, h in tramos)
    horizontal = sum(a.malla_horizontal.cortinas * masa(a.malla_horizontal.barra)
                     * (D.LW - 2 * D.RECUBRIMIENTO) * h / a.malla_horizontal.separacion
                     for a, h in tramos)
    estribos = 0.0
    for filas in zonas_de_estribos(dis, base, red, corte).values():
        for f in filas:
            e, est = f["elemento"], f["est"]
            if est is None:
                continue
            d = diametro(e.estribo)
            x0, x1, y0, y1 = e.contorno_del_estribo()
            a, b = x1 - x0 - d, y1 - y0 - d                    # por el eje del estribo
            largo = 2 * (a + b) + 2 * gancho(d)
            largo += max(0, est["apoyo"]["ramas_y"] - 2) * (b + 2 * gancho(d))
            largo += max(0, est["apoyo"]["ramas_x"] - 2) * (a + 2 * gancho(d))
            estribos += largo * masa(e.estribo) * (f["z1"] - f["z0"]) / e.separacion_estribo
    return {"vertical": vertical, "horizontal": horizontal, "estribos": estribos,
            "total": vertical + horizontal + estribos}


def texto_del_armado(armado: Armado, cambio: float | None = None) -> str:
    """El armado en una línea: barras de cada elemento, «longitud», estribos y
    mallas. cambio: la altura de cambio de los estribos de la base, si se
    conoce."""
    b = colocar_barras(armado)
    partes = []
    for e in armado.elementos:
        if absorbido(e, armado):
            partes.append(f"derecho: lo absorbe el ala (L = {e.caja.ancho:.2f} m)")
            continue
        n = int((b.zona == e.nombre).sum())
        largo = f", L = {e.caja.ancho:.2f} m" if e.nombre != "ala" else ""
        estribo = (f"estribo {e.estribo}@{e.separacion_estribo * 100:.0f}" if e.estribo is not None
                   else "sin estribo")
        if e.separacion_arriba:
            donde = f"{cambio:.2f} m" if cambio is not None else "la altura de cambio"
            estribo += f" (@{e.separacion_arriba * 100:.0f} encima de {donde})"
        partes.append(f"{e.nombre}: {n}×{e.barra} ({e.nx}×{e.ny}){largo}, {estribo}")
    partes += [f"malla vertical {armado.malla_vertical.texto()}",
               f"malla horizontal {armado.malla_horizontal.texto()}"]
    return " | ".join(partes)
