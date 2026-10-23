"""ESTUDIO DEL MEF: DESPLAZAMIENTOS Y ESFUERZOS (aparte del diseño).

Se corre solo cuando se quiere, igual que ejecutar.py:

    python estudio_mef.py                         (el muro de MURO_A_EJECUTAR)
    python estudio_mef.py edificacion_01/datos_P1.py

El diseño toma del MEF solo las fuerzas de sección, y esas salen de la
estática: el muro aislado y empotrado es isostático. Este estudio usa lo que
solo da el MEF, con el caso estático de sismo en el plano del muro
(CASOS_ESTATICOS, paso 1: sin excentricidad):

1. DESPLAZAMIENTOS frente a ETABS. El desplazamiento del muro en el techo de
   cada piso con tres modelos: el alma sola con la malla del programa
   (MALLA_MEF), el alma sola con elementos de TAMANO_FINO y, en los tipos II y
   III, el alma con el ala (placas plegadas: cada una en su plano, unidas en la
   línea de encuentro, que comparte el desplazamiento vertical; el ala sale del
   eje del alma, como en el modelo de ETABS). El alma sola lleva las fuerzas
   del plano (P, Vx, Mx); el modelo con ala, además, las de fuera del plano
   (Vy, My), porque en una sección sin simetría las dos flexiones van juntas.
   Se comparan con los de ETABS en etabs/desplazamientos_piers.csv (leídos
   del modelo con la API de ETABS).

2. ESFUERZOS. El esfuerzo vertical σz del modelo más completo en la base y a
   media altura del primer entrepiso, frente al de secciones planas,
   σ = N/A + a·(x − x̄) + b·(y − ȳ), con [a, b] de las inercias Ixx, Iyy, Ixy
   y los momentos Mx, My (flexión biaxial). En los tipos II y III, además, la
   eficacia del ala en cada cota: ψ = F_ala,MEF / F_ala,secciones planas, la
   fuerza vertical que toma el ala en el MEF frente a la que le asigna la
   hipótesis de secciones planas con el ala entera (la que admite la norma
   cuando el ancho efectivo cubre el ala).

Estos esfuerzos no deciden los elementos de borde: el programa los decide por
desplazamientos (§18.10.6.2); la vía por esfuerzos (§18.10.6.3) no se usa en
los muros estudiados.

Escribe en resultados/<MURO>/estudio_mef/ su informe (.txt), las tablas (.csv)
y la figura (.png). El diseño no cambia.
"""

from __future__ import annotations

import csv
import math
import sys

sys.dont_write_bytecode = True      # sin carpetas __pycache__ junto a los archivos

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402

import ejecutar  # noqa: E402  (carga el archivo de datos y los cálculos)
from ejecutar import A, D, F, mef  # noqa: E402

TAMANO_FINO = 0.05                   # m, los modelos finos del estudio
ARCHIVO_DESPLAZAMIENTOS = "etabs/desplazamientos_piers.csv"
MALLA_DEL_PROGRAMA = D.MALLA_MEF


def particion(a: float, b: float, tamano: float) -> list[float]:
    n = max(1, math.ceil((b - a) / tamano - 1e-9))
    return [a + (b - a) * k / n for k in range(n + 1)]


def tributaria(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    t = np.zeros(len(v))
    t[:-1] += np.diff(v) / 2
    t[1:] += np.diff(v) / 2
    return t


class Modelo:
    """Alma (plano x-z, en y = 0) y, si `con_ala`, el ala (plano y-z, en la x del ala).

    Tres grados de libertad por nudo (ux, uy, uz): el alma trabaja con (ux, uz)
    y el ala con (uy, uz); los nudos del encuentro tienen los tres. Los que no
    tienen rigidez se fijan, y todos los de la base."""

    def __init__(self, tamano: float, con_ala: bool):
        sec = A.seccion()
        D.MALLA_MEF = tamano
        self.malla = mef.crear_malla()                     # cotas, losas y bases
        D.MALLA_MEF = MALLA_DEL_PROGRAMA
        self.zs = self.malla.zs
        self.con_ala = con_ala and sec.tipo != "I"
        if not self.con_ala:
            self.xs = list(self.malla.xs) if tamano == MALLA_DEL_PROGRAMA else particion(0.0, sec.lw, tamano)
            self.xa, self.ys = None, []
        else:
            self.xa = D.ALA_X if sec.tipo == "II" else sec.lw - sec.ancho_ala / 2
            self.xs = particion(0.0, self.xa, tamano)
            if sec.tipo == "II":
                self.xs = self.xs[:-1] + particion(self.xa, sec.lw, tamano)
            self.ys = particion(0.0, sec.largo_ala - sec.b / 2, tamano)
        self.indice: dict = {}
        self.coords: list = []
        for z in self.zs:
            for x in self.xs:
                self._nudo(x, 0.0, z)
            for y in self.ys:
                self._nudo(self.xa, y, z)
        self.Dm = mef.matriz_D()
        self.elementos = []                                 # (nudos, gdl, coord. locales, plano, Ke)
        for j in range(len(self.zs) - 1):
            zz = self.zs[[j, j, j + 1, j + 1]]
            for i in range(len(self.xs) - 1):
                xx = np.array([self.xs[i], self.xs[i + 1], self.xs[i + 1], self.xs[i]])
                n = [self.indice[self._clave(x, 0.0, z)] for x, z in zip(xx, zz)]
                self._elemento(n, [g for k in n for g in (3 * k, 3 * k + 2)], xx, zz, sec.b, "alma")
            for i in range(len(self.ys) - 1):
                yy = np.array([self.ys[i], self.ys[i + 1], self.ys[i + 1], self.ys[i]])
                n = [self.indice[self._clave(self.xa, y, z)] for y, z in zip(yy, zz)]
                self._elemento(n, [g for k in n for g in (3 * k + 1, 3 * k + 2)], yy, zz, sec.ancho_ala, "ala")
        ngdl = 3 * len(self.coords)
        self.K = self._ensamblar(ngdl)
        diag = self.K.diagonal()
        base = [g for k, (_, _, z) in enumerate(self.coords) if z == 0.0 for g in (3 * k, 3 * k + 1, 3 * k + 2)]
        fijos = set(base) | set(np.where(diag <= 1e-9 * diag.max())[0].tolist())
        self.libres = np.array(sorted(set(range(ngdl)) - fijos))

    def _elemento(self, nudos, gdl, u, z, t, plano):
        self.elementos.append((nudos, gdl, (u, z), plano, mef.rigidez_q4(u, z, t, self.Dm)))

    def _ensamblar(self, ngdl):
        filas = np.concatenate([np.repeat(e[1], 8) for e in self.elementos])
        cols = np.concatenate([np.tile(e[1], 8) for e in self.elementos])
        vals = np.concatenate([e[4].ravel() for e in self.elementos])
        return coo_matrix((vals, (filas, cols)), shape=(ngdl, ngdl)).tocsr()

    @staticmethod
    def _clave(x, y, z):
        return (round(x, 6), round(y, 6), round(z, 6))

    def _nudo(self, x, y, z):
        c = self._clave(x, y, z)
        if c not in self.indice:
            self.indice[c] = len(self.coords)
            self.coords.append((x, y, z))
        return self.indice[c]

    def seccion_del_modelo(self) -> dict:
        """Área, centroide e inercias (Ixx = ∫(x − x̄)² dA, Iyy, Ixy) de la sección del modelo."""
        sec = A.seccion()
        L = self.xs[-1] - self.xs[0]
        piezas = [(sec.b * L, (self.xs[0] + self.xs[-1]) / 2, 0.0, sec.b * L**3 / 12, L * sec.b**3 / 12)]
        if self.con_ala:
            Lf = self.ys[-1] - self.ys[0]
            piezas.append((sec.ancho_ala * Lf, self.xa, (self.ys[0] + self.ys[-1]) / 2,
                           Lf * sec.ancho_ala**3 / 12, sec.ancho_ala * Lf**3 / 12))
        area = sum(p[0] for p in piezas)
        xb = sum(p[0] * p[1] for p in piezas) / area
        yb = sum(p[0] * p[2] for p in piezas) / area
        return {"A": area, "xb": xb, "yb": yb,
                "Ixx": sum(p[3] + p[0] * (p[1] - xb) ** 2 for p in piezas),
                "Iyy": sum(p[4] + p[0] * (p[2] - yb) ** 2 for p in piezas),
                "Ixy": sum(p[0] * (p[1] - xb) * (p[2] - yb) for p in piezas)}

    def _plano(self, j):
        z = self.zs[j]
        return ([self.indice[self._clave(x, 0.0, z)] for x in self.xs] +
                [self.indice[self._clave(self.xa, y, z)] for y in self.ys])

    def cargas(self, cx, cy, pesos) -> np.ndarray:
        """Como mef.vector_de_cargas, sobre toda la sección: Hx en el alma y Hy en el ala; N, los
        pares Cx y Cy y el peso W, como fuerzas verticales con el área tributaria de cada nudo
        (los pares, lineales con x e y, de resultante nula)."""
        sec = A.seccion()
        a_alma = tributaria(self.xs) * sec.b
        a_ala = tributaria(self.ys) * sec.ancho_ala if self.ys else np.zeros(0)
        area = np.concatenate([a_alma, a_ala])
        dx = np.concatenate([self.xs, [self.xa] * len(self.ys)])
        dy = np.concatenate([np.zeros(len(self.xs)), self.ys])
        dx = dx - (dx * area).sum() / area.sum()
        dy = dy - (dy * area).sum() / area.sum()
        inercia = np.array([[(dx * dx * area).sum(), (dx * dy * area).sum()],
                            [(dx * dy * area).sum(), (dy * dy * area).sum()]])
        f = np.zeros(3 * len(self.coords))
        for s, j in enumerate(self.malla.losas):
            Hx, N, Cx = cx[s]
            Hy, Cy = (cy[s][0], cy[s][2]) if cy else (0.0, 0.0)
            a, b = (np.linalg.solve(inercia, [Cx, Cy]) if cy else (Cx / inercia[0, 0], 0.0))
            for k, n in enumerate(self._plano(j)):
                if k < len(self.xs):
                    f[3 * n] += Hx * area[k] / a_alma.sum()
                else:
                    f[3 * n + 1] += Hy * area[k] / a_ala.sum()
                f[3 * n + 2] += area[k] * (N / area.sum() + a * dx[k] + b * dy[k])
        for s, W in enumerate(pesos):
            interiores = list(range(self.malla.bases[s] + 1, self.malla.losas[s]))
            trib = np.array([(self.zs[j + 1] - self.zs[j - 1]) / 2 for j in interiores])
            for j, parte in zip(interiores, trib / trib.sum()):
                for k, n in enumerate(self._plano(j)):
                    f[3 * n + 2] += W * parte * area[k] / area.sum()
        return f

    def resolver(self, f: np.ndarray) -> np.ndarray:
        U = np.zeros_like(f)
        U[self.libres] = spsolve(self.K[self.libres][:, self.libres].tocsc(), f[self.libres])
        return U

    def desplazamiento_de_losas(self, U: np.ndarray) -> list[float]:
        """ux medio de los nudos del alma en cada losa."""
        return [float(np.mean([U[3 * self.indice[self._clave(x, 0.0, self.zs[j])]] for x in self.xs]))
                for j in self.malla.losas]

    def sigma_z(self, U: np.ndarray, z: float, plano: str) -> list[tuple[float, float, float]]:
        """(coordenada, ancho, σz) en el centro horizontal de cada elemento del `plano`, a la cota z."""
        j = int(np.searchsorted(self.zs, z, side="right") - 1)
        j = min(max(j, 0), len(self.zs) - 2)
        eta = 2 * (z - self.zs[j]) / (self.zs[j + 1] - self.zs[j]) - 1
        return [(float(u[0] + u[1]) / 2, float(u[1] - u[0]), self._sz(u, zz, U[gdl], eta))
                for _, gdl, (u, zz), p, _ in self.elementos if p == plano and abs(zz[0] - self.zs[j]) < 1e-9]

    def _sz(self, u, z, ue, eta) -> float:
        """σz en el centro horizontal (ξ = 0) del elemento, a la altura natural η."""
        xi_n = np.array([-1, 1, 1, -1])
        eta_n = np.array([-1, -1, 1, 1])
        dN_dxi = 0.25 * xi_n * (1 + eta * eta_n)
        dN_deta = 0.25 * eta_n
        J = np.array([[dN_dxi @ u, dN_dxi @ z], [dN_deta @ u, dN_deta @ z]])
        dN = np.linalg.solve(J, np.vstack([dN_dxi, dN_deta]))
        B = np.zeros((3, 8))
        B[0, 0::2] = dN[0]
        B[1, 1::2] = dN[1]
        B[2, 0::2] = dN[1]
        B[2, 1::2] = dN[0]
        return float((self.Dm @ B @ ue)[1])


def caso_en_el_plano() -> tuple[str, np.ndarray]:
    """El caso estático que empuja en la dirección global del alma, y esa dirección."""
    casos = getattr(D, "CASOS_ESTATICOS", None) or {}
    ex = mef.direcciones_globales()["X"]
    sentido = "X" if abs(ex[0]) >= abs(ex[1]) else "Y"
    if sentido not in casos:
        raise mef.DatoInvalido(f"el estudio necesita el caso estático en {sentido} global (CASOS_ESTATICOS)")
    return casos[sentido], ex


def fuerzas_del_caso(caso: str, s: dict) -> tuple[list, list, list]:
    """(P, Vx, Mx) y (P, Vy, My) en la base de cada entrepiso, con los momentos en el centroide
    de la sección del modelo, y P en el tope de cada entrepiso."""
    fz = mef.mapear(mef.leer_fuerzas_etabs([caso]))[(caso, "paso 1")]
    sec = A.seccion()
    pisos = range(len(D.PISOS))
    bx, by = [], []
    for k in pisos:
        r = fz[k]["Bottom"]
        bx.append((r["P"], r["Vx"], r["Mx"] + r["P"] * (sec.xe - s["xb"])))
        by.append((r["P"], r["Vy"], r["My"] + r["P"] * (sec.ye - s["yb"])))
    return bx, by, [fz[k]["Top"]["P"] for k in pisos]


def secciones_planas(s: dict, N: float, Mx: float, My: float, con_y: bool):
    """σ(x, y) = N/A + a·(x − x̄) + b·(y − ȳ) de la flexión biaxial."""
    if con_y:
        a, b = np.linalg.solve([[s["Ixx"], s["Ixy"]], [s["Ixy"], s["Iyy"]]], [Mx, My])
    else:
        a, b = Mx / s["Ixx"], 0.0
    return lambda x, y: N / s["A"] + a * (x - s["xb"]) + b * (y - s["yb"])


def en_la_cota(bx, by, z: float) -> tuple[float, float, float]:
    """N, Mx y My a la cota z (V constante en el entrepiso: dM/dz = V)."""
    z0 = 0.0
    for k, h in enumerate(D.ALTURAS):
        if z <= z0 + h + 1e-9 or k == len(D.ALTURAS) - 1:
            return bx[k][0], bx[k][2] + bx[k][1] * (z - z0), by[k][2] + by[k][1] * (z - z0)
        z0 += h
    raise ValueError(z)


def desplazamientos_etabs(caso: str, ex: np.ndarray) -> list[float] | None:
    """Desplazamiento de ETABS en el plano del muro (+X del programa) en el techo de cada piso."""
    ruta = mef.carpeta_de_los_datos() / ARCHIVO_DESPLAZAMIENTOS
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8-sig") as f:
        filas = [r for r in csv.DictReader(f) if r["Pier"] == D.MURO and r["OutputCase"] == caso]
    por_piso = {r["Story"]: float(r["Ux (m)"]) * ex[0] + float(r["Uy (m)"]) * ex[1] for r in filas}
    if any(p not in por_piso for p in D.PISOS):
        return None
    return [por_piso[p] for p in D.PISOS]


def figura(ruta, cotas, perfiles, cortes, ala) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paneles = 3 if ala else 2
    with plt.rc_context(F.ESTILO_FIGURAS):
        fig, ejes = plt.subplots(1, paneles, figsize=(3.1 * paneles + 0.4, 4.0))
        ax = ejes[0]
        for nombre, (u, estilo, marca) in perfiles.items():
            ax.plot([0.0] + [1000 * v for v in u], [0.0] + cotas, color="k", ls=estilo, marker=marca, ms=4,
                    lw=1.0, label=nombre)
        ax.set_xlabel("Desplazamiento en el plano (mm)")
        ax.set_ylabel("Cota (m)")
        ax.grid(True, color="0.85", ls=":", lw=0.6)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_title("(a) Desplazamiento de cada losa")
        for ax, plano, datos, titulo in ((ejes[1], "alma", cortes, "(b) σz en el alma"),
                                         (ejes[2] if ala else None, "ala", ala, "(c) σz en el ala")):
            if ax is None:
                continue
            for (rotulo, valores, planas), estilo in zip(datos, ("-", "--")):
                ax.plot([c for c, _, _ in valores], [v / 1000 for _, _, v in valores], color="k", ls=estilo,
                        lw=1.0, label=f"MEF, {rotulo}")
                ax.plot([c for c, _, _ in valores], [v / 1000 for v in planas], color="0.55", ls=estilo,
                        lw=1.0, label=f"secciones planas, {rotulo}")
            ax.axhline(0, color="0.4", lw=0.5)
            ax.set_xlabel(f"{'x a lo largo del alma' if plano == 'alma' else 'y a lo largo del ala'} (m)")
            ax.set_ylabel("σz (MPa), tracción +")
            ax.grid(True, color="0.85", ls=":", lw=0.6)
            ax.legend(fontsize=6.8, loc="best")
            ax.set_title(titulo)
        fig.suptitle(f"Muro {D.MURO}: estudio del MEF con el sismo estático en su plano", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    out = ejecutar.Informe()
    sec = A.seccion()
    caso, ex = caso_en_el_plano()
    out.titulo(f"ESTUDIO DEL MEF: DESPLAZAMIENTOS Y ESFUERZOS — MURO {D.MURO} (tipo {sec.tipo})")
    out(f"  Caso: {caso}, paso 1 (sin excentricidad), el que empuja en el plano del muro "
        f"({mef.nombre_global(ex)} = +X del programa).")
    out("  Los elementos de borde no se deciden con estos esfuerzos: el programa usa la vía por")
    out("  desplazamientos (§18.10.6.2); la vía por esfuerzos (§18.10.6.3) no se usa aquí.")
    out("")

    modelos = {f"alma, {MALLA_DEL_PROGRAMA:.2f} m (programa)": Modelo(MALLA_DEL_PROGRAMA, False),
               f"alma, {TAMANO_FINO:.2f} m": Modelo(TAMANO_FINO, False)}
    if sec.tipo != "I":
        modelos[f"alma y ala, {TAMANO_FINO:.2f} m"] = Modelo(TAMANO_FINO, True)
    resultados = {}
    for nombre, m in modelos.items():
        s = m.seccion_del_modelo()
        bx, by, tope = fuerzas_del_caso(caso, s)
        cx = mef.cargas_por_losa(bx, tope)
        cy = mef.cargas_por_losa(by, tope) if m.con_ala else None
        U = m.resolver(m.cargas(cx, cy, mef.pesos_propios(bx, tope)))
        resultados[nombre] = (m, U, s, bx, by)
    cotas = [float(c) for c in np.cumsum(D.ALTURAS)]
    etabs = desplazamientos_etabs(caso, ex)

    out("1. DESPLAZAMIENTO DE CADA LOSA EN EL PLANO DEL MURO (mm) Y DIFERENCIA CON ETABS")
    out(f"  {'piso':<12}{'cota (m)':>9}" + "".join(f"{n:>28}" for n in modelos) + f"{'ETABS':>10}")
    perfiles = {nombre: m.desplazamiento_de_losas(U) for nombre, (m, U, *_) in resultados.items()}
    filas_d = []
    for k, piso in enumerate(D.PISOS):
        linea = f"  {piso:<12}{cotas[k]:9.2f}"
        fila = {"piso": piso, "cota_m": round(cotas[k], 3)}
        for nombre in modelos:
            u = perfiles[nombre][k]
            dif = f" ({100 * (u - etabs[k]) / etabs[k]:+.1f} %)" if etabs else ""
            linea += f"{1000 * u:>17.3f}{dif:>11}"
            fila[f"u {nombre} (mm)"] = round(1000 * u, 4)
        if etabs:
            linea += f"{1000 * etabs[k]:>10.3f}"
            fila["u ETABS (mm)"] = round(1000 * etabs[k], 4)
        out(linea)
        filas_d.append(fila)
    if etabs is None:
        out(f"  (sin {ARCHIVO_DESPLAZAMIENTOS}: se compara solo con los modelos de elementos finitos)")
    else:
        out("  ETABS: el promedio de los nudos del pier en el techo de cada piso (diafragma rígido).")
    mejor = list(modelos)[-1]
    out("")
    out(f"  Deriva elástica de entrepiso del muro, Δu/h, con el modelo «{mejor}»:")
    u = [0.0] + perfiles[mejor]
    ue = [0.0] + (etabs or [])
    for k, piso in enumerate(D.PISOS):
        der = (u[k + 1] - u[k]) / D.ALTURAS[k]
        filas_d[k]["deriva MEF"] = round(der, 6)
        linea = f"    {piso:<12}{abs(der):.5f}"
        if etabs:
            de = (ue[k + 1] - ue[k]) / D.ALTURAS[k]
            filas_d[k]["deriva ETABS"] = round(de, 6)
            linea += f"   ETABS {abs(de):.5f}"
        out(linea)
    out("")

    m, U, s, bx, by = resultados[mejor]
    out("2. ESFUERZO VERTICAL σz EN EL ALMA FRENTE A SECCIONES PLANAS")
    out(f"  Sección del modelo: A = {s['A']:.4f} m², x̄ = {s['xb']:.4f} m, ȳ = {s['yb']:.4f} m, "
        f"Ixx = {s['Ixx']:.5f}, Iyy = {s['Iyy']:.5f}, Ixy = {s['Ixy']:+.5f} m⁴.")
    zb = float(m.zs[1]) / 2
    cortes, cortes_ala, filas_s = [], [], []
    for rotulo, z in ((f"z = {zb:.3f} m", zb), (f"z = {D.ALTURAS[0] / 2:.2f} m", D.ALTURAS[0] / 2)):
        N, Mx, My = en_la_cota(bx, by, z)
        sp = secciones_planas(s, N, Mx, My, m.con_ala)
        fem = m.sigma_z(U, z, "alma")
        planas = [sp(c, 0.0) for c, _, _ in fem]
        cortes.append((rotulo, fem, planas))
        out(f"  {rotulo}: N = {N:.1f} kN, Mx = {Mx:.1f} kN·m" + (f", My = {My:.1f} kN·m" if m.con_ala else ""))
        for k in (0, -1):
            c, _, v = fem[k]
            out(f"    x = {c:.3f} m: MEF {v / 1000:+.3f} MPa, secciones planas {planas[k] / 1000:+.3f} MPa "
                f"(cociente {v / planas[k]:.3f})")
        for (c, _, v), p in zip(fem, planas):
            filas_s.append({"cota_m": round(z, 4), "plano": "alma", "coordenada_m": round(c, 4),
                            "sigma_MEF_MPa": round(v / 1000, 5), "sigma_planas_MPa": round(p / 1000, 5)})
        if m.con_ala:
            fem_a = m.sigma_z(U, z, "ala")
            planas_a = [sp(m.xa, c) for c, _, _ in fem_a]
            cortes_ala.append((rotulo, fem_a, planas_a))
            for (c, _, v), p in zip(fem_a, planas_a):
                filas_s.append({"cota_m": round(z, 4), "plano": "ala", "coordenada_m": round(c, 4),
                                "sigma_MEF_MPa": round(v / 1000, 5), "sigma_planas_MPa": round(p / 1000, 5)})
    out("")

    filas_a = []
    if m.con_ala and sec.tipo == "II":
        out("3. EL ALA DEL TIPO II")
        out(f"  El ala está en x = {m.xa:.3f} m, sobre el eje neutro de la flexión en el plano (x̄ = {s['xb']:.3f} m):")
        out("  esa flexión casi no le da fuerza, y su eficacia ψ no se mide. Como placa, el ala no tiene")
        out("  rigidez para flexionarse en x (su espesor b' no cuenta en ese sentido): el desplazamiento en el")
        out("  plano es prácticamente el del alma sola.")
        out("")
    if m.con_ala and sec.tipo == "III":
        hw = sum(D.ALTURAS)
        out("3. EFICACIA DEL ALA: ψ = F_ala (MEF) / F_ala (secciones planas, ala entera)")
        out(f"  Ala: b' = {sec.ancho_ala:.2f} m, l'w = {sec.largo_ala:.2f} m; en el modelo, de y = 0 (eje del "
            f"alma) a {m.ys[-1]:.3f} m, en x = {m.xa:.3f} m.")
        out(f"  Ancho efectivo de la norma a cada lado del alma: ACI 318-25 §18.10.5.2, 0.25·hw = {0.25 * hw:.2f} m;")
        out(f"  NTE E.060 §21.9.6.3, 0.10·hw = {0.10 * hw:.2f} m (o la mitad de la distancia al muro vecino): los")
        out("  dos pasan del largo del ala, que la norma toma entera. El MEF mide cuánto trabaja de verdad.")
        filas_z = [float(m.zs[j] + m.zs[j + 1]) / 2 for j in range(len(m.zs) - 1)]
        for z in filas_z:
            N, Mx, My = en_la_cota(bx, by, z)
            sp = secciones_planas(s, N, Mx, My, True)
            fem_a = m.sigma_z(U, z, "ala")
            f_mef = sum(v * w for _, w, v in fem_a) * sec.ancho_ala
            f_pl = sum(sp(m.xa, c) * w for c, w, _ in fem_a) * sec.ancho_ala
            filas_a.append({"cota_m": round(z, 4), "F_ala_MEF_kN": round(f_mef, 3),
                            "F_ala_planas_kN": round(f_pl, 3)})
        tope_f = max(abs(f["F_ala_planas_kN"]) for f in filas_a)
        for f in filas_a:
            f["psi"] = (round(f["F_ala_MEF_kN"] / f["F_ala_planas_kN"], 4)
                        if abs(f["F_ala_planas_kN"]) >= 0.10 * tope_f else "")
        out(f"  {'cota (m)':>10}{'F_ala MEF (kN)':>16}{'F_ala planas (kN)':>19}{'ψ':>8}")
        paso = max(1, len(filas_a) // 14)
        for f in filas_a[::paso]:
            psi = f"{f['psi']:8.3f}" if f["psi"] != "" else f"{'–':>8}"
            out(f"  {f['cota_m']:10.3f}{f['F_ala_MEF_kN']:16.1f}{f['F_ala_planas_kN']:19.1f}{psi}")
        psis = [f["psi"] for f in filas_a if f["psi"] != ""]
        cerca = [f["psi"] for f in filas_a if f["psi"] != "" and f["cota_m"] >= sec.lw]
        out(f"  ψ entre {min(psis):.3f} y {max(psis):.3f} (donde la fuerza del ala pasa del 10 % de la mayor);")
        if cerca:
            out(f"  a partir de z = ℓw = {sec.lw:.2f} m, entre {min(cerca):.3f} y {max(cerca):.3f}.")
        out("")

    carpeta = F.carpeta_del_muro() / "estudio_mef"
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, filas in (("desplazamientos", filas_d), ("esfuerzos", filas_s), ("ala", filas_a)):
        ruta = carpeta / f"estudio_mef_{nombre}_{D.MURO}.csv"
        if not filas:
            ruta.unlink(missing_ok=True)
            continue
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
    estilos = [("--", "s"), (":", "^"), ("-", "o")]
    signo = 1.0 if perfiles[mejor][-1] >= 0 else -1.0       # la figura, con el desplazamiento positivo
    trazos = {n: ([signo * v for v in u], *estilos[k]) for k, (n, u) in enumerate(perfiles.items())}
    if etabs:
        trazos["ETABS"] = ([signo * v for v in etabs], "none", "D")
    figura(carpeta / f"estudio_mef_{D.MURO}.png", cotas, trazos, cortes, cortes_ala)
    (carpeta / f"informe_{D.MURO}_estudio_mef.txt").write_text("\n".join(out.lineas) + "\n", encoding="utf-8")
    out(f"  Informe, tablas (.csv) y figura (.png) en {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
