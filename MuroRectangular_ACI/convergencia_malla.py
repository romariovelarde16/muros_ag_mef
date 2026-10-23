"""ESTUDIO DE CONVERGENCIA DE MALLA DEL MEF (aparte del diseño).

Se corre solo cuando se quiere, igual que ejecutar.py:

    python convergencia_malla.py                         (el muro de MURO_A_EJECUTAR)
    python convergencia_malla.py edificacion_01/datos_P17.py

Con el estado del sismo en el plano (la fila Max de la primera combinación de
COMBINACIONES_DEL_SISMO_EN_X) resuelve el MEF del alma con elementos de 0.80,
0.40, 0.20, 0.10 y 0.05 m, y con dos elementos:
  - Q4: el cuadrilátero bilineal que usa el programa (mef.rigidez_q4);
  - Q6: el mismo con dos modos incompatibles (Wilson et al., 1973), solo para
    comparar.
Y mide:
  - las fuerzas de sección en la base de cada entrepiso frente a ETABS: el muro
    aislado y empotrado es isostático, así que tienen que coincidir con
    cualquier malla;
  - el desplazamiento del techo frente al de una viga en voladizo de Timoshenko
    con las mismas cargas (flexión más corte, κ = 5/6).

Escribe en resultados/<MURO>/convergencia_malla/ su informe (.txt), la tabla
(.csv) y la figura (.png). El diseño no cambia: usa siempre MALLA_MEF.
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
from ejecutar import D, F, mef  # noqa: E402

TAMANOS = (0.80, 0.40, 0.20, 0.10, 0.05)     # m, tamaño máximo del elemento
KAPPA = 5.0 / 6.0                            # factor de corte de la sección rectangular
MALLA_DEL_PROGRAMA = D.MALLA_MEF             # la del diseño (resolver cambia D.MALLA_MEF)


def rigidez_q6(x: np.ndarray, z: np.ndarray, t: float, Dm: np.ndarray) -> np.ndarray:
    """Cuadrilátero de 4 nudos con los modos incompatibles (1 − ξ²) y (1 − η²)
    en cada dirección, condensados (8×8); el jacobiano de los modos, en el
    centro del elemento."""
    g = 1.0 / math.sqrt(3.0)
    xi_n = np.array([-1, 1, 1, -1])
    eta_n = np.array([-1, -1, 1, 1])
    J0 = np.array([[0.25 * xi_n @ x, 0.25 * xi_n @ z], [0.25 * eta_n @ x, 0.25 * eta_n @ z]])
    K = np.zeros((12, 12))
    for xi in (-g, g):
        for eta in (-g, g):
            dN_dxi = 0.25 * xi_n * (1 + eta * eta_n)
            dN_deta = 0.25 * eta_n * (1 + xi * xi_n)
            J = np.array([[dN_dxi @ x, dN_dxi @ z], [dN_deta @ x, dN_deta @ z]])
            dN = np.linalg.solve(J, np.vstack([dN_dxi, dN_deta]))
            dI = np.linalg.solve(J0, np.array([[-2 * xi, 0.0], [0.0, -2 * eta]]))
            B = np.zeros((3, 12))
            B[0, 0:8:2] = dN[0]
            B[1, 1:8:2] = dN[1]
            B[2, 0:8:2] = dN[1]
            B[2, 1:8:2] = dN[0]
            B[0, 8:10] = dI[0]
            B[1, 10:12] = dI[1]
            B[2, 8:10] = dI[1]
            B[2, 10:12] = dI[0]
            K += B.T @ Dm @ B * t * np.linalg.det(J)
    Kcc, Kci, Kii = K[:8, :8], K[:8, 8:], K[8:, 8:]
    return Kcc - Kci @ np.linalg.solve(Kii, Kci.T)


def resolver(fuerzas: dict, estado: tuple, tamano: float, elemento: str) -> dict:
    """El MEF del alma con elementos de `tamano` como máximo: desplazamiento del
    techo y mayor diferencia de las fuerzas de sección con ETABS (relativa)."""
    D.MALLA_MEF = tamano
    malla = mef.crear_malla()
    nx, nz = malla.nx, len(malla.zs)
    Dm = mef.matriz_D()
    rigidez = mef.rigidez_q4 if elemento == "Q4" else rigidez_q6
    filas, columnas, valores, elementos = [], [], [], []
    for j in range(nz - 1):
        for i in range(nx - 1):
            nudos = [malla.nudo(i, j), malla.nudo(i + 1, j), malla.nudo(i + 1, j + 1), malla.nudo(i, j + 1)]
            Ke = rigidez(malla.xs[[i, i + 1, i + 1, i]], malla.zs[[j, j, j + 1, j + 1]], D.ESPESOR, Dm)
            gdl = np.array([[2 * n, 2 * n + 1] for n in nudos]).ravel()
            filas.append(np.repeat(gdl, 8))
            columnas.append(np.tile(gdl, 8))
            valores.append(Ke.ravel())
            elementos.append((nudos, gdl, Ke, j))
    n_gdl = 2 * nx * nz
    K = coo_matrix((np.concatenate(valores), (np.concatenate(filas), np.concatenate(columnas))),
                   shape=(n_gdl, n_gdl)).tocsr()
    pisos = range(len(D.PISOS))
    base = [(fuerzas[estado][s]["Bottom"]["P"], fuerzas[estado][s]["Bottom"]["Vx"],
             fuerzas[estado][s]["Bottom"]["Mx"]) for s in pisos]
    tope = [fuerzas[estado][s]["Top"]["P"] for s in pisos]
    cargas = mef.cargas_por_losa(base, tope)
    Fv = mef.vector_de_cargas(malla, cargas, mef.pesos_propios(base, tope))
    fijos = np.array([[2 * malla.nudo(i, 0), 2 * malla.nudo(i, 0) + 1] for i in range(nx)]).ravel()
    libres = np.setdiff1d(np.arange(n_gdl), fijos)
    U = np.zeros(n_gdl)
    U[libres] = spsolve(K[libres][:, libres].tocsc(), Fv[libres])
    # fuerzas de sección en la base de cada entrepiso, como mef.resolver
    encima = np.zeros((nz, 3))
    for nudos, gdl, Ke, j in elementos:
        f = Ke @ U[gdl]
        for k in (0, 1):
            x = malla.xs[nudos[k] % nx]
            encima[j, 0] -= f[2 * k + 1]
            encima[j, 1] -= f[2 * k]
            encima[j, 2] -= f[2 * k + 1] * (x - D.LW / 2)
    error_fuerzas = 0.0
    for s in pisos:
        for k in range(3):
            escala = max(abs(r[k]) for r in base) or 1.0
            error_fuerzas = max(error_fuerzas, abs(base[s][k] - encima[malla.bases[s], k]) / escala)
    techo = [malla.nudo(i, nz - 1) for i in range(nx)]
    return {"elementos": len(elementos), "u_techo": float(np.mean(U[[2 * n for n in techo]])),
            "error_fuerzas": error_fuerzas, "cargas": cargas}


def timoshenko(cargas: list) -> tuple[float, float]:
    """Desplazamiento del techo de la viga en voladizo de Timoshenko con las
    cargas de cada losa (H y C), y la parte que es de corte."""
    E = D.EC * mef.MPA
    G = E / (2 * (1 + D.NU))
    area = D.ESPESOR * D.LW
    inercia = D.ESPESOR * D.LW ** 3 / 12
    L = sum(D.ALTURAS)
    flexion = corte = 0.0
    z = 0.0
    for (H, _N, C), h in zip(cargas, D.ALTURAS):
        z += h
        flexion += H * z * z * (3 * L - z) / (6 * E * inercia) - C * z * (2 * L - z) / (2 * E * inercia)
        corte += H * z / (KAPPA * G * area)
    return flexion + corte, corte / (flexion + corte)


def figura(filas: list[dict], ruta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(F.ESTILO_FIGURAS):
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        ax.axhspan(-5, 5, color="0.92", lw=0, zorder=0, label="±5 %")
        for elemento, estilo, marca in (("Q4", "-", "o"), ("Q6", "--", "s")):
            f = [r for r in filas if r["elemento"] == elemento]
            ax.plot([r["tamano_m"] for r in f], [r["error_u_pct"] for r in f], color="k", ls=estilo,
                    marker=marca, ms=4, label=f"{elemento}" + (" (el del programa)" if elemento == "Q4"
                                                               else " (modos incompatibles)"))
        ax.axhline(0, color="0.4", lw=0.5)
        ax.axvline(MALLA_DEL_PROGRAMA, color="0.45", lw=0.8, ls=":", label="malla del programa")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks(TAMANOS)
        ax.set_xticklabels([f"{t:.2f}" for t in TAMANOS])
        ax.minorticks_off()
        ax.set_xlabel("Tamaño máximo del elemento (m)")
        ax.set_ylabel("Error del desplazamiento del techo (%)")
        ax.grid(True, color="0.82", ls=":", lw=0.6)
        ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Muro {D.MURO} — convergencia de malla del MEF", fontsize=11)
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    out = ejecutar.Informe()
    estado = (D.COMBINACIONES_DEL_SISMO_EN_X[0], "Max")
    fuerzas = mef.mapear(mef.leer_fuerzas_etabs())
    out.titulo(f"CONVERGENCIA DE MALLA DEL MEF — MURO {D.MURO}")
    out(f"  Estado: {estado[0]} {estado[1]} (el sismo en el plano).  hw/ℓw = {sum(D.ALTURAS) / D.LW:.2f};  "
        f"el diseño usa MALLA_MEF = {MALLA_DEL_PROGRAMA:.2f} m.")
    out("  Error u: el desplazamiento del techo frente a la viga de Timoshenko (flexión + corte, κ = 5/6).")
    out("  Error de fuerzas: la mayor diferencia con ETABS en la base de cada entrepiso (P, V y M),")
    out("  relativa al mayor valor de cada una (el muro es isostático: tiene que ser ~0).")
    out("")
    out(f"  {'elemento':<9}{'tamaño (m)':>11}{'elementos':>11}{'u MEF (mm)':>12}{'u Timosh. (mm)':>16}"
        f"{'error u (%)':>13}{'error fuerzas':>15}")
    filas = []
    for elemento in ("Q4", "Q6"):
        for tamano in TAMANOS:
            r = resolver(fuerzas, estado, tamano, elemento)
            ref, parte_corte = timoshenko(r["cargas"])
            fila = {"elemento": elemento, "tamano_m": tamano, "elementos": r["elementos"],
                    "u_techo_mm": 1000 * r["u_techo"], "u_timoshenko_mm": 1000 * ref,
                    "error_u_pct": 100 * (r["u_techo"] - ref) / ref, "error_fuerzas": r["error_fuerzas"],
                    "parte_de_corte_pct": 100 * parte_corte}
            filas.append(fila)
            out(f"  {elemento:<9}{tamano:>11.2f}{fila['elementos']:>11d}{fila['u_techo_mm']:>12.4f}"
                f"{fila['u_timoshenko_mm']:>16.4f}{fila['error_u_pct']:>+13.2f}{fila['error_fuerzas']:>15.1e}")
    out("")
    out(f"  En la viga de Timoshenko, el corte es el {filas[0]['parte_de_corte_pct']:.1f} % del desplazamiento.")
    carpeta = F.carpeta_del_muro() / "convergencia_malla"
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / f"convergencia_malla_{D.MURO}.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    figura(filas, carpeta / f"convergencia_malla_{D.MURO}.png")
    (carpeta / f"informe_{D.MURO}_convergencia_malla.txt").write_text("\n".join(out.lineas) + "\n",
                                                                     encoding="utf-8")
    out(f"  Informe, tabla (.csv) y figura (.png) en {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
