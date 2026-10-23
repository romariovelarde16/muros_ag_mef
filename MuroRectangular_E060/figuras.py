"""Las figuras del informe, en estilo de tesis (letra serif, negro y grises,
300 ppp), cada una en la carpeta de su parte dentro de resultados/<MURO>/:

    0_seccion_y_demanda   planta, planta_reducido, envolventes_X, envolventes_Y
    1_flexocompresion     diagrama_PM, biaxial, dc_en_altura
    2_corte               cortante_en_altura

Las llama ejecutar.py (figuras()); no hace falta tocar este archivo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import datos as D
import diseno_e060 as A
import mef

# Las carpetas de resultados/<MURO>/, una por parte del informe (con su texto
# y sus figuras)
DEMANDA = "0_seccion_y_demanda"
MODULO_1 = "1_flexocompresion"
MODULO_2 = "2_corte"
MODULO_3 = "3_elementos_de_borde"
VERIFICACION = "4_verificacion_completa"
FIGURAS = {"planta": DEMANDA, "planta_reducido": DEMANDA, "envolventes_X": DEMANDA,
           "envolventes_Y": DEMANDA,
           "diagrama_PM": MODULO_1, "biaxial": MODULO_1, "dc_en_altura": MODULO_1,
           "cortante_en_altura": MODULO_2}


def carpeta_del_muro() -> Path:
    """resultados/<MURO>/ en la carpeta de la edificación (la del archivo de
    datos), o resultados/<CARPETA_RESULTADOS>/ si los datos la traen (el armado
    óptimo usa «<MURO>_optimo» para no pisar el informe)."""
    return mef.carpeta_de_los_datos() / "resultados" / (getattr(D, "CARPETA_RESULTADOS", None) or D.MURO)


def ruta_de_figura(nombre: str) -> Path:
    """resultados/<MURO>/<carpeta del módulo>/<nombre>_<MURO>.png"""
    return carpeta_del_muro() / FIGURAS[nombre] / f"{nombre}_{D.MURO}.png"


# Estilo de las figuras: sobrio, para una tesis (serif, negro y grises, marcas
# hacia dentro, 300 ppp). Times New Roman si está (Windows); si no, Liberation
# Serif o STIX, que tienen las mismas medidas.
ESTILO_FIGURAS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "0.25",
    "legend.framealpha": 1.0,
    "savefig.dpi": 300,
    "axes.unicode_minus": True,
}


def figuras(out, dis, base, red, vb, cc, filas_corte, corte: float) -> None:
    """Las figuras, en estilo de tesis: la sección con el armado de la base, el
    diagrama P-M de la base, el contorno biaxial, el D/C biaxial en altura, el
    cortante de diseño por entrepiso y las envolventes de momento y cortante
    del MEF en X y en Y (la envolvente total, con los valores en cada piso).
    Cada una va a la carpeta de su parte (FIGURAS)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Patch, Rectangle
        from matplotlib.ticker import NullLocator
        from matplotlib.transforms import blended_transform_factory
    except ImportError:
        out("  (matplotlib no está instalado: no se dibujan figuras)")
        return
    for carpeta in set(FIGURAS.values()):
        (carpeta_del_muro() / carpeta).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(ESTILO_FIGURAS):
        _dibujar(plt, Line2D, Circle, Patch, Rectangle, NullLocator, blended_transform_factory,
                 dis, base, red, vb, cc, filas_corte, corte)
    out(f"  Figuras en {carpeta_del_muro()}:")
    hechas = [n for n in FIGURAS if n != "planta_reducido" or red is not None]
    for carpeta in dict.fromkeys(FIGURAS.values()):
        out(f"    {carpeta}/: " + ", ".join(f"{n}_{D.MURO}.png" for n in hechas
                                           if FIGURAS[n] == carpeta))


NOMBRES_DE_ELEMENTOS = {"izquierdo": "Extremo izquierdo", "derecho": "Extremo derecho", "ala": "Ala"}


def cuadro_de_armado(arm, b, cambio: float | None = None) -> str:
    """El texto del cuadro de armado de la planta: por elemento, sus barras
    verticales y su estribo; y las dos mallas del alma (cortinas, barra y
    separación). b: las barras colocadas del armado; cambio: la altura de
    cambio de la separación de estribos de la base."""
    lineas = ["ARMADO"]
    for e in arm.elementos:
        n = int((b.zona == e.nombre).sum())
        nombre = NOMBRES_DE_ELEMENTOS[e.nombre]
        if n == 0:
            lineas.append(f"{nombre}: dentro del ala (lo confina el ala)")
            continue
        lineas.append(f"{nombre}: {n} Ø {e.barra}")
        lineas.append(f"    estribo Ø {e.estribo} @ {e.separacion_estribo * 100:.0f} cm"
                      if e.estribo is not None else "    sin estribo")
        if e.separacion_arriba and cambio is not None:
            lineas[-1] += f" hasta {cambio:.2f} m"
            lineas.append(f"    y @ {e.separacion_arriba * 100:.0f} cm encima")
    for texto, m in (("Malla vertical", arm.malla_vertical), ("Malla horizontal", arm.malla_horizontal)):
        if m is arm.malla_vertical and not (b.zona == "malla").any():
            lineas.append(f"{texto}: no lleva (tramos ≤ {m.separacion * 100:.0f} cm)")
            continue
        cortinas = "1 cortina" if m.cortinas == 1 else f"{m.cortinas} cortinas"
        lineas.append(f"{texto}: {cortinas}, Ø {m.barra} @ {m.separacion * 100:.0f} cm")
    return "\n".join(lineas)


# Colores de las combinaciones con sismo en el diagrama P-M, en orden fijo
# (azul, naranja, aguamarina y violeta, validados para daltonismo entre todos
# los pares; si hubiera más de cuatro, siguen magenta, verde, rojo y amarillo,
# que ya no se distinguen tan bien); las combinaciones sin sismo van en negro.
COLORES_COMBINACIONES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
                         "#e87ba4", "#008300", "#e34948", "#eda100"]


def _dibujar(plt, Line2D, Circle, Patch, Rectangle, NullLocator, blended, dis, base, red,
             vb, cc, filas_corte, corte: float) -> None:
    sec = A.seccion()
    dem = dis.demanda
    pisos = range(len(D.ALTURAS))
    zb, zt = dem.z_base, dem.z_tope
    niveles = [0.0] + list(zt)

    def num(v, decimales=1):
        """El número con el signo menos tipográfico."""
        return f"{v:.{decimales}f}".replace("-", "−")

    def guardar(fig, nombre):
        fig.savefig(ruta_de_figura(nombre), bbox_inches="tight")
        plt.close(fig)

    def rejilla(ax):
        ax.grid(True, which="major", color="0.82", ls=":", lw=0.6)
        ax.set_axisbelow(True)

    def eje_de_niveles(ax, rotular=True):
        """El eje z acotado en cada nivel (la base y cada losa) y las losas
        dibujadas como líneas finas."""
        ax.set_yticks(niveles)
        ax.set_yticklabels([f"{z:.2f}" for z in niveles] if rotular else [])
        ax.yaxis.set_minor_locator(NullLocator())
        for z in niveles[1:-1]:
            ax.axhline(z, color="0.55", lw=0.5, ls="-.", zorder=1)
        margen = 0.03 * niveles[-1]
        ax.set_ylim(-margen, niveles[-1] + margen)

    def nombres_de_pisos(ax):
        """«Piso i» a la derecha del eje, a media altura de cada entrepiso."""
        t = blended(ax.transAxes, ax.transData)
        for s in pisos:
            ax.text(1.02, (zb[s] + zt[s]) / 2, f"Piso {s + 1}", transform=t, rotation=90,
                    va="center", ha="left", fontsize=9)

    def titulo(fig, texto):
        fig.suptitle(f"Muro {D.MURO} — {texto}", fontsize=11)

    # --- 0. sección con su armado: la de la base y la del reducido ---------------
    def planta(arm, nombre, subtitulo):
        """La sección con el armado `arm`: barras, estribos, zonas confinadas
        (las cajas) y sus cotas."""
        b = dis.barras(arm)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for p in sec.piezas:
            ax.add_patch(Rectangle((p.x0, p.y0), p.ancho, p.alto, fc="0.92", ec="none", zorder=1))
        for p in sec.piezas:
            ax.add_patch(Rectangle((p.x0, p.y0), p.ancho, p.alto, fc="none", ec="k", lw=1.0, zorder=3))
        for e in arm.elementos:
            c = e.caja
            ax.add_patch(Rectangle((c.x0, c.y0), c.ancho, c.alto, fc="none", ec="0.35", lw=0.8,
                                   ls=(0, (4, 2)), zorder=2))
            if e.estribo is not None and (b.zona == e.nombre).any():
                # el eje del estribo: medio diámetro por dentro de su cara exterior
                x0, x1, y0, y1 = e.contorno_del_estribo()
                m = A.diametro(e.estribo) / 2
                ax.add_patch(Rectangle((x0 + m, y0 + m), x1 - x0 - 2 * m, y1 - y0 - 2 * m,
                                       fc="none", ec="k", lw=0.7, zorder=4))
        for x, y, d, z in zip(b.x, b.y, b.d, b.zona):
            ax.add_patch(Circle((x, y), d / 2, fc="w" if z == "malla" else "k", ec="k", lw=0.6,
                                zorder=5))
        ax.plot(sec.xc, sec.yc, "k+", ms=9, mew=1.0, zorder=6)
        # cotas: la longitud de cada zona confinada y ℓw, debajo de la sección
        y_cota = -sec.b / 2 - 0.07

        def cota(x0, x1, y, texto):
            ax.annotate("", xy=(x0, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color="k", shrinkA=0, shrinkB=0,
                                        mutation_scale=7))
            ax.text((x0 + x1) / 2, y - 0.012, texto, ha="center", va="top", fontsize=8)

        marcas = sorted({0.0, sec.lw} | {e.caja.x0 for e in arm.elementos if e.nombre != "ala"}
                        | {e.caja.x1 for e in arm.elementos if e.nombre != "ala"})
        for x0, x1 in zip(marcas[:-1], marcas[1:]):
            cota(x0, x1, y_cota, f"{(x1 - x0) * 100:.0f}")
        cota(0.0, sec.lw, y_cota - 0.09, rf"$\ell_w$ = {sec.lw * 100:.0f}")
        for x in marcas:
            ax.plot([x, x], [-sec.b / 2 - 0.015, y_cota - 0.09 - 0.01], color="k", lw=0.4)

        def cota_vertical(x, y0, y1, texto, lado):
            ax.annotate("", xy=(x, y0), xytext=(x, y1),
                        arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color="k", shrinkA=0, shrinkB=0,
                                        mutation_scale=7))
            ax.text(x + lado * 0.012, (y0 + y1) / 2, texto, rotation=90, va="center",
                    ha="right" if lado < 0 else "left", fontsize=8)

        x_b = -0.05
        cota_vertical(x_b, -sec.b / 2, sec.b / 2, f"{sec.b * 100:.0f}", -1)
        for y in (-sec.b / 2, sec.b / 2):
            ax.plot([x_b - 0.01, -0.015], [y, y], color="k", lw=0.4)
        if sec.ala is not None:
            a = sec.ala
            # l'w, desde la cara exterior del alma: a la derecha de todo el muro
            x_l = sec.lw + 0.05
            cota_vertical(x_l, a.y0, a.y1, rf"$\ell'_w$ = {(a.y1 - a.y0) * 100:.0f}", 1)
            ax.plot([sec.lw + 0.015, x_l + 0.01], [a.y0, a.y0], color="k", lw=0.4)
            ax.plot([a.x1 + 0.015, x_l + 0.01], [a.y1, a.y1], color="k", lw=0.4)
            y_a = a.y1 + 0.05
            cota(a.x0, a.x1, y_a, "")
            ax.text((a.x0 + a.x1) / 2, y_a + 0.012, f"$b'$ = {(a.x1 - a.x0) * 100:.0f}", ha="center",
                    va="bottom", fontsize=8)
            for x in (a.x0, a.x1):
                ax.plot([x, x], [a.y1 + 0.015, y_a + 0.01], color="k", lw=0.4)
        # ejes del programa con su dirección global
        dg = mef.direcciones_globales()
        largo = 0.10 * sec.lw
        x_ej = -0.10 - largo * 0.2
        y_ej = sec.ala.y1 + 0.06 if sec.ala is not None else sec.b / 2 + 0.06
        ax.annotate("", xy=(x_ej + largo, y_ej), xytext=(x_ej, y_ej),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", mutation_scale=8))
        ax.text(x_ej + largo, y_ej, f" $+X$ ({mef.nombre_global(dg['X'])})", va="center", fontsize=8)
        ax.annotate("", xy=(x_ej, y_ej + largo), xytext=(x_ej, y_ej),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", mutation_scale=8))
        ax.text(x_ej, y_ej + largo, f" $+Y$ ({mef.nombre_global(dg['Y'])})", va="bottom", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(x_ej - 0.04, sec.lw + 0.20)
        ax.set_ylim(y_cota - 0.17, max(y_ej + largo, sec.ala.y1 + 0.12 if sec.ala is not None else 0) + 0.07)
        ax.axis("off")
        leyenda = [Line2D([], [], marker="o", ls="", mfc="k", mec="k", ms=5,
                          label="Barra de elemento confinado")]
        if (b.zona == "malla").any():
            leyenda.append(Line2D([], [], marker="o", ls="", mfc="w", mec="k", ms=5, label="Barra de la malla"))
        leyenda += [Line2D([], [], color="k", lw=0.7, label="Estribo"),
                   Line2D([], [], color="0.35", lw=0.8, ls=(0, (4, 2)), label="Zona confinada"),
                   Line2D([], [], marker="+", ls="", color="k", ms=8, label="Centroide de la sección")]
        caja = ax.legend(handles=leyenda, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        # el cuadro de armado, debajo de la leyenda: las barras y el estribo de
        # cada elemento y las dos mallas del alma
        fig.canvas.draw()
        abajo = caja.get_window_extent().transformed(ax.transAxes.inverted()).y0
        ax.text(1.01, abajo - 0.04, cuadro_de_armado(arm, b, dis.altura_de_cambio()), transform=ax.transAxes,
                ha="left",
                va="top", fontsize=8, linespacing=1.35,
                bbox=dict(boxstyle="square,pad=0.6", fc="w", ec="0.3", lw=0.8))
        ax.text(0.0, y_cota - 0.155, "Cotas en cm.", fontsize=8, style="italic")
        titulo(fig, f"sección tipo {sec.tipo}, {subtitulo}")
        guardar(fig, nombre)

    planta(base, "planta", "armado de la base")
    if red is not None:
        planta(red, "planta_reducido",
               f"armado reducido (desde z = {corte:.2f} m)" if corte < A.H - A.TOL
               else "armado reducido (no se usa: el corte cae en el techo o por encima)")

    # --- 1. diagrama P-M con la demanda de la base (compresión hacia arriba) -----
    diag = dis.diagrama(base)
    nom, dis_ = diag.nominal, diag.diseno
    est0 = dis.estados(0.0, 0)
    combos = list(dict.fromkeys(e.combo for e in est0))
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.plot(np.r_[nom[:, 1], nom[0, 1]], -np.r_[nom[:, 0], nom[0, 0]], color="k", ls="--", lw=1.0,
            label=r"Nominal, $M_n$–$P_n$")
    ax.plot(np.r_[dis_[:, 1], dis_[0, 1]], -np.r_[dis_[:, 0], dis_[0, 0]], color="k", lw=1.4,
            label=r"De diseño, $\phi M_n$–$\phi P_n$")
    # cada combinación con su color (la misma marca para todas); el color
    # sigue a la combinación, en el orden de COMBINACIONES
    con_sismo = [c for c in D.COMBINACIONES if any(e.con_sismo for e in est0 if e.combo == c)]
    for combo in combos:
        pts = [(e.M, -e.P) for e in est0 if e.combo == combo]
        color = (COLORES_COMBINACIONES[con_sismo.index(combo) % len(COLORES_COMBINACIONES)]
                 if combo in con_sismo else "k")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="", marker="o", ms=5, mfc=color,
                mec="w", mew=0.5, label=combo, zorder=3)
    g = vb.estado_biaxial
    hacia = 1 if g.M < 0 else -1                        # hacia el interior del diagrama
    ax.annotate(f"{g.nombre}\nD/C biaxial = {vb.dc_biaxial:.3f}", xy=(g.M, -g.P),
                xytext=(18 * hacia, 22), textcoords="offset points", fontsize=8,
                ha="left" if hacia > 0 else "right", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_xlabel(r"Momento flector $M_x$ (kN$\cdot$m)")
    ax.set_ylabel(r"Carga axial $P$ (kN), compresión positiva")
    rejilla(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, fontsize=8)
    titulo(fig, r"diagrama de interacción $P$–$M_x$ en la base (referencia)")
    guardar(fig, "diagrama_PM")

    # --- 2. contorno biaxial al Pu del estado que gobierna -------------------------
    e = vb.estado_biaxial
    borde = dis.superficie(base).contorno(e.P)
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    if borde is not None:
        ax.plot(np.r_[borde[:, 0], borde[0, 0]], np.r_[borde[:, 1], borde[0, 1]], color="k", lw=1.4,
                label=r"Contorno de diseño, $\phi M_{nx}$–$\phi M_{ny}$")
    ax.plot(e.M, e.My, ls="", marker="o", ms=5, mfc="k", mec="k",
            label=f"Demanda, {e.nombre} (D/C = {vb.dc_biaxial:.3f})")
    ax.plot([0, e.M], [0, e.My], color="0.4", lw=0.6, ls=":")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_xlabel(r"Momento $M_{ux}$, en el plano del muro (kN$\cdot$m)")
    ax.set_ylabel(r"Momento $M_{uy}$ (kN$\cdot$m)")
    rejilla(ax)
    ax.legend(loc="best", fontsize=8)
    titulo(fig, f"contorno de interacción biaxial en la base, $P_u$ = {abs(e.P):.1f} kN "
                f"({'compresión' if e.P < 0 else 'tracción'})")
    guardar(fig, "biaxial")

    # --- 3. D/C de flexocompresión biaxial en altura ----------------------------------
    # cada armado en toda la altura (el reducido, para ver dónde puede ir) y,
    # acotado en la base y el tope de cada entrepiso, el D/C del armado que
    # lleva el muro ahí
    cotas = dis.cotas()
    zc = np.array([z for z, _, _ in cotas])
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    for arm, estilo in [(base, "-"), (red, "--")]:
        if arm is None:
            continue
        dc = [A.dc_biaxial_en(dis, arm, z, s)[0] for z, s, _ in cotas]
        ax.plot(dc, zc, color="k", ls=estilo, lw=1.3,
                label="Armado de base" if arm is base else "Armado reducido")
    ax.axvline(1.0, color="0.45", lw=1.6, label="D/C = 1")
    ax.axvline(vb.dc_biaxial, color="k", ls=":", lw=1.0,
               label=f"D/C en la base = {vb.dc_biaxial:.3f}")
    for s in pisos:
        for z, desplaza, va in ((zb[s], 2, "bottom"), (zt[s], -2, "top")):
            dc, _ = A.dc_biaxial_en(dis, A.armado_en(dis, base, red, corte, s), z, s)
            ax.plot(dc, z, "o", ms=2.5, color="k")
            ax.annotate(f"{dc:.3f}", xy=(dc, z), xytext=(4, desplaza), textcoords="offset points",
                        ha="left", va=va, fontsize=7.5)
    eje_de_niveles(ax)
    if cc is not None:
        t = blended(ax.transAxes, ax.transData)
        for z, texto, estilo in ((cc.z_estrella, rf"$z^*$ = {cc.z_estrella:.2f} m", (0, (1, 2))),
                                 (cc.corte, f"corte = {cc.corte:.2f} m", (0, (6, 2)))):
            if z < niveles[-1]:
                ax.axhline(z, color="k", lw=0.8, ls=estilo)
                ax.text(0.98, z, texto, transform=t, ha="right", va="bottom", fontsize=8)
    nombres_de_pisos(ax)
    ax.set_xlim(0, max(1.1, ax.get_xlim()[1]))
    ax.set_xlabel("Relación demanda/capacidad biaxial, D/C")
    ax.set_ylabel(r"Altura $z$ (m)")
    rejilla(ax)
    ax.legend(loc="upper right", fontsize=8)
    titulo(fig, "flexocompresión biaxial, D/C en altura")
    guardar(fig, "dc_en_altura")

    # --- 4. cortante de diseño por entrepiso ---------------------------------------------
    resumen = []
    for s in pisos:
        filas = [f for f in filas_corte if f["piso"] == s]
        peor = max(filas, key=lambda f: f["dc"])
        resumen.append({"piso": s, "z0": zb[s], "z1": zt[s], "Vua": max(f["Vua"] for f in filas),
                        "Vu": max(f["Vu"] for f in filas), "phiVn": peor["phiVn"], "dc": peor["dc"],
                        "combo": peor["combo"]})

    def escalones(ax, clave, **kw):
        xs, ys = [], []
        for r in resumen:
            xs += [r[clave], r[clave]]
            ys += [r["z0"], r["z1"]]
        ax.plot(xs, ys, **kw)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    escalones(ax, "Vua", color="k", ls="--", lw=1.1, label=r"$|V_{ua}|$, del análisis")
    escalones(ax, "Vu", color="k", lw=1.5, label=r"$|V_u|$, de diseño")
    escalones(ax, "phiVn", color="0.45", lw=2.2, label=r"$\phi V_n$")
    for r in resumen:
        zm = (r["z0"] + r["z1"]) / 2
        ax.annotate(num(r["Vu"]), xy=(r["Vu"], zm), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8)
        ax.annotate(num(r["Vua"]), xy=(r["Vua"], zm), xytext=(-3, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8)
        ax.annotate(f"D/C = {r['dc']:.2f}", xy=(r["phiVn"], zm), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8)
    eje_de_niveles(ax)
    tope = max(r["phiVn"] for r in resumen)
    ax.set_xlim(0, max(tope, max(r["Vu"] for r in resumen)) * 1.38)
    ax.set_xlabel("Fuerza cortante en el plano (kN)")
    ax.set_ylabel(r"Altura $z$ (m)")
    ax.set_title("(a) En altura", loc="left")
    rejilla(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    alto = 0.26
    for r in resumen:
        s = r["piso"]
        for k, (clave, color, rayado) in enumerate((("Vua", "w", "////"), ("Vu", "0.35", None),
                                                    ("phiVn", "0.8", None))):
            bx.barh(s + (1 - k) * alto, r[clave], height=alto, color=color, edgecolor="k", lw=0.6,
                    hatch=rayado)
            bx.text(r[clave], s + (1 - k) * alto, f" {r[clave]:.0f}", va="center", fontsize=7.5)
    bx.set_yticks(list(pisos))
    bx.set_yticklabels([f"Piso {s + 1}" for s in pisos])
    bx.yaxis.set_minor_locator(NullLocator())
    bx.set_xlim(0, tope * 1.18)
    bx.set_xlabel("Fuerza cortante en el plano (kN)")
    bx.set_title("(b) Por entrepiso", loc="left")
    bx.legend(handles=[Patch(fc="w", ec="k", hatch="////", label=r"$|V_{ua}|$"),
                       Patch(fc="0.35", ec="k", label=r"$|V_u|$"),
                       Patch(fc="0.8", ec="k", label=r"$\phi V_n$")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    bx.grid(True, axis="x", color="0.82", ls=":", lw=0.6)
    bx.set_axisbelow(True)
    titulo(fig, r"fuerza cortante de diseño, $V_u = V_{ua}\,(M_n/M_{ua})$ (E.060 §21.9.5.5)")
    guardar(fig, "cortante_en_altura")

    # --- 5 y 6. envolventes del MEF en X y en Y -------------------------------------------
    # La envolvente total: el máximo y el mínimo de todos los estados de diseño
    # en cada cota. Los momentos, en el punto de referencia de ETABS, como los
    # dan ETABS y el MEF (sin el traslado P·d al centroide de la sección real,
    # que solo se usa en el diseño): así la figura se compara directamente con
    # ETABS. Se acota en la base y el tope de cada entrepiso (el momento salta
    # en las losas) y el cortante, constante en el entrepiso, a media altura.
    por_cota = [dis.estados(z, s) for z, s, _ in cotas]
    indice = {(round(z, 9), s): i for i, (z, s, _) in enumerate(cotas)}
    for eje, campo_M, campo_V, simbolo, plano, d in (
            ("X", "M", "V", "x", "en el plano del muro", sec.xe - sec.xc),
            ("Y", "My", "Vy", "y", "fuera del plano del muro", sec.ye - sec.yc)):

        def valor(x, campo, d=d, campo_M=campo_M):
            """El valor del estado `x`; el momento, sin el traslado al centroide."""
            v = getattr(x, campo)
            return v - x.P * d if campo == campo_M else v

        fig, (am, av) = plt.subplots(1, 2, figsize=(7.4, 6.0), sharey=True)
        for ax, campo in ((am, campo_M), (av, campo_V)):
            vmax = np.array([max(valor(x, campo) for x in est) for est in por_cota])
            vmin = np.array([min(valor(x, campo) for x in est) for est in por_cota])
            ax.fill_betweenx(zc, vmin, vmax, color="0.9", lw=0, zorder=0)
            ax.plot(vmax, zc, color="k", lw=1.3, label="Envolvente máxima")
            ax.plot(vmin, zc, color="k", lw=1.3, ls="--", label="Envolvente mínima")
            ax.axvline(0, color="0.4", lw=0.5)
            for s in pisos:
                i0 = indice[(round(zb[s], 9), s)]
                i1 = indice[(round(zt[s], 9), s)]
                if ax is am:
                    puntos = [(i0, (0, 2), "bottom"), (i1, (0, -2), "top")]
                else:
                    puntos = [(i0, None, "center")]
                for i, desplaza, va in puntos:
                    z = (zb[s] + zt[s]) / 2 if desplaza is None else zc[i]
                    for v, lado in ((vmax[i], 1), (vmin[i], -1)):
                        dy = 0 if desplaza is None else desplaza[1]
                        ax.plot(v, z, "o", ms=2.5, color="k")
                        ax.annotate(num(v, 2), xy=(v, z), xytext=(4 * lado, dy),
                                    textcoords="offset points", ha="left" if lado > 0 else "right",
                                    va=va, fontsize=7.5)
            ancho = max(vmax.max(), 0.0) - min(vmin.min(), 0.0) or 1.0
            ax.set_xlim(min(vmin.min(), 0.0) - 0.32 * ancho, max(vmax.max(), 0.0) + 0.32 * ancho)
            rejilla(ax)
        eje_de_niveles(am)
        nombres_de_pisos(av)
        am.set_xlabel(rf"Momento flector $M_{simbolo}$ (kN$\cdot$m)")
        av.set_xlabel(rf"Fuerza cortante $V_{simbolo}$ (kN)")
        am.set_ylabel(r"Altura $z$ (m)")
        am.set_title(f"(a) Momento flector $M_{simbolo}$", loc="left")
        av.set_title(f"(b) Fuerza cortante $V_{simbolo}$" + (" (del análisis)" if eje == "X" else ""),
                     loc="left")
        am.legend(loc="upper center", bbox_to_anchor=(1.0, -0.1), ncol=2, fontsize=8)
        if abs(d) > 1e-9:
            am.text(0.0, -0.2, rf"$M_{simbolo}$ como lo da ETABS (su punto de referencia). En el diseño "
                    rf"se lleva al centroide de la sección real: $M_{simbolo}$ + $P$·({num(d, 4)} m).",
                    transform=am.transAxes, fontsize=7.5, style="italic", va="top")
        titulo(fig, f"envolventes de diseño en la dirección {eje} ({plano})")
        guardar(fig, f"envolventes_{eje}")
