"""
Presentation figures of the PHANGS galaxies in Euclid DR1, built from the bare PNGs that make_phangs_rgb.py writes.

  conda run --no-capture-output -n astro python make_poster.py            # all variants + single-galaxy cards
  conda run --no-capture-output -n astro python make_poster.py --only grid_black_colour cards

Outputs in output/figures/ (PNG + JPG for every multipanel, PDF for the main ones; cards in output/figures/cards/).

Overlays from the PHANGS sample table (v1.6; Leroy et al. 2021): R_eff (dashed) and R_25 (dotted) as ellipses
projected with each galaxy's inclination and position angle, a kpc scale bar from the distance, and Hubble type,
distance, stellar mass and SFR. Panels run along the Hubble sequence (T-type). North is up, east is left.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.table import Table
from PIL import Image

import euclid_tools as et
from make_phangs_rgb import SAMPLE_TABLE, OUT_FITS, OUT_PNG, SUMMARY_TABLE

FIG_DIR = et.ROOT / 'output' / 'figures'
EXCLUDE = {'ngc3031'}                 # only 31% of its field has data
HERO = 'ngc1566'                      # the big tile in the hero layouts
PANEL_PX = 1400                       # resolution each grid panel is resampled to

# Avenir Next for text; DejaVu Sans supplies the glyphs Avenir lacks (★, ☉, arrows)
plt.rcParams.update({'font.family': ['Avenir Next', 'DejaVu Sans'], 'font.size': 10, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Avenir Next', 'mathtext.it': 'Avenir Next:italic',
                     'mathtext.bf': 'Avenir Next:bold'})

BAND_KEY = (('I$_E$', '0.7 µm', '#5c8cff'), ('Y$_E$', '1.1 µm', '#62d36b'), ('H$_E$', '1.8 µm', '#ff6b5c'))
LS_REFF, LS_R25 = (0, (5, 3)), (0, (1, 2.0))
CREDIT = 'Euclid DR1 MER mosaics (ESA / Euclid Consortium)'
SAMPLE_CREDIT = 'PHANGS sample table v1.6 (Leroy et al. 2021)'
REFF_TXT = 'R$_{\\rm eff}$  stellar half-light radius (3.4 µm)'
R25_TXT = 'R$_{25}$  B-band 25 mag arcsec$^{-2}$ isophote'


def overlay_colours(light_sky):
    """(R_eff, R_25, halo) colours that read on the sky behind them."""
    if light_sky:
        return '#0a6fb8', '#d06a00', '#111111'
    return '#6fd6ff', '#ffc857', '#000000'


# ------------------------------------------------------------------------------------------------ data

def pretty_name(n):
    n = n.lower()
    for pre in ('ngc', 'ic', 'eso'):
        if n.startswith(pre):
            rest = n[len(pre):]
            return f'{pre.upper()} {rest.lstrip("0") if rest.isdigit() else rest}'
    return n.upper()


def load_galaxies():
    summ = Table.read(SUMMARY_TABLE)
    ok = [str(n) for n in summ['name'][summ['status'] == 'ok'] if str(n) not in EXCLUDE]
    t = Table.read(SAMPLE_TABLE)
    gals = []
    for r in t:
        n = str(r['name'])
        if n not in ok:
            continue
        hdr = fits.getheader(OUT_FITS / f'{n}_euclid_VIS.fits')

        def val(c):
            v = r[c]
            return float(v) if v is not np.ma.masked and np.isfinite(float(v)) else np.nan
        gals.append(dict(
            name=n, label=pretty_name(n), half=hdr['NAXIS1'] * hdr['PIXSCALE'] / 2,
            morph=str(r['morph_string']).strip(), ttype=val('morph_t'), dist=val('dist'),
            incl=val('orient_incl'), pa=val('orient_posang'), reff=val('size_reff'), r25=val('size_r25'),
            mstar=val('props_mstar'), sfr=val('props_sfr')))
    gals.sort(key=lambda g: (g['ttype'], g['name']))
    return gals


_cache = {}


def panel_image(g, style, px):
    key = (g['name'], style, px)
    if key not in _cache:
        im = Image.open(OUT_PNG / f'{g["name"]}_euclid_{style}_noaxes.png').convert('RGB')
        if im.size[0] > px:
            im = im.resize((px, px), Image.LANCZOS)
        _cache[key] = np.asarray(im)
    return _cache[key]


# ------------------------------------------------------------------------------------------------ one panel

def nice_bar(kpc_per_arcsec, half):
    """A common 5 kpc bar wherever it spans 10-45% of the panel width, otherwise the nearest round length that does."""
    width_kpc = 2 * half * kpc_per_arcsec
    if 0.10 <= 5 / width_kpc <= 0.45:
        return 5
    for L in (10, 2, 20, 1, 50, 0.5):
        if 0.10 <= L / width_kpc <= 0.45:
            return L
    return 5


def _fade_band(ax, top, colour, frac, alpha):
    """Soft gradient band along the top or bottom of the panel, behind the text."""
    n = 64
    ramp = np.linspace(alpha, 0, n) ** 1.3
    rgba = np.zeros((n, 1, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(colour)
    rgba[:, 0, 3] = ramp if top else ramp
    ext = (0, 1, 1 - frac, 1) if top else (0, 1, 0, frac)
    ax.imshow(rgba if top else rgba[::-1], extent=ext, transform=ax.transAxes, aspect='auto', zorder=8,
              origin='upper', interpolation='bilinear')


def stats_line(g):
    parts = []
    if np.isfinite(g['mstar']):
        parts.append(f"log M★ {np.log10(g['mstar']):.1f}")
    if np.isfinite(g['sfr']):
        parts.append(f"SFR {g['sfr']:.1f}" if g['sfr'] >= 1 else f"SFR {g['sfr']:.2f}")
    return '   '.join(parts)


def draw_panel(ax, g, style, overlays=True, info='full', fs=1.0, px=PANEL_PX, compass=False, bands=True):
    """
    One galaxy in ax. Data coordinates are arcsec offsets from the centre with east = +x (shown on the left),
    so an ellipse at position angle PA (east of north) has matplotlib angle 90 - PA.
    info: 'full' (name, type & distance, M*/SFR), 'short' (name, type & distance), 'name'.
    """
    h = g['half']
    ax.imshow(panel_image(g, style, px), extent=(h, -h, -h, h), origin='upper', interpolation='lanczos', zorder=1)
    ax.set_xlim(h, -h)
    ax.set_ylim(-h, h)
    ax.set_axis_off()

    light = style == 'rgbgrey'
    txt = '#111111' if light else 'white'
    back = 'white' if light else 'black'
    c_reff, c_r25, halo = overlay_colours(light)
    eff = [pe.withStroke(linewidth=2.2 * fs, foreground=back, alpha=0.7)]

    if bands:
        _fade_band(ax, True, back, 0.20 if info == 'full' else 0.15, 0.62)
        _fade_band(ax, False, back, 0.12, 0.55)

    if overlays and np.isfinite(g['pa']):
        q = np.cos(np.radians(g['incl'])) if np.isfinite(g['incl']) else 1.0
        for R, c, ls, lw in ((g['r25'], c_r25, LS_R25, 1.9), (g['reff'], c_reff, LS_REFF, 1.6)):
            if np.isfinite(R):
                ax.add_patch(Ellipse((0, 0), 2 * R, 2 * R * q, angle=90.0 - g['pa'], fill=False, ec=c, ls=ls,
                                     lw=lw * fs, zorder=4,
                                     path_effects=[pe.Stroke(linewidth=(lw + 1.6) * fs, foreground=halo,
                                                             alpha=0.35 if light else 0.3), pe.Normal()]))

    # text block, top left: stacked in points so spacing scales with the font
    name_fs, sub_fs = 14.5 * fs, 10.0 * fs
    ax.annotate(g['label'], xy=(0.045, 0.955), xycoords='axes fraction', color=txt, fontsize=name_fs,
                fontweight='bold', va='top', ha='left', path_effects=eff, zorder=10)
    lines = []
    if info in ('full', 'short'):
        lines.append(f"{g['morph']}   {g['dist']:.1f} Mpc" if g['morph'] else f"{g['dist']:.1f} Mpc")
    if info == 'full':
        s = stats_line(g)
        if s:
            lines.append(s)
    for i, line in enumerate(lines):
        ax.annotate(line, xy=(0.045, 0.955), xycoords='axes fraction',
                    xytext=(0, -(name_fs * 1.25 + i * sub_fs * 1.35)), textcoords='offset points',
                    color=txt, fontsize=sub_fs, va='top', ha='left', alpha=0.92, path_effects=eff, zorder=10)

    # kpc scale bar, bottom left
    if np.isfinite(g['dist']):
        kpa = g['dist'] * 1e3 * np.pi / 180 / 3600
        L = nice_bar(kpa, h)
        x0, y0 = h * 0.91, -h * 0.88
        ax.plot([x0, x0 - L / kpa], [y0, y0], color=txt, lw=2.4 * fs, solid_capstyle='butt', zorder=10,
                path_effects=[pe.Stroke(linewidth=4.2 * fs, foreground=back, alpha=0.5), pe.Normal()])
        ax.annotate(f'{L:g} kpc', xy=(x0 - L / kpa / 2, y0), xycoords='data', xytext=(0, 3.5 * fs),
                    textcoords='offset points', color=txt, fontsize=sub_fs, ha='center', va='bottom',
                    path_effects=eff, zorder=10)

    if compass:
        draw_compass(ax, txt, back, fs)


def draw_compass(ax, txt, back, fs):
    """N/E arrows, bottom right (east is to the left)."""
    ox, oy, L = 0.93, 0.07, 0.075
    for dx, dy, lab in ((0, 1, 'N'), (-1, 0, 'E')):
        ax.annotate('', xy=(ox + dx * L, oy + dy * L), xytext=(ox, oy), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color=txt, lw=1.4 * fs, mutation_scale=9 * fs,
                                    shrinkA=0, shrinkB=0), zorder=10)
        ax.annotate(lab, xy=(ox + dx * L * 1.38, oy + dy * L * 1.32), xycoords='axes fraction', color=txt,
                    fontsize=9.5 * fs, ha='center', va='center', zorder=10,
                    path_effects=[pe.withStroke(linewidth=2 * fs, foreground=back, alpha=0.7)])


# ------------------------------------------------------------------------------------------------ keys / text rows

class Row:
    """Lay out legend items left-to-right in figure inches, advancing by measured text widths."""

    def __init__(self, fig, x_in, y_in, colour, fontsize):
        self.fig, self.x, self.y, self.c, self.fs = fig, x_in, y_in, colour, fontsize
        self.W, self.H = fig.get_figwidth(), fig.get_figheight()
        self.r = fig.canvas.get_renderer()

    def text(self, s, colour=None, gap=0.32, weight='normal', size=None):
        t = self.fig.text(self.x / self.W, self.y / self.H, s, color=colour or self.c, fontsize=size or self.fs,
                          va='center', ha='left', fontweight=weight)
        w = t.get_window_extent(self.r).width / self.fig.dpi
        self.x += w + gap
        return t

    def swatch(self, colour, size=0.15, gap=0.08):
        self.fig.patches.append(Rectangle((self.x / self.W, (self.y - size / 2) / self.H), size / self.W,
                                          size / self.H, transform=self.fig.transFigure, color=colour))
        self.x += size + gap

    def line(self, colour, ls, length=0.5, lw=2.0, gap=0.1):
        self.fig.lines.append(Line2D([self.x / self.W, (self.x + length) / self.W], [self.y / self.H] * 2,
                                     transform=self.fig.transFigure, color=colour, ls=ls, lw=lw))
        self.x += length + gap


def footer(fig, x0, x1, y_top, theme, overlays, style, fs=12):
    fg = 'white' if theme == 'dark' else '#111111'
    mid = '#a8a8a8' if theme == 'dark' else '#555555'
    c_reff, c_r25, _ = overlay_colours(style == 'rgbgrey')      # same colours as drawn in the panels
    W, H = fig.get_figwidth(), fig.get_figheight()
    rows = [y_top - 0.12, y_top - 0.50]

    r = Row(fig, x0, rows[0], fg, fs)
    r.text('Colour', mid, gap=0.25, weight='medium')
    for lab, wl, c in BAND_KEY:
        r.swatch(c)
        r.text(f'{lab} {wl}', gap=0.35)
    if style == 'rgbgrey':
        r.text('·   faint light: inverted greyscale', mid, gap=0.3)
    fig.text(x1 / W, rows[0] / H, CREDIT, color=mid, fontsize=fs - 1.5, ha='right', va='center')

    r = Row(fig, x0, rows[1], fg, fs)
    if overlays:
        r.text('Overlays', mid, gap=0.25, weight='medium')
        r.line(c_reff, LS_REFF)
        r.text(REFF_TXT, gap=0.45)
        r.line(c_r25, LS_R25)
        r.text(R25_TXT, gap=0.45)
        r.text('log M★ [M☉]  ·  SFR [M☉ yr⁻¹]', mid, gap=0.4)
    r.text('N up, E left', mid)
    fig.text(x1 / W, rows[1] / H, SAMPLE_CREDIT if overlays else 'Sample: PHANGS (v1.6)', color=mid,
             fontsize=fs - 1.5, ha='right', va='center')


def title_block(fig, x0, x1, y_top, n, fg, mid, big=36):
    W, H = fig.get_figwidth(), fig.get_figheight()
    fig.text(x0 / W, (y_top - 0.42) / H, 'PHANGS  ×  Euclid', color=fg, fontsize=big, fontweight='bold',
             va='center', ha='left')
    fig.text(x0 / W, (y_top - 1.0) / H, f'{n} nearby PHANGS galaxies in Euclid DR1  ·  '
             'ordered along the Hubble sequence, S0 → Scd', color=mid, fontsize=15, va='center', ha='left')
    fig.text(x1 / W, (y_top - 0.42) / H, 'VIS  +  NISP', color=fg, fontsize=22, va='center', ha='right',
             fontweight='medium')
    fig.text(x1 / W, (y_top - 1.0) / H, 'I$_E$, Y$_E$, H$_E$   ·   0.7 – 1.8 µm', color=mid, fontsize=15,
             va='center', ha='right')


# ------------------------------------------------------------------------------------------------ layouts

def _frame(ax, colour):
    for s in ('left', 'right', 'top', 'bottom'):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(colour)
        ax.spines[s].set_linewidth(0.8)
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])


def make_grid(gals, style='rgb', theme='dark', overlays=True, info='full', ncol=4, panel_in=4.2, gap_in=None,
              title=True):
    n = len(gals)
    nrow = int(np.ceil(n / ncol))
    gap_in = gap_in if gap_in is not None else (0.07 if theme == 'dark' else 0.16)
    side, top, bottom = 0.5, (1.65 if title else 0.4), (1.25 if overlays else 0.9)
    W = 2 * side + ncol * panel_in + (ncol - 1) * gap_in
    H = top + bottom + nrow * panel_in + (nrow - 1) * gap_in
    bg = 'black' if theme == 'dark' else 'white'
    fg = 'white' if theme == 'dark' else '#111111'
    mid = '#a8a8a8' if theme == 'dark' else '#555555'
    fig = plt.figure(figsize=(W, H), facecolor=bg, dpi=250)
    fs = panel_in / 3.4
    for i, g in enumerate(gals):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([(side + c * (panel_in + gap_in)) / W, (H - top - (r + 1) * panel_in - r * gap_in) / H,
                           panel_in / W, panel_in / H])
        draw_panel(ax, g, style, overlays=overlays, info=info, fs=fs)
        if theme != 'dark':
            _frame(ax, '#9a9a9a')
    if title:
        title_block(fig, side, W - side, H - 0.15, n, fg, mid)
    footer(fig, side, W - side, bottom - 0.12, theme, overlays, style, fs=12.5)
    return fig


def make_hero(gals, style='rgb', overlays=True, info='short', ncol=5, panel_in=3.6, gap_in=0.07, hero=HERO):
    """Hero layout: one galaxy as a 2x2 tile top-left, the other 11 around it (5 x 3 cells)."""
    big = next(g for g in gals if g['name'] == hero)
    rest = [g for g in gals if g['name'] != hero]
    cells = [(r, c) for r in range(3) for c in range(ncol) if not (r < 2 and c < 2)]
    side, top, bottom = 0.5, 1.65, 1.25 if overlays else 0.9
    W = 2 * side + ncol * panel_in + (ncol - 1) * gap_in
    H = top + bottom + 3 * panel_in + 2 * gap_in
    fig = plt.figure(figsize=(W, H), facecolor='black', dpi=250)
    fs = panel_in / 3.4

    def rect(r, c, span=1):
        s = span * panel_in + (span - 1) * gap_in
        return [(side + c * (panel_in + gap_in)) / W, (H - top - r * (panel_in + gap_in) - s) / H, s / W, s / H]
    ax = fig.add_axes(rect(0, 0, 2))
    draw_panel(ax, big, style, overlays=overlays, info='full', fs=fs * 1.55, px=2600, compass=True)
    for g, (r, c) in zip(rest, cells):
        draw_panel(fig.add_axes(rect(r, c)), g, style, overlays=overlays, info=info, fs=fs)
    title_block(fig, side, W - side, H - 0.15, len(gals), 'white', '#a8a8a8')
    footer(fig, side, W - side, bottom - 0.12, 'dark', overlays, style, fs=12.5)
    return fig


def mass_sfr_inset(fig, rect, gals, fg='white', mid='#a8a8a8'):
    """log SFR vs log M*, points coloured by T-type, labelled."""
    ax = fig.add_axes(rect, facecolor='none')
    m = np.array([np.log10(g['mstar']) for g in gals])
    s = np.array([np.log10(g['sfr']) for g in gals])
    t = np.array([g['ttype'] for g in gals])
    # main sequence used for the PHANGS Delta-MS values (Leroy et al. 2019); reproduces props_deltams exactly
    xx = np.linspace(9.0, 11.4, 10)
    ms = 0.68 * (xx - 10) - 0.17
    ax.plot(xx, ms, color=mid, lw=1, ls='--', alpha=0.7)
    ax.fill_between(xx, ms - 0.4, ms + 0.4, color=mid, alpha=0.09, lw=0)
    sc = ax.scatter(m, s, c=t, cmap='RdYlBu', vmin=-1, vmax=8, s=60, edgecolor='white', linewidth=0.6, zorder=3)
    nudge = {'ngc1559': (-30, 6), 'ngc1792': (6, -12), 'ngc1546': (-30, 5), 'ngc1326': (6, 4), 'ic1954': (-38, 6),
             'ngc1566': (6, 3), 'ngc1365': (-30, -4)}
    for g, x, y in zip(gals, m, s):
        ax.annotate(g['label'].replace('NGC ', ''), (x, y), xytext=nudge.get(g['name'], (6, 3)),
                    textcoords='offset points', color=fg, fontsize=9, alpha=0.9)
    ax.set_xlabel('log M★  [M☉]', color=fg, fontsize=11)
    ax.set_ylabel('log SFR  [M☉ yr⁻¹]', color=fg, fontsize=11)
    ax.tick_params(colors=mid, labelsize=9.5)
    for sp in ax.spines.values():
        sp.set_color(mid)
    ax.set_xlim(9.1, 11.35)
    ax.set_ylim(-1.2, 1.55)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.2, aspect=30)
    cb.set_label('T-type  (S0 → Scd)', color=fg, fontsize=10)
    cb.ax.tick_params(colors=mid, labelsize=9)
    cb.outline.set_edgecolor(mid)
    ax.text(0.03, 0.95, 'dashed: main sequence (Leroy+19) ± 0.4 dex', transform=ax.transAxes, color=mid,
            fontsize=9, va='top')
    return ax


def make_slide(gals, style='rgb', overlays=True, info='short', ncol=4):
    """16:9 slide (3840 x 2160 px): 4x3 grid on the left; title, key and an M*-SFR inset on the right."""
    W, H = 19.2, 10.8
    m, gap = 0.3, 0.05
    nrow = int(np.ceil(len(gals) / ncol))
    panel_in = (H - 2 * m - (nrow - 1) * gap) / nrow
    fig = plt.figure(figsize=(W, H), facecolor='black', dpi=200)
    fs = panel_in / 3.4
    for i, g in enumerate(gals):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([(m + c * (panel_in + gap)) / W, (H - m - (r + 1) * panel_in - r * gap) / H,
                           panel_in / W, panel_in / H])
        draw_panel(ax, g, style, overlays=overlays, info=info, fs=fs)
    x0_in = m + ncol * panel_in + (ncol - 1) * gap + 0.5
    x0 = x0_in / W
    fig.text(x0, 1 - 0.85 / H, 'PHANGS × Euclid', color='white', fontsize=34, fontweight='bold', va='center')
    fig.text(x0, 1 - 1.5 / H, f'{len(gals)} nearby galaxies in Euclid DR1', color='#c8c8c8', fontsize=17,
             va='center')
    r = Row(fig, x0_in, H - 2.2, 'white', 14)
    for lab, wl, c in BAND_KEY:
        r.swatch(c, size=0.17)
        r.text(f'{lab} {wl}', gap=0.3)
    c_reff, c_r25, _ = overlay_colours(style == 'rgbgrey')
    if overlays:
        r = Row(fig, x0_in, H - 2.65, 'white', 14)
        r.line(c_reff, LS_REFF, length=0.45)
        r.text('R$_{\\rm eff}$', gap=0.45)
        r.line(c_r25, LS_R25, length=0.45)
        r.text('R$_{25}$', gap=0.45)
        r.text('N up, E left', '#a8a8a8')
    mass_sfr_inset(fig, [(x0_in + 0.75) / W, 2.55 / H, (W - x0_in - 1.2) / W, 4.6 / H], gals)
    fig.text(x0, 0.95 / H, 'Panels ordered by Hubble type; bars give physical scale.', color='#a8a8a8',
             fontsize=12.5, va='bottom')
    fig.text(x0, 0.35 / H, f'{CREDIT}\n{SAMPLE_CREDIT}', color='#8a8a8a', fontsize=11.5, va='bottom',
             linespacing=1.45)
    return fig


def make_card(g, style='rgb'):
    """Single galaxy: large image with overlays, compass and a self-contained key underneath."""
    P, strip = 8.0, 1.3
    fig = plt.figure(figsize=(P, P + strip), facecolor='black', dpi=300)
    ax = fig.add_axes([0, strip / (P + strip), 1, P / (P + strip)])
    draw_panel(ax, g, style, overlays=True, info='full', fs=1.75, px=2400, compass=True)
    c_reff, c_r25, _ = overlay_colours(style == 'rgbgrey')
    r = Row(fig, 0.3, strip - 0.3, 'white', 11)
    for lab, wl, c in BAND_KEY:
        r.swatch(c, size=0.14)
        r.text(f'{lab} {wl}', gap=0.28)
    if style == 'rgbgrey':
        r.text('·  faint light: inverted greyscale', '#a8a8a8')
    r = Row(fig, 0.3, strip - 0.64, 'white', 10.5)
    r.line(c_reff, LS_REFF, length=0.4)
    r.text('R$_{\\rm eff}$ half-light (3.4 µm)', gap=0.35)
    r.line(c_r25, LS_R25, length=0.4)
    r.text('R$_{25}$ (B 25 mag arcsec$^{-2}$)', gap=0.35)
    r.text('M★ [M☉], SFR [M☉ yr⁻¹]', '#a8a8a8')
    fig.text(0.3 / P, (strip - 0.98) / (P + strip), f'{CREDIT}   ·   {SAMPLE_CREDIT}   ·   N up, E left',
             color='#9a9a9a', fontsize=9.5, ha='left', va='center')
    return fig


# ------------------------------------------------------------------------------------------------ main

ALL = ('png', 'jpg', 'pdf')
VARIANTS = {
    # name: (builder, kwargs, formats)
    'grid_black_colour': (make_grid, dict(style='rgb'), ALL),
    'grid_black_colourgrey': (make_grid, dict(style='rgbgrey'), ALL),
    'grid_black_colour_clean': (make_grid, dict(style='rgb', overlays=False, info='name'), ALL),
    'grid_black_colourgrey_clean': (make_grid, dict(style='rgbgrey', overlays=False, info='name'), ('png', 'jpg')),
    'grid_white_colourgrey': (make_grid, dict(style='rgbgrey', theme='light'), ('png', 'jpg', 'pdf')),
    'hero_black_colour': (make_hero, dict(style='rgb'), ALL),
    'hero_black_colourgrey': (make_hero, dict(style='rgbgrey'), ('png', 'jpg')),
    'grid_wide_black_colour': (make_grid, dict(style='rgb', ncol=6, panel_in=3.8, info='short'), ('png', 'jpg')),
    'slide16x9_colour': (make_slide, dict(style='rgb'), ('png', 'pdf')),
    'slide16x9_colourgrey': (make_slide, dict(style='rgbgrey'), ('png', 'pdf')),
}


def save(fig, stem, formats, dpi=None):
    out_dir = FIG_DIR / stem.rsplit('/', 1)[0] if '/' in stem else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in formats:
        out = FIG_DIR / f'{stem}.{f}'
        kw = dict(dpi=dpi or fig.dpi, facecolor=fig.get_facecolor())
        if f == 'jpg':
            kw['pil_kwargs'] = {'quality': 93}
        fig.savefig(out, **kw)
        print(f'-> {out.relative_to(et.ROOT)}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='variant names (default: all), and/or "cards"')
    args = ap.parse_args()
    gals = load_galaxies()
    print('galaxies:', ', '.join(g['label'] for g in gals))
    for name in args.only or list(VARIANTS) + ['cards']:
        if name == 'cards':
            for g in gals:
                for style in ('rgb', 'rgbgrey'):
                    save(make_card(g, style), f'cards/{g["name"]}_{style}_card', ('png', 'jpg'))
            continue
        builder, kw, formats = VARIANTS[name]
        save(builder(gals, **kw), name, formats)


if __name__ == '__main__':
    main()
