"""
Presentation figures of the GECKOS galaxies in Euclid DR1, built from the bare PNGs that make_geckos_rgb.py writes.

  conda run --no-capture-output -n astro python make_geckos_poster.py
  conda run --no-capture-output -n astro python make_geckos_poster.py --only strips_black_colour cards

The galaxies are edge-on, so the main layouts rotate each image by its position angle until the disk is horizontal
and crop it to a 3:1 strip (±4 R_e along the disk where the field allows). A compass in each strip shows where north
and east went. Single-galaxy cards are north up, east left. Overlays from the GECKOS master catalogue: R_e (r band)
as an ellipse with the catalogue ellipticity, kpc bar from d_mpc, log M* and log SFR. Strips are ordered by M*.
Output: output_geckos/figures/ (cards in cards/).
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.table import Table
from PIL import Image

import euclid_tools as et
import make_poster as mpo          # also sets the fonts
from make_poster import Row, BAND_KEY, LS_REFF, CREDIT, overlay_colours, nice_bar, _fade_band, _frame, pretty_name
import make_geckos_rgb as mg

FIG_DIR = mg.OUT / 'figures'
mpo.FIG_DIR = FIG_DIR              # mpo.save writes here
mpo.OUT_PNG = mg.OUT_PNG           # mpo.draw_panel (cards) reads the GECKOS PNGs
EXCLUDE = set()
STRIP_ASPECT = 3.0                 # strip width / height
STRIP_RE = 4.0                     # strip half-length in R_e (shortened where the rotated field runs out)
STRIP_PX = 3000                    # strip width the image is resampled to
SAMPLE_CREDIT = 'GECKOS master catalogue (v2026-02-09)'
REFF_TXT = 'R$_{\\rm e}$  r-band effective radius'


# ------------------------------------------------------------------------------------------------ data

def load_galaxies():
    summ = Table.read(mg.SUMMARY_TABLE)
    ok = [str(n) for n in summ['name'][summ['status'] == 'ok'] if str(n) not in EXCLUDE]
    gals = []
    for r in Table.read(mg.SAMPLE_TABLE):
        n = mg.gal_name(r)
        if n not in ok:
            continue
        hdr = fits.getheader(mg.OUT_FITS / f'{n}_euclid_VIS.fits')
        ell = float(r['ellip_e_r'])
        gals.append(dict(
            name=n, label=pretty_name(n), half=hdr['NAXIS1'] * hdr['PIXSCALE'] / 2, morph='',
            dist=float(r['d_mpc']), pa=float(r['PA']), ellip=ell, reff=float(r['R_e_r']), r25=np.nan,
            # inclination that reproduces the catalogue axis ratio, so mpo.draw_panel draws the same R_e ellipse
            incl=np.degrees(np.arccos(1 - ell)),
            lmstar=float(r['lmstar']), lsfr=float(r['lsfr']),
            mstar=10 ** float(r['lmstar']), sfr=10 ** float(r['lsfr'])))
    gals.sort(key=lambda g: (-g['lmstar'], g['name']))
    return gals


def stats_line(g):
    return f"log M★ {g['lmstar']:.1f}   log SFR {g['lsfr']:+.2f}"


# ------------------------------------------------------------------------------------------------ strip panel

def strip_geometry(g):
    """Rotation (deg, counter-clockwise on the displayed image) that makes the disk horizontal, and the strip
    half-length/half-height (arcsec) that stay inside the rotated square field."""
    rot = -((g['pa'] + 90.0) % 180.0)
    if rot < -90:
        rot += 180.0                                   # smallest rotation; the disk is symmetric
    t = np.radians(rot)
    c, s = abs(np.cos(t)), abs(np.sin(t))
    k = 1 / STRIP_ASPECT
    a_max = g['half'] / max(c + s * k, s + c * k)
    a = min(STRIP_RE * g['reff'], a_max) * 0.995
    return rot, a, a / STRIP_ASPECT


_cache = {}


def strip_image(g, style):
    key = (g['name'], style)
    if key not in _cache:
        rot, a, b = strip_geometry(g)
        im = Image.open(mg.OUT_PNG / f'{g["name"]}_euclid_{style}_noaxes.png').convert('RGB')
        im = im.rotate(rot, resample=Image.BICUBIC, expand=False)
        npx = im.size[0]
        pix = 2 * g['half'] / npx                       # arcsec per pixel
        cx = cy = npx / 2
        box = (int(round(cx - a / pix)), int(round(cy - b / pix)), int(round(cx + a / pix)), int(round(cy + b / pix)))
        im = im.crop(box)
        if im.size[0] > STRIP_PX:
            im = im.resize((STRIP_PX, int(round(STRIP_PX * im.size[1] / im.size[0]))), Image.LANCZOS)
        _cache[key] = (np.asarray(im), a, b)
    return _cache[key]


def draw_strip(ax, g, style, overlays=True, fs=1.0, compass=True):
    """One rotated galaxy. Data coordinates: arcsec along the disk (x) and across it (y)."""
    img, a, b = strip_image(g, style)
    ax.imshow(img, extent=(-a, a, -b, b), origin='upper', interpolation='lanczos', zorder=1)
    ax.set_xlim(-a, a)
    ax.set_ylim(-b, b)
    ax.set_aspect('equal')
    ax.set_axis_off()

    light = style == 'rgbgrey'
    txt = '#111111' if light else 'white'
    back = 'white' if light else 'black'
    c_reff, _, halo = overlay_colours(light)
    eff = [pe.withStroke(linewidth=2.2 * fs, foreground=back, alpha=0.7)]
    _fade_band(ax, True, back, 0.30, 0.55)
    _fade_band(ax, False, back, 0.22, 0.5)

    if overlays and np.isfinite(g['reff']):
        R = g['reff']
        ax.add_patch(Ellipse((0, 0), 2 * R, 2 * R * (1 - g['ellip']), fill=False, ec=c_reff, ls=LS_REFF,
                             lw=1.6 * fs, zorder=4,
                             path_effects=[pe.Stroke(linewidth=3.2 * fs, foreground=halo, alpha=0.3), pe.Normal()]))

    # text, top left: name on one line, distance and properties after it
    name_fs, sub_fs = 15 * fs, 10.5 * fs
    t = ax.annotate(g['label'], xy=(0.015, 0.94), xycoords='axes fraction', color=txt, fontsize=name_fs,
                    fontweight='bold', va='top', ha='left', path_effects=eff, zorder=10)
    sub = f"{g['dist']:.1f} Mpc     {stats_line(g)}" if overlays else f"{g['dist']:.1f} Mpc"
    ax.annotate(sub, xy=(1, 0), xycoords=t, xytext=(12 * fs, 1.2 * fs), textcoords='offset points', color=txt,
                fontsize=sub_fs, va='bottom', ha='left', alpha=0.92, path_effects=eff, zorder=10)

    # kpc scale bar, bottom left
    kpa = g['dist'] * 1e3 * np.pi / 180 / 3600
    L = nice_bar(kpa, b * STRIP_ASPECT / 2)       # sized against the strip length
    x0, y0 = -a + 0.03 * 2 * a, -b * 0.80
    ax.plot([x0, x0 + L / kpa], [y0, y0], color=txt, lw=2.4 * fs, solid_capstyle='butt', zorder=10,
            path_effects=[pe.Stroke(linewidth=4.2 * fs, foreground=back, alpha=0.5), pe.Normal()])
    ax.annotate(f'{L:g} kpc', xy=(x0 + L / kpa / 2, y0), xycoords='data', xytext=(0, 3.5 * fs),
                textcoords='offset points', color=txt, fontsize=sub_fs, ha='center', va='bottom',
                path_effects=eff, zorder=10)
    if compass:
        draw_rotated_compass(ax, g, a, b, txt, back, fs)


def draw_rotated_compass(ax, g, a, b, txt, back, fs):
    """N and E arrows, bottom right, rotated with the image (east is left before rotation)."""
    rot, _, _ = strip_geometry(g)
    t = np.radians(rot)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    L = 0.26 * b
    o = np.array([a - 0.45 * b, -0.45 * b])
    for vec, lab in (((0, 1), 'N'), ((-1, 0), 'E')):
        d = R @ np.array(vec, float)
        ax.annotate('', xy=o + d * L, xytext=o, xycoords='data',
                    arrowprops=dict(arrowstyle='-|>', color=txt, lw=1.4 * fs, mutation_scale=9 * fs,
                                    shrinkA=0, shrinkB=0), zorder=10)
        ax.annotate(lab, xy=o + d * L * 1.4, xycoords='data', color=txt, fontsize=9.5 * fs, ha='center',
                    va='center', zorder=10, path_effects=[pe.withStroke(linewidth=2 * fs, foreground=back, alpha=0.7)])


# ------------------------------------------------------------------------------------------------ layouts

def key_rows(fig, x0, x1, y_top, theme, style, overlays=True, fs=12):
    fg = 'white' if theme == 'dark' else '#111111'
    mid = '#a8a8a8' if theme == 'dark' else '#555555'
    c_reff, _, _ = overlay_colours(style == 'rgbgrey')
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
        r.text('Overlay', mid, gap=0.25, weight='medium')
        r.line(c_reff, LS_REFF)
        r.text(REFF_TXT, gap=0.45)
        r.text('M★ [M☉]  ·  SFR [M☉ yr⁻¹]', mid, gap=0.45)
    r.text('disks rotated horizontal; arrows give N and E', mid)
    fig.text(x1 / W, rows[1] / H, SAMPLE_CREDIT, color=mid, fontsize=fs - 1.5, ha='right', va='center')


def title_block(fig, x0, x1, y_top, n, fg, mid, big=34):
    W, H = fig.get_figwidth(), fig.get_figheight()
    fig.text(x0 / W, (y_top - 0.42) / H, 'GECKOS  ×  Euclid', color=fg, fontsize=big, fontweight='bold',
             va='center', ha='left')
    fig.text(x0 / W, (y_top - 1.0) / H, f'{n} edge-on GECKOS galaxies in Euclid DR1', color=mid, fontsize=15,
             va='center', ha='left')
    fig.text(x1 / W, (y_top - 0.42) / H, 'VIS  +  NISP', color=fg, fontsize=22, va='center', ha='right',
             fontweight='medium')
    fig.text(x1 / W, (y_top - 1.0) / H, 'I$_E$, Y$_E$, H$_E$   ·   0.7 – 1.8 µm', color=mid, fontsize=15,
             va='center', ha='right')


def make_strips(gals, style='rgb', theme='dark', overlays=True, strip_w=15.0, gap_in=None, title=True):
    n = len(gals)
    strip_h = strip_w / STRIP_ASPECT
    gap_in = gap_in if gap_in is not None else (0.07 if theme == 'dark' else 0.16)
    side, top, bottom = 0.5, (1.65 if title else 0.4), 1.25
    W = 2 * side + strip_w
    H = top + bottom + n * strip_h + (n - 1) * gap_in
    bg = 'black' if theme == 'dark' else 'white'
    fg = 'white' if theme == 'dark' else '#111111'
    mid = '#a8a8a8' if theme == 'dark' else '#555555'
    fig = plt.figure(figsize=(W, H), facecolor=bg, dpi=250)
    fs = strip_h / 3.4
    for i, g in enumerate(gals):
        ax = fig.add_axes([side / W, (H - top - (i + 1) * strip_h - i * gap_in) / H, strip_w / W, strip_h / H])
        draw_strip(ax, g, style, overlays=overlays, fs=fs)
        if theme != 'dark':
            _frame(ax, '#9a9a9a')
    if title:
        title_block(fig, side, W - side, H - 0.15, n, fg, mid)
    key_rows(fig, side, W - side, bottom - 0.12, theme, style, overlays=overlays, fs=12.5)
    return fig


def make_slide(gals, style='rgb'):
    """16:9 slide (3840 x 2160 px): strips stacked on the left, title and key on the right."""
    W, H = 19.2, 10.8
    m, gap = 0.3, 0.08
    n = len(gals)
    strip_w = 13.2
    strip_h = strip_w / STRIP_ASPECT
    y_start = H / 2 + (n * strip_h + (n - 1) * gap) / 2
    fig = plt.figure(figsize=(W, H), facecolor='black', dpi=200)
    fs = strip_h / 3.4
    for i, g in enumerate(gals):
        ax = fig.add_axes([m / W, (y_start - (i + 1) * strip_h - i * gap) / H, strip_w / W, strip_h / H])
        draw_strip(ax, g, style, fs=fs)
    x0_in = m + strip_w + 0.45
    x0 = x0_in / W
    fig.text(x0, 1 - 0.95 / H, 'GECKOS\n× Euclid', color='white', fontsize=34, fontweight='bold', va='top',
             linespacing=1.1)
    fig.text(x0, 1 - 2.95 / H, f'{n} edge-on galaxies\nin Euclid DR1', color='#c8c8c8', fontsize=17, va='top',
             linespacing=1.3)
    y = H - 4.6
    for lab, wl, c in BAND_KEY:
        r = Row(fig, x0_in, y, 'white', 14)
        r.swatch(c, size=0.17)
        r.text(f'{lab} {wl}')
        y -= 0.42
    c_reff, _, _ = overlay_colours(style == 'rgbgrey')
    r = Row(fig, x0_in, y - 0.15, 'white', 14)
    r.line(c_reff, LS_REFF, length=0.45)
    r.text('R$_{\\rm e}$ (r band)')
    fig.text(x0, 1.55 / H, 'Disks rotated horizontal;\narrows give N and E.', color='#a8a8a8', fontsize=12.5,
             va='bottom', linespacing=1.4)
    fig.text(x0, 0.35 / H, f'{CREDIT}\n{SAMPLE_CREDIT}', color='#8a8a8a', fontsize=10, va='bottom',
             linespacing=1.45)
    return fig


def make_card(g, style='rgb'):
    """Single galaxy, north up: large image with R_e, compass and a self-contained key underneath."""
    P, strip = 8.0, 1.3
    fig = plt.figure(figsize=(P, P + strip), facecolor='black', dpi=300)
    ax = fig.add_axes([0, strip / (P + strip), 1, P / (P + strip)])
    mpo.draw_panel(ax, g, style, overlays=True, info='name', fs=1.75, px=2400, compass=True)
    ax.annotate(f"{g['dist']:.1f} Mpc     {stats_line(g)}", xy=(0.045, 0.955), xycoords='axes fraction',
                xytext=(0, -14.5 * 1.75 * 1.25), textcoords='offset points', color='#111111' if style == 'rgbgrey'
                else 'white', fontsize=10 * 1.75, va='top', ha='left', alpha=0.92, zorder=10,
                path_effects=[pe.withStroke(linewidth=2.2 * 1.75, foreground='white' if style == 'rgbgrey'
                                            else 'black', alpha=0.7)])
    c_reff, _, _ = overlay_colours(style == 'rgbgrey')
    r = Row(fig, 0.3, strip - 0.3, 'white', 11)
    for lab, wl, c in BAND_KEY:
        r.swatch(c, size=0.14)
        r.text(f'{lab} {wl}', gap=0.28)
    if style == 'rgbgrey':
        r.text('·  faint light: inverted greyscale', '#a8a8a8')
    r = Row(fig, 0.3, strip - 0.64, 'white', 10.5)
    r.line(c_reff, LS_REFF, length=0.4)
    r.text(REFF_TXT, gap=0.35)
    r.text('M★ [M☉], SFR [M☉ yr⁻¹]', '#a8a8a8')
    fig.text(0.3 / P, (strip - 0.98) / (P + strip), f'{CREDIT}   ·   {SAMPLE_CREDIT}   ·   N up, E left',
             color='#9a9a9a', fontsize=9.5, ha='left', va='center')
    return fig


# ------------------------------------------------------------------------------------------------ main

ALL = ('png', 'jpg', 'pdf')
VARIANTS = {
    'strips_black_colour': (make_strips, dict(style='rgb'), ALL),
    'strips_black_colourgrey': (make_strips, dict(style='rgbgrey'), ALL),
    'strips_white_colourgrey': (make_strips, dict(style='rgbgrey', theme='light'), ALL),
    'strips_black_colour_clean': (make_strips, dict(style='rgb', overlays=False), ('png', 'jpg')),
    'slide16x9_colour': (make_slide, dict(style='rgb'), ('png', 'pdf')),
    'slide16x9_colourgrey': (make_slide, dict(style='rgbgrey'), ('png', 'pdf')),
}


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
                    mpo.save(make_card(g, style), f'cards/{g["name"]}_{style}_card', ('png', 'jpg'))
            continue
        builder, kw, formats = VARIANTS[name]
        mpo.save(builder(gals, **kw), name, formats)


if __name__ == '__main__':
    main()
