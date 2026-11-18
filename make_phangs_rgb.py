"""
Euclid DR1 RGB images (R = H_E, G = Y_E, B = I_E) for the PHANGS galaxies covered by the DR1 MER mosaics.

  1. Coverage of every galaxy in the PHANGS sample table, written to data/phangs_euclid_dr1_coverage.ecsv.
  2. For each covered galaxy: multi-tile mosaic per band on a common grid -> output/fits/<gal>_euclid_<band>.fits
  3. RGB PNG -> output/png/<gal>_euclid_rgb.png
  4. Gallery of all of them -> output/phangs_euclid_dr1_gallery.png

Run (all covered galaxies, skipping ones already done):
  conda run --no-capture-output -n astro python make_phangs_rgb.py
Only some galaxies / redo:
  conda run --no-capture-output -n astro python make_phangs_rgb.py ngc1365 ngc1566 --overwrite
Just redo PNGs from existing FITS (e.g. after tweaking the stretch), or only the gallery:
  conda run --no-capture-output -n astro python make_phangs_rgb.py --png-only
  conda run --no-capture-output -n astro python make_phangs_rgb.py --gallery-only
"""

import argparse

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import euclid_tools as et

SAMPLE_TABLE = '/Users/abarnes/Library/CloudStorage/Dropbox/Data/Extragalactic/catalogues/sample_table/phangs_sample_table_v1p6.fits'
COVERAGE_TABLE = et.DATA_DIR / 'phangs_euclid_dr1_coverage.ecsv'
OUT_FITS = et.ROOT / 'output' / 'fits'
OUT_PNG = et.ROOT / 'output' / 'png'
RAW_DIR = et.ROOT / 'output' / 'tmp_raw'

SIZE_FACTOR = 1.3                  # field half-size = SIZE_FACTOR * R25
MIN_HALF, MAX_HALF = 1.5 * u.arcmin, 15 * u.arcmin
MAX_NPIX = 4000                    # output image side; sets the pixel scale for big galaxies
MIN_COVER = 0.0                    # skip galaxies whose field is covered less than this fraction
MIN_DATA_FRAC = 0.02               # tiles are zero-padded: below this fraction of real pixels, call it not covered
SUMMARY_TABLE = et.ROOT / 'output' / 'summary.ecsv'

# Colour stretch, tuned by eye against the ESA release images. Tweak, then rerun with --png-only (no re-download).
# colour_sigma fades colour to grey in faint outskirts, where per-band MER background residuals make blotches.
RGB_KW = dict(black_sigma=-0.5, white_percentile=99.8, soften=30.0, saturation=2.0, colour_sigma=(2, 15),
              weights={'H': 0.85, 'Y': 1.0, 'VIS': 1.3})
# Large-scale colour flattening (display only, see et.flatten_colour): removes colour halos/casts from the per-band
# MER background subtraction around big galaxies (NGC 7793, 1365). Set to None to switch off.
FLATTEN_ARCSEC = 30.0
# per galaxy: (scale in arcsec, anchor to own bright-region colour?). Own colour is wrong where a band's background
# is off even in the brightest parts: NGC 1291 (H crushed around the masked bulge, on small scales) and NGC 7793 (VIS).
FLATTEN_OVERRIDE = {'ngc1291': (8.0, False), 'ngc7793': (30.0, False)}
# Second style: colour galaxy on an inverted-greyscale sky that shows low surface brightness light (et.colour_on_grey)
GREY_KW = dict(grey_black_sigma=5.0, grey_smooth_arcsec=0.35, blend_range=(0.07, 0.20))


def field_half_size(row):
    r25 = float(row['size_r25']) * u.arcsec
    if not np.isfinite(r25.value):
        r25 = 2 * u.arcmin
    return np.clip(SIZE_FACTOR * r25.to(u.arcmin), MIN_HALF, MAX_HALF)


def coverage(sample, tiles):
    rows = []
    for r in sample:
        c = SkyCoord(r['orient_ra'], r['orient_dec'], unit='deg')
        h = field_half_size(r)
        hits, frac, centre = et.tiles_overlapping(c, h, tiles)
        rows.append((str(r['name']), c.ra.deg, c.dec.deg, h.to_value(u.arcmin), frac, centre,
                     ','.join(str(k) for k, _, _ in hits)))
    cov = Table(rows=rows, names=['name', 'ra', 'dec', 'half_size_arcmin', 'cover_frac', 'centre_covered', 'tiles'])
    cov['half_size_arcmin'].unit = u.arcmin
    cov.write(COVERAGE_TABLE, overwrite=True)
    return cov


def save_png(name, images, w, scale, cover_frac):
    flat, own = FLATTEN_OVERRIDE.get(name, (FLATTEN_ARCSEC, True))
    rgb = et.make_rgb(images, **RGB_KW, flatten_arcsec=flat, flatten_own_colour=own, pixscale=scale)
    side = rgb.shape[0] * scale / 60
    for style, img in (('rgb', rgb), ('rgbgrey', et.colour_on_grey(images, rgb, scale, **GREY_KW))):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(projection=w)
        ax.imshow(img, origin='lower')
        ax.coords[0].set_axislabel('RA')
        ax.coords[1].set_axislabel('Dec')
        ax.set_title(f"{name.upper()}  Euclid DR1   R: H$_E$  G: Y$_E$  B: I$_E$   "
                     f"({side:.1f}'  @ {scale:.2f}\"/pix{'' if cover_frac > 0.99 else f', {cover_frac:.0%} covered'})",
                     fontsize=10)
        fig.savefig(OUT_PNG / f'{name}_euclid_{style}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        # plain full-resolution image without axes, for slides etc.
        plt.imsave(OUT_PNG / f'{name}_euclid_{style}_noaxes.png', np.clip(img[::-1], 0, 1))


GALLERY = et.ROOT / 'output' / 'phangs_euclid_dr1_gallery{suffix}.png'


def make_gallery(names, ncol=4, panel_pix=700, style='rgb'):
    """Multi-panel figure of the bare images (each panel = that galaxy's full field, 2 x 1.3 R25)."""
    from PIL import Image
    names = [n for n in names if (OUT_PNG / f'{n}_euclid_{style}_noaxes.png').exists()]
    dark = style == 'rgb'
    fg, bg = ('white', 'black') if dark else ('black', 'white')
    if not names:
        return
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), facecolor=bg)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.96, wspace=0.02, hspace=0.02)
    for ax in np.ravel(axes):
        ax.set_axis_off()
    for ax, n in zip(np.ravel(axes), names):
        im = Image.open(OUT_PNG / f'{n}_euclid_{style}_noaxes.png')
        im.thumbnail((panel_pix, panel_pix), Image.LANCZOS)
        ax.imshow(np.asarray(im))
        hdr = fits.getheader(OUT_FITS / f'{n}_euclid_VIS.fits')
        side = hdr['NAXIS1'] * hdr['PIXSCALE'] / 60
        ax.text(0.03, 0.97, n.upper(), color=fg, fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
        ax.text(0.97, 0.03, f"{side:.0f}'" if side >= 10 else f"{side:.1f}'", color='0.8' if dark else '0.2', fontsize=8,
                ha='right', va='bottom', transform=ax.transAxes)
    fig.suptitle('PHANGS galaxies in Euclid DR1   (R: H$_E$, G: Y$_E$, B: I$_E$)', color=fg, fontsize=13)
    out = str(GALLERY).format(suffix='' if dark else f'_{style}')
    fig.savefig(out, dpi=150, facecolor=bg)
    plt.close(fig)
    print(f'-> {out}')


def make_galleries(names):
    for style in ('rgb', 'rgbgrey'):
        make_gallery(names, style=style)


def _one_galaxy(r, name, fits_fmt, have, args, tiles):
    if args.png_only or (have and not args.overwrite):
        if not have:
            raise FileNotFoundError('no FITS yet')
        images = {b: fits.getdata(str(fits_fmt).format(band=b)) for b in et.BANDS}
        hdr = fits.getheader(str(fits_fmt).format(band='VIS'))
        w, scale = WCS(hdr), hdr['PIXSCALE']
    else:
        c = SkyCoord(r['ra'], r['dec'], unit='deg')
        h = r['half_size_arcmin'] * u.arcmin
        w, shape, scale = et.output_wcs(c, h, MAX_NPIX)
        hits, _, _ = et.tiles_overlapping(c, h, tiles)
        print(f'  grid {shape[0]}x{shape[1]} @ {scale:.2f}"/pix, {len(hits)} tile(s)')
        images = et.build_band_mosaics(c, h, hits, tiles, w, shape, scale, RAW_DIR / name)
        et.write_band_fits(images, w, scale, fits_fmt,
                           {'OBJECT': name, 'TILES': r['tiles'][:60], 'COVFRAC': round(float(r['cover_frac']), 3)})
    data_frac = min(float(np.isfinite(images[b]).mean()) for b in et.BANDS)
    print(f'  field fraction with data in all bands: {data_frac:.0%}')
    if data_frac < MIN_DATA_FRAC:
        raise ValueError(f'only {data_frac:.1%} of the field has data (tile footprint is zero-padded here)')
    save_png(name, images, w, scale, data_frac)
    print(f'  -> {OUT_PNG / (name + "_euclid_rgb.png")}')
    return name, data_frac, scale, images['VIS'].shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', help='galaxy names as in the sample table (default: all covered)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--png-only', action='store_true', help='remake PNGs from existing FITS')
    ap.add_argument('--gallery-only', action='store_true', help='just rebuild the gallery from existing PNGs')
    ap.add_argument('--refresh-tiles', action='store_true', help='re-download the DR1 tile list')
    args = ap.parse_args()

    for d in (OUT_FITS, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)

    tiles = et.complete_tiles(et.load_tiles(refresh=args.refresh_tiles))
    sample = Table.read(SAMPLE_TABLE)
    cov = coverage(sample, tiles)
    covered = cov[cov['cover_frac'] > MIN_COVER]
    print(f'{len(covered)}/{len(cov)} PHANGS galaxies overlap Euclid DR1 (VIS+Y+H) -> {COVERAGE_TABLE}')
    for r in covered:
        print(f"  {r['name']:<12} cover={r['cover_frac']:.2f} centre={r['centre_covered']!s:<5} "
              f"half-size={r['half_size_arcmin']:.1f}'  tiles={r['tiles']}")

    if args.gallery_only:
        make_galleries([str(n) for n in covered['name']])
        return
    todo = [r for r in covered if not args.names or r['name'] in args.names]
    done, failed = [], []
    for r in todo:
        name = str(r['name'])
        fits_fmt = OUT_FITS / f'{name}_euclid_{{band}}.fits'
        have = all((OUT_FITS / f'{name}_euclid_{b}.fits').exists() for b in et.BANDS)
        print(f'\n=== {name}')
        try:
            done.append(_one_galaxy(r, name, fits_fmt, have, args, tiles))
        except Exception as err:      # keep going with the rest of the sample
            print(f'  FAILED: {err!r}')
            failed.append((name, repr(err)[:120]))
    print(f'\nDone: {len(done)} galaxies; failed/not covered: {[f[0] for f in failed] or "none"}')
    if not args.names:
        summ = Table(rows=[(n, 'ok', f, s, p, '') for n, f, s, p in done] +
                          [(n, 'failed', np.nan, np.nan, 0, e) for n, e in failed],
                     names=['name', 'status', 'data_frac', 'pix_arcsec', 'npix', 'note'])
        summ.write(SUMMARY_TABLE, overwrite=True)
        print(f'-> {SUMMARY_TABLE}')
    make_galleries([str(n) for n in covered['name']])


if __name__ == '__main__':
    main()
