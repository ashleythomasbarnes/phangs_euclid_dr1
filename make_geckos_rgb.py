"""
Euclid DR1 RGB images (R = H_E, G = Y_E, B = I_E) for the GECKOS galaxies covered by the DR1 MER mosaics.
Same machinery and colour settings as make_phangs_rgb.py; only the sample, field size and output folder differ.

  1. Coverage of every galaxy in the GECKOS master catalogue -> data/geckos_euclid_dr1_coverage.ecsv
  2. Per covered galaxy: per-band mosaics -> output_geckos/fits/<gal>_euclid_<band>.fits
  3. RGB PNGs (rgb, rgbgrey, with and without axes) -> output_geckos/png/

  conda run --no-capture-output -n astro python make_geckos_rgb.py
  conda run --no-capture-output -n astro python make_geckos_rgb.py eso079-003 --overwrite
  conda run --no-capture-output -n astro python make_geckos_rgb.py --png-only

Figures: make_geckos_poster.py.
"""

import argparse

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import euclid_tools as et
import make_phangs_rgb as mp

SAMPLE_TABLE = et.DATA_DIR / 'geckos_mastercat_v20260209.csv'      # GECKOS master catalogue (J. van de Sande)
COVERAGE_TABLE = et.DATA_DIR / 'geckos_euclid_dr1_coverage.ecsv'
OUT = et.ROOT / 'output_geckos'
OUT_FITS, OUT_PNG, RAW_DIR = OUT / 'fits', OUT / 'png', OUT / 'tmp_raw'
SUMMARY_TABLE = OUT / 'summary.ecsv'

SIZE_FACTOR = 4.0                  # field half-size = SIZE_FACTOR * R_e (r band); the catalogue has no R25
MIN_HALF, MAX_HALF = 1.5 * u.arcmin, 15 * u.arcmin
# Colour flattening off: on these small fields the 30" kernel is wider than the edge-on disks are thick, so it removes
# real colour structure and leaves teal fringes along the disk. Per-galaxy (scale arcsec, own colour?) overrides below.
FLATTEN = (None, True)
FLATTEN_OVERRIDE = {}
# Hole filling beyond the PHANGS settings (display only, see et.make_rgb / et.fill_from_reference):
#  fill_vis_from_y: MER VIS has masked patches where NISP is fine (13" patch beside ESO 120-016's disk)
#  local_thin_fill: NISP ghost arcs crossing a galaxy (ESO 079-003) are filled with the local H/VIS, Y/VIS ratio,
#                   not one ratio for the whole arc, which put the sky's colour onto the disk
RGB_KW = dict(mp.RGB_KW, fill_vis_from_y=True, local_thin_fill=True)


def gal_name(row):
    return str(row['object']).strip().lower()


def field_half_size(row):
    re = float(row['R_e_r']) * u.arcsec
    if not np.isfinite(re.value):
        re = 30 * u.arcsec
    return np.clip(SIZE_FACTOR * re.to(u.arcmin), MIN_HALF, MAX_HALF)


def coverage(sample, tiles):
    rows = []
    for r in sample:
        c = SkyCoord(r['ra'], r['dec'], unit='deg')
        h = field_half_size(r)
        hits, frac, centre = et.tiles_overlapping(c, h, tiles)
        rows.append((gal_name(r), c.ra.deg, c.dec.deg, h.to_value(u.arcmin), frac, centre,
                     ','.join(str(k) for k, _, _ in hits)))
    cov = Table(rows=rows, names=['name', 'ra', 'dec', 'half_size_arcmin', 'cover_frac', 'centre_covered', 'tiles'])
    cov['half_size_arcmin'].unit = u.arcmin
    cov.write(COVERAGE_TABLE, overwrite=True)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', help='galaxy names, lower case as in the catalogue (default: all covered)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--png-only', action='store_true', help='remake PNGs from existing FITS')
    ap.add_argument('--refresh-tiles', action='store_true', help='re-download the DR1 tile list')
    args = ap.parse_args()

    for d in (OUT_FITS, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)

    tiles = et.complete_tiles(et.load_tiles(refresh=args.refresh_tiles))
    cov = coverage(Table.read(SAMPLE_TABLE), tiles)
    covered = cov[cov['cover_frac'] > 0]
    print(f'{len(covered)}/{len(cov)} GECKOS galaxies overlap Euclid DR1 (VIS+Y+H) -> {COVERAGE_TABLE}')
    for r in covered:
        print(f"  {r['name']:<12} cover={r['cover_frac']:.2f} centre={r['centre_covered']!s:<5} "
              f"half-size={r['half_size_arcmin']:.1f}'  tiles={r['tiles']}")

    done, failed = [], []
    for r in covered:
        name = str(r['name'])
        if args.names and name not in args.names:
            continue
        fits_fmt = OUT_FITS / f'{name}_euclid_{{band}}.fits'
        have = all((OUT_FITS / f'{name}_euclid_{b}.fits').exists() for b in et.BANDS)
        print(f'\n=== {name}')
        try:
            done.append(mp._one_galaxy(r, name, fits_fmt, have, args, tiles, out_png=OUT_PNG, raw_dir=RAW_DIR,
                                       survey='GECKOS  ', flatten_override=FLATTEN_OVERRIDE,
                                       flatten=FLATTEN, rgb_kw=RGB_KW))
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


if __name__ == '__main__':
    main()
