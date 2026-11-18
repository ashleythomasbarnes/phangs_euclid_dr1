"""
Euclid DR1 RGB image (R = H_E, G = Y_E, B = I_E) of any single target, using the same machinery as make_phangs_rgb.py.

  conda run --no-capture-output -n astro python euclid_rgb_cutout.py "NGC 1365" --half-size 8
  conda run --no-capture-output -n astro python euclid_rgb_cutout.py --coord 53.4015 -36.1404 --half-size 3 --name test

Outputs go to output/fits and output/png, as for the PHANGS batch.
"""

import argparse

import astropy.units as u
from astropy.coordinates import SkyCoord

import euclid_tools as et
import make_phangs_rgb as mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', nargs='?', help='Simbad/NED-resolvable name')
    ap.add_argument('--coord', nargs=2, type=float, metavar=('RA', 'DEC'), help='deg; overrides target')
    ap.add_argument('--half-size', type=float, default=1.5, help='field half-size in arcmin')
    ap.add_argument('--name', help='output label (default: from target)')
    ap.add_argument('--max-npix', type=int, default=mp.MAX_NPIX)
    args = ap.parse_args()

    c = SkyCoord(*args.coord, unit='deg') if args.coord else SkyCoord.from_name(args.target)
    name = et.file_label(args.name or args.target or f'{c.ra.deg:.4f}_{c.dec.deg:+.4f}').lower()
    h = args.half_size * u.arcmin

    tiles = et.complete_tiles(et.load_tiles())
    hits, frac, centre = et.tiles_overlapping(c, h, tiles)
    print(f'{name}: {c.to_string("hmsdms")}, {len(hits)} tile(s), {frac:.0%} of field covered')
    if not hits:
        raise SystemExit('Not covered by Euclid DR1 (VIS+Y+H).')

    for d in (mp.OUT_FITS, mp.OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)
    w, shape, scale = et.output_wcs(c, h, args.max_npix)
    images = et.build_band_mosaics(c, h, hits, tiles, w, shape, scale, mp.RAW_DIR / name)
    et.write_band_fits(images, w, scale, mp.OUT_FITS / f'{name}_euclid_{{band}}.fits', {'OBJECT': name})
    mp.save_png(name, images, w, scale, frac)
    print(f'-> {mp.OUT_PNG / (name + "_euclid_rgb.png")}')


if __name__ == '__main__':
    main()
