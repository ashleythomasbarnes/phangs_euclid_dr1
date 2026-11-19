"""
Shared tools for making Euclid DR1 RGB images of (large) galaxies from the MER background-subtracted mosaics.

Pipeline for one galaxy:
  1. Footprints: the DR1 list of MER tiles (VIS, NIR_Y, NIR_H) is fetched once from the archive TAP and cached in
     data/euclid_dr1_mer_tiles.ecsv. Coverage of a galaxy is computed locally from the tile polygons (the archive
     `fov` column is an array, not an ADQL geometry, so spatial ADQL queries on it return nothing).
  2. Output grid: a square TAN grid centred on the galaxy, side = 2 * SIZE_FACTOR * R25 (clipped), with a pixel scale
     chosen so the image is at most MAX_NPIX across (never finer than the native 0.1").
  3. Cutout planning in each tile's own pixel grid (0.1"/pix TAN about crval/crpix, image 2*crpix wide): the field is
     intersected with the tile core (fov) and the image bounds, then covered by square cutouts (the service only does
     squares, and refuses any that reach past the image edge). Each square is probed with a few tiny cutouts first and
     skipped if it holds no data (tiles are zero-padded outside the observed area).
  4. Cutouts are fetched directly from the ESA cutout service, several in parallel (astroquery's downloader is ~15x
     slower), then block-averaged towards the output pixel scale, converted to uJy/pix (MAGZERO), and reprojected
     onto just the part of the output grid they cover. The closest tile wins where tiles overlap. Raw files are
     deleted once used.
  5. RGB: R = H_E, G = Y_E, B = I_E, asinh stretch with black/white points set from the sky noise and galaxy
     brightness (see make_rgb).
"""

import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
TILE_LIST = DATA_DIR / 'euclid_dr1_mer_tiles.ecsv'

# band key -> archive filter_name. Order sets R, G, B.
BANDS = {'H': 'NIR_H', 'Y': 'NIR_Y', 'VIS': 'VIS'}
NATIVE_SCALE = 0.1 * u.arcsec     # MER mosaics are 0.1"/pix in all bands
CUTOUT_URL = 'https://eas.esac.esa.int/sas-cutout/cutout'
N_WORKERS = 4                     # parallel downloads; the service gives ~5-6 MB/s in total with 4
MAX_PIECE_PIX = 6000              # largest square cutout (10'), keeps memory per piece < 150 MB
EDGE_MARGIN_PIX = 20              # stay this far inside the tile image (the service pads cutouts by ~5 px)
REFETCH_TRIES = 2                 # re-downloads of a band whose mosaic is less complete than the others
REFETCH_SHORTFALL = 0.03          # ... i.e. by more than this fraction of the field (NISP covers ~1% less than VIS)
PROBE_GRID = 3                    # probe each piece at PROBE_GRID^2 points for data before downloading it


# ----------------------------------------------------------------------------------------------- tile footprints

def load_tiles(refresh=False):
    """DR1 MER background-subtracted mosaics in VIS/Y/H, one row per (tile, band). Cached on disk."""
    if TILE_LIST.exists() and not refresh:
        return Table.read(TILE_LIST)
    from astroquery.esa.euclid import Euclid
    filters = ','.join(f"'{f}'" for f in BANDS.values())
    query = ("SELECT tile_index, filter_name, ra, dec, fov, crpix1, crpix2, crval1, crval2, file_name, file_path, "
             "release_name, zero_point "
             "FROM dr1.mosaic_product WHERE product_type='DpdMerBksMosaic' AND category='SCIENCE' "
             f"AND filter_name IN ({filters})")
    tab = Euclid.launch_job_async(query).get_results()
    DATA_DIR.mkdir(exist_ok=True)
    tab.write(TILE_LIST, overwrite=True)
    return tab


def complete_tiles(tiles):
    """{tile_index: {'ra', 'dec', 'poly' (N,2 deg), band: full server file path}} for tiles with all bands."""
    out = {}
    tiles = tiles[np.argsort(np.asarray(tiles['release_name'], str))]   # later releases (DR1_R2 > DR1_R1) win
    for row in tiles:
        band = next((b for b, f in BANDS.items() if f == str(row['filter_name'])), None)
        if band is None:
            continue
        t = out.setdefault(int(row['tile_index']), {'ra': float(row['ra']), 'dec': float(row['dec'])})
        t['poly'] = _parse_fov(row['fov'])
        t['crval'] = (float(row['crval1']), float(row['crval2']))
        t['crpix'] = (float(row['crpix1']), float(row['crpix2']))
        t[band] = f"{str(row['file_path']).rstrip('/')}/{row['file_name']}"
        t[f'{band}_release'] = str(row['release_name'])
    return {k: v for k, v in out.items() if all(b in v for b in BANDS)}


def _parse_fov(fov):
    """fov comes back as an array, or (from the ECSV cache) as a string like '[ra dec ra dec ...]'."""
    if isinstance(fov, str):
        return np.array(fov.replace('[', ' ').replace(']', ' ').split(), float).reshape(-1, 2)
    return np.asarray(fov, dtype=float).reshape(-1, 2)


def _offsets(center, ra, dec):
    """Tangent-plane offsets (deg) of ra/dec from center; x increases to the east."""
    c = SkyCoord(ra, dec, unit='deg')
    off = c.transform_to(center.skyoffset_frame())
    return off.lon.deg, off.lat.deg


def tiles_overlapping(center, half_size, tiles):
    """Tiles whose footprint overlaps the square field, and the covered fraction of that field."""
    h = half_size.to_value(u.deg)
    n = 60
    gx, gy = np.meshgrid(np.linspace(-h, h, n), np.linspace(-h, h, n))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    covered = np.zeros(len(pts), bool)
    centre_in = False
    hits = []
    keys = list(tiles)
    sep = center.separation(SkyCoord([tiles[k]['ra'] for k in keys], [tiles[k]['dec'] for k in keys], unit='deg')).deg
    for k in np.asarray(keys)[sep < h * 1.5 + 0.5]:
        k = int(k)
        t = tiles[k]
        x, y = _offsets(center, t['poly'][:, 0], t['poly'][:, 1])
        inside = MplPath(np.column_stack([x, y])).contains_points(pts)
        box_hit = (x.max() > -h) & (x.min() < h) & (y.max() > -h) & (y.min() < h)
        if inside.any() or box_hit:
            covered |= inside
            centre_in |= MplPath(np.column_stack([x, y])).contains_point((0.0, 0.0))
            hits.append((k, x, y))
    return hits, covered.mean(), bool(centre_in)


# ----------------------------------------------------------------------------------------------- output grid

def output_wcs(center, half_size, max_npix):
    side = 2 * half_size.to(u.arcsec)
    scale = max(NATIVE_SCALE.to_value(u.arcsec), side.value / max_npix)
    npix = int(np.ceil(side.value / scale))
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w, (npix, npix), scale


# ----------------------------------------------------------------------------------------------- cutouts + mosaic

def _to_ujy(data, header):
    zp = header.get('MAGZERO')
    if zp is None:
        raise ValueError('no MAGZERO in header')
    return data * 10 ** (-0.4 * (zp - 23.9))


def _binned_wcs(w, f):
    """WCS of an image block-averaged by f. (WCS.slice with a step ignores a CD matrix, which MER headers use.)"""
    w = w.deepcopy()
    w.wcs.crpix = (w.wcs.crpix - 0.5) / f + 0.5
    if w.wcs.has_cd():
        w.wcs.cd = w.wcs.cd * f
    else:
        w.wcs.cdelt = w.wcs.cdelt * f
    return w


def tile_wcs(t):
    """Pixel grid of a MER tile: TAN about crval, 0.1"/pix, north up; image is 2*crpix pixels on a side."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = t['crval']
    w.wcs.crpix = t['crpix']
    s = NATIVE_SCALE.to_value(u.deg)
    w.wcs.cdelt = [-s, s]
    return w


# --- direct HTTP access to the cutout service (thread-safe, unlike astroquery's TAP connection handler)

_local = threading.local()


def _session():
    import requests
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
    return _local.s


def fetch_cutout(file_path, ra, dec, radius_deg, out_file, tries=4):
    """Download one square cutout (side 2*radius) to out_file. Returns the path, or None if it is off the image."""
    import requests
    params = {'TAPCLIENT': 'ASTROQUERY', 'FILEPATH': file_path, 'POS': f'CIRCLE,{ra},{dec},{radius_deg}'}
    # float32 at native scale; the service now and then returns a valid but near-empty FITS, so retry those
    # (on the last attempt accept anything valid, in case the cutout really is clipped by the image edge)
    expected = 4 * (2 * radius_deg * 3600 / NATIVE_SCALE.to_value(u.arcsec)) ** 2
    for attempt in range(tries):
        try:
            with _session().post(CUTOUT_URL, data=params, timeout=1800, stream=True) as r:
                if r.status_code == 200:
                    with open(out_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1 << 22):
                            f.write(chunk)
                    size = Path(out_file).stat().st_size
                    with open(out_file, 'rb') as f:
                        if f.read(6) == b'SIMPLE' and size >= 2880 and (size >= 0.5 * expected
                                                                         or attempt == tries - 1):
                            return out_file
                elif 'not within image' in r.text:
                    return None
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'cutout failed {tries} times: {Path(file_path).name} at {ra:.5f},{dec:+.5f} r={radius_deg:.4f}')


def _has_data(file_path, ra, dec, out_file):
    """Tiny (~2") cutout: does the mosaic hold any data here?"""
    f = fetch_cutout(file_path, ra, dec, 1 / 3600, out_file)
    if f is None:
        return False
    d = fits.getdata(f)
    Path(f).unlink(missing_ok=True)
    return bool(np.any(np.isfinite(d) & (d != 0)))


def _widen(a, b, target, lo, hi):
    """Grow [a, b] to length target about its centre, shifted/clipped to stay within [lo, hi]."""
    c = (a + b) / 2
    a, b = c - target / 2, c + target / 2
    if a < lo:
        a, b = lo, lo + target
    if b > hi:
        a, b = hi - target, hi
    return max(a, lo), min(b, hi)


def plan_pieces(center, half_size, hits, tiles):
    """
    Square cutouts covering the field, per tile, closest tile first. Each piece is planned in the tile's pixel grid:
    field box  ∩  tile core (fov)  ∩  image bounds (minus a margin), tiled by squares of side <= MAX_PIECE_PIX.
    """
    h = half_size.to_value(u.deg)
    frame = center.skyoffset_frame()
    edge = np.linspace(-h, h, 9)
    ring = np.concatenate([np.column_stack([edge, -h + 0 * edge]), np.column_stack([edge, h + 0 * edge]),
                           np.column_stack([-h + 0 * edge, edge]), np.column_stack([h + 0 * edge, edge])])
    field = SkyCoord(ring[:, 0] * u.deg, ring[:, 1] * u.deg, frame=frame).icrs
    hits = sorted(hits, key=lambda t: np.hypot(np.mean(t[1]), np.mean(t[2])))
    pieces = []
    for k, _, _ in hits:
        t = tiles[k]
        w = tile_wcs(t)
        fx, fy = w.world_to_pixel(field)
        cx, cy = w.world_to_pixel(SkyCoord(t['poly'][:, 0], t['poly'][:, 1], unit='deg'))
        nx_img, ny_img = 2 * t['crpix'][0], 2 * t['crpix'][1]
        x0 = max(fx.min(), cx.min(), EDGE_MARGIN_PIX)
        x1 = min(fx.max(), cx.max(), nx_img - 1 - EDGE_MARGIN_PIX)
        y0 = max(fy.min(), cy.min(), EDGE_MARGIN_PIX)
        y1 = min(fy.max(), cy.max(), ny_img - 1 - EDGE_MARGIN_PIX)
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        # cutouts are square: widen the short side towards the long one where the image allows (it may extend past
        # the field or core, which costs a little extra data but saves a whole second request)
        target = min(max(x1 - x0, y1 - y0), MAX_PIECE_PIX)
        lo_x, hi_x = EDGE_MARGIN_PIX, nx_img - 1 - EDGE_MARGIN_PIX
        lo_y, hi_y = EDGE_MARGIN_PIX, ny_img - 1 - EDGE_MARGIN_PIX
        if x1 - x0 < target:
            x0, x1 = _widen(x0, x1, target, lo_x, hi_x)
        if y1 - y0 < target:
            y0, y1 = _widen(y0, y1, target, lo_y, hi_y)
        side = min(x1 - x0, y1 - y0, MAX_PIECE_PIX)
        nx = int(np.ceil((x1 - x0) / side - 1e-6))
        ny = int(np.ceil((y1 - y0) / side - 1e-6))
        xs = np.linspace(x0 + side / 2, x1 - side / 2, nx) if nx > 1 else [(x0 + x1) / 2]
        ys = np.linspace(y0 + side / 2, y1 - side / 2, ny) if ny > 1 else [(y0 + y1) / 2]
        for px in xs:
            for py in ys:
                c = w.pixel_to_world(px, py)
                g = (np.arange(PROBE_GRID) + 0.5) / PROBE_GRID - 0.5
                probes = w.pixel_to_world(*(a.ravel() for a in np.meshgrid(px + g * side, py + g * side)))
                pieces.append({'tile': k, 'ra': c.ra.deg, 'dec': c.dec.deg,
                               'radius': side / 2 * NATIVE_SCALE.to_value(u.deg), 'side_pix': side,
                               'probes': [(p.ra.deg, p.dec.deg) for p in probes]})
    return pieces


def _split_fetch(ex, tile, band, piece, stem, depth=2):
    """Fetch a refused piece as four quarter squares (recursing up to `depth` levels); returns the paths fetched."""
    w = tile_wcs(tile)
    px, py = w.world_to_pixel(SkyCoord(piece['ra'], piece['dec'], unit='deg'))
    q = piece['side_pix'] / 4
    subs = []
    for j, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        c = w.pixel_to_world(px + dx * q, py + dy * q)
        subs.append({'ra': c.ra.deg, 'dec': c.dec.deg, 'side_pix': 2 * q,
                     'radius': q * NATIVE_SCALE.to_value(u.deg)})
    futs = [ex.submit(fetch_cutout, tile[band], sp['ra'], sp['dec'], sp['radius'], Path(f'{stem}_{j}.fits'))
            for j, sp in enumerate(subs)]
    out = []
    for j, (sp, f) in enumerate(zip(subs, futs)):
        path = f.result()
        if path is not None:
            out.append(path)
        elif depth > 1:
            out += _split_fetch(ex, tile, band, sp, f'{stem}_{j}', depth - 1)
    return out


def _process_piece(path, bin_fac, pix_area_ratio, out_wcs, shape):
    """Raw cutout -> (slices, image) on the part of out_wcs it covers; uJy per output pixel, NaN where no data."""
    from reproject import reproject_interp
    with fits.open(path, memmap=False) as hdul:
        hdu = next(hh for hh in hdul if hh.data is not None)
        data = hdu.data.astype(np.float32)
        hdr = hdu.header
    Path(path).unlink(missing_ok=True)
    data[data == 0] = np.nan              # zero = no data in MER mosaics
    if not np.isfinite(data).any():
        return None
    data = _to_ujy(data, hdr)
    w = WCS(hdr).celestial
    if bin_fac > 1:
        ny, nx = (s - s % bin_fac for s in data.shape)
        data = block_reduce(data[:ny, :nx], bin_fac, func=np.nanmean)
        w = _binned_wcs(w, bin_fac)
    data *= pix_area_ratio                # surface brightness preserved; now uJy per output pixel
    # reproject only onto the output sub-grid this piece covers
    ny, nx = data.shape
    corners = w.pixel_to_world([-0.5, nx - 0.5, nx - 0.5, -0.5], [-0.5, -0.5, ny - 0.5, ny - 0.5])
    ox, oy = out_wcs.world_to_pixel(corners)
    xa, xb = max(0, int(np.floor(ox.min()))), min(shape[1], int(np.ceil(ox.max())) + 1)
    ya, yb = max(0, int(np.floor(oy.min()))), min(shape[0], int(np.ceil(oy.max())) + 1)
    if xb <= xa or yb <= ya:
        return None
    img, foot = reproject_interp((data, w), out_wcs[ya:yb, xa:xb], shape_out=(yb - ya, xb - xa))
    img[foot == 0] = np.nan
    return (slice(ya, yb), slice(xa, xb)), img


def _fill_band(ex, mosaic, band, keep, futs, tiles, raw_dir, bin_fac, pix_area_ratio, out_wcs, shape):
    """Fill the NaN pixels of mosaic from the downloaded pieces (futures by piece index); returns bytes read."""
    nbytes = 0
    for i, p in enumerate(keep):
        paths = [futs[i].result()]
        if paths[0] is None:     # rare: refused although inside the image grid -> split into quarters
            paths = _split_fetch(ex, tiles[p['tile']], band, p, raw_dir / f'{band}_{i}')
            print(f'      {band} piece {i} (tile {p["tile"]}) refused by the service; '
                  f'{len(paths)} sub-piece(s) recovered')
        for path in paths:
            nbytes += Path(path).stat().st_size
            res = _process_piece(path, bin_fac, pix_area_ratio, out_wcs, shape)
            if res is None:
                continue
            sl, img = res
            sub = mosaic[sl]
            fill = np.isnan(sub) & np.isfinite(img)
            sub[fill] = img[fill]
    return nbytes


def build_band_mosaics(center, half_size, hits, tiles, out_wcs, shape, scale, raw_dir, verbose=True):
    """Plan, probe, download (in parallel) and mosaic cutouts onto out_wcs. Returns {band: image in uJy/pix}."""
    t_start = time.time()
    bin_fac = max(1, int(np.floor(scale / NATIVE_SCALE.to_value(u.arcsec))))
    pix_area_ratio = (scale / NATIVE_SCALE.to_value(u.arcsec)) ** 2   # block-mean -> per output pixel
    raw_dir.mkdir(parents=True, exist_ok=True)

    pieces = plan_pieces(center, half_size, hits, tiles)
    with ThreadPoolExecutor(N_WORKERS) as ex:
        # 1. probe each piece in VIS; drop pieces with no data at any probe point
        jobs = [[ex.submit(_has_data, tiles[p['tile']]['VIS'], ra, dec, raw_dir / f'probe_{i}_{j}.fits')
                 for j, (ra, dec) in enumerate(p['probes'])] for i, p in enumerate(pieces)]
        keep = [p for p, js in zip(pieces, jobs) if any(j.result() for j in js)]
        if verbose:
            print(f'    {len(pieces)} piece(s) planned over {len({p["tile"] for p in pieces})} tile(s), '
                  f'{len(keep)} with data  [{time.time() - t_start:.0f}s]')

        # 2. download every (band, piece) in parallel; mosaic in planned order (closest tile first)
        futs = {(band, i): ex.submit(fetch_cutout, tiles[p['tile']][band], p['ra'], p['dec'], p['radius'],
                                     raw_dir / f'{band}_{i}.fits')
                for band in BANDS for i, p in enumerate(keep)}
        out, nbytes = {}, 0
        for band in BANDS:
            out[band] = np.full(shape, np.nan)
            nbytes += _fill_band(ex, out[band], band, keep, {i: futs[(band, i)] for i in range(len(keep))}, tiles,
                                 raw_dir, bin_fac, pix_area_ratio, out_wcs, shape)
            if verbose:
                print(f'    {band:>3}: field filled {np.isfinite(out[band]).mean():.0%}  '
                      f'[{nbytes / 1e6:.0f} MB so far, {time.time() - t_start:.0f}s]')

        # 3. the service occasionally returns a full-size cutout with part of it zeroed (a different band each time):
        #    re-download any band clearly less complete than the best one, filling only its missing pixels
        for attempt in range(REFETCH_TRIES):
            best = max(np.isfinite(m).mean() for m in out.values())
            short = [b for b in BANDS if np.isfinite(out[b]).mean() < best - REFETCH_SHORTFALL]
            if not short:
                break
            for band in short:
                f = {i: ex.submit(fetch_cutout, tiles[p['tile']][band], p['ra'], p['dec'], p['radius'],
                                  raw_dir / f'{band}_{i}_r{attempt}.fits') for i, p in enumerate(keep)}
                nbytes += _fill_band(ex, out[band], band, keep, f, tiles, raw_dir, bin_fac, pix_area_ratio,
                                     out_wcs, shape)
                if verbose:
                    print(f'    {band:>3}: re-fetched, field filled {np.isfinite(out[band]).mean():.0%}  '
                          f'[{time.time() - t_start:.0f}s]')
    shutil.rmtree(raw_dir, ignore_errors=True)
    return out


def write_band_fits(images, out_wcs, scale, path_fmt, meta):
    for band, img in images.items():
        hdr = out_wcs.to_header()
        hdr['BUNIT'] = 'uJy/pix'
        hdr['BAND'] = BANDS[band]
        hdr['PIXSCALE'] = (scale, 'arcsec/pix')
        for k, v in meta.items():
            hdr[k] = v
        fits.PrimaryHDU(img.astype(np.float32), header=hdr).writeto(str(path_fmt).format(band=band), overwrite=True)


# ----------------------------------------------------------------------------------------------- RGB

def fill_small_holes(img, sigma_pix=3.0, max_iter=4):
    """
    Fill NaN holes (masked pixels, saturated cores, thin gaps in one band) by normalised Gaussian interpolation, for
    display only. Repeated a few times so holes up to ~ max_iter * 3 * sigma_pix px across are filled; larger
    unobserved areas stay NaN.
    """
    from scipy.ndimage import gaussian_filter
    img = np.array(img, float)
    for _ in range(max_iter):
        bad = ~np.isfinite(img)
        if not bad.any():
            break
        w = gaussian_filter((~bad).astype(float), sigma_pix)
        v = gaussian_filter(np.where(bad, 0.0, img), sigma_pix)
        with np.errstate(invalid='ignore', divide='ignore'):
            interp = v / w
        fill = bad & (w > 0.05)
        img[fill] = interp[fill]
    return img


def fill_from_reference(img, ref, ring_pix=(3, 15), min_hole_pix=30):
    """
    Fill holes in a NISP band where the reference band (VIS) has data: ref scaled by the median img/ref ratio in a
    ring just outside each hole. For display only. Needed because DR1 MER masks bright bulges in Y/H (NGC 1291 loses
    its whole bulge) while VIS is fine there.
    """
    from scipy.ndimage import binary_dilation, binary_opening, find_objects, label
    img = np.array(img, float)
    hole = ~np.isfinite(img) & np.isfinite(ref)
    # only the thick parts of holes (masked bulges) are filled from the reference; thin masked streaks and arcs
    # (NISP artefacts, which can join onto a bulge hole) are left to fill_small_holes' interpolation
    hole = binary_opening(hole, iterations=8)
    lab, n = label(hole)
    if n == 0:
        return img
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        pad = ring_pix[1] + 1
        sl = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl)
        m = lab[sl] == i
        if m.sum() < min_hole_pix:          # small holes: left to fill_small_holes
            continue
        if m.sum() / max(m.shape) < 8:      # thin masked streaks/arcs (NISP artefacts): interpolate instead
            continue
        outer = binary_dilation(m, iterations=ring_pix[1])
        inner = binary_dilation(m, iterations=ring_pix[0])
        ring = outer & ~inner & np.isfinite(img[sl]) & np.isfinite(ref[sl]) & (ref[sl] > 0)
        if ring.sum() < 10:
            continue
        ratio = np.median(img[sl][ring] / ref[sl][ring])
        sub = img[sl]
        sub[m] = ref[sl][m] * ratio
    return img


# Typical bright-region flux ratios (uJy) of the PHANGS galaxies in DR1 with clean data (IC 1954, NGC 685, 1317,
# 1326, 1433, 1559, 1566, 1792): H/VIS ~ 2-2.9, Y/VIS ~ 1.6-2.0.
REF_COLOUR = {'H': 2.5, 'Y': 1.85, 'VIS': 1.0}
REF_RANGE = {'H': (1.9, 3.3), 'Y': (1.5, 2.4), 'VIS': (1.0, 1.0)}


def _smooth_nan(img, sigma_pix):
    """Normalised Gaussian smoothing that ignores NaNs, done on a block-averaged copy for speed."""
    from scipy.ndimage import gaussian_filter, zoom
    f = max(1, int(sigma_pix // 4))
    ny, nx = (s - s % f for s in img.shape)
    small = block_reduce(img[:ny, :nx], f, func=np.nanmean) if f > 1 else img
    good = np.isfinite(small)
    w = gaussian_filter(good.astype(float), sigma_pix / f)
    v = gaussian_filter(np.where(good, small, 0.0), sigma_pix / f)
    with np.errstate(invalid='ignore', divide='ignore'):
        sm = np.where(w > 1e-3, v / w, 0.0)
    out = np.zeros(img.shape)
    big = zoom(sm, f, order=1) if f > 1 else sm
    out[:big.shape[0], :big.shape[1]] = big
    if big.shape != img.shape:                      # edge rows/cols lost to block_reduce
        out[big.shape[0]:, :] = out[big.shape[0] - 1:big.shape[0], :]
        out[:, big.shape[1]:] = out[:, big.shape[1] - 1:big.shape[1]]
    return out


def flatten_colour(images, pixscale, sigma_arcsec=30.0, ref=REF_COLOUR, own_colour=True, n_iter=3):
    """
    Remove large-scale colour errors left by the per-band MER background subtraction (display only).

    Each band is modelled as ref[b] * L + residual, with L the reference-colour-weighted mean of the bands. The
    residual smoothed on sigma_arcsec scales (sky over/under-subtraction under the galaxy) is subtracted, so the
    large-scale colour is pulled to the reference while colour structure on smaller scales (arms, dust lanes,
    star-forming regions) is kept. Genuine colour gradients on larger scales (e.g. bulge vs disc) are flattened too.
    """
    sig = sigma_arcsec / pixscale
    out = {b: np.array(images[b], float) for b in images}
    # anchor to the galaxy's own colour in its brightest 3% of pixels, clipped to the range seen in clean galaxies,
    # so normal galaxies are barely changed and only outliers (bad background in one band) are pulled in
    # (own_colour=False: use the fixed sample reference, for galaxies whose bright regions are themselves affected)
    if own_colour:
        L0 = np.nanmean(np.stack([out[b] / ref[b] for b in out]), axis=0)
        bright = np.isfinite(L0) & (L0 > np.nanpercentile(L0, 97))
        own = {b: np.nanmedian(out[b][bright] / out['VIS'][bright]) for b in out}
        ref = {b: float(np.clip(own[b], *REF_RANGE[b])) for b in out}
    for _ in range(n_iter):
        L = np.nanmean(np.stack([out[b] / ref[b] for b in out]), axis=0)
        for b in out:
            out[b] = out[b] - _smooth_nan(out[b] - ref[b] * L, sig)
    return out


def clip_middle_band(images, factor=1.6, min_sigma=5.0):
    """
    Y_E lies between I_E and H_E in wavelength, so a real source cannot be much brighter in Y than both neighbours
    suggest. Cap Y at `factor` x the geometric-mean prediction from I_E and H_E (scaled by the reference colours),
    where the prediction is significant. Removes NISP Y-only artefacts (e.g. the green streaks near NGC 1365's
    nucleus) without touching normal colours. Display only.
    """
    H, Y, V = (np.asarray(images[b], float) for b in ('H', 'Y', 'VIS'))
    with np.errstate(invalid='ignore'):
        pred = np.sqrt(np.clip(H / REF_COLOUR['H'], 0, None) * np.clip(V, 0, None)) * REF_COLOUR['Y']
    _, _, sy = sigma_clipped_stats(Y[np.isfinite(Y)], sigma=3, maxiters=5)
    cap = factor * pred + min_sigma * sy
    return np.where(np.isfinite(cap) & (Y > cap), cap, Y)


def make_rgb(images, bands=tuple(BANDS), weights=None, black_sigma=1.0, white_percentile=99.8,
             soften=30.0, saturation=1.25, gamma=1.0, colour_sigma=(3.0, 30.0), flatten_arcsec=None, pixscale=None,
             flatten_own_colour=True, clip_y=False):
    """
    asinh RGB in the style of the ESA Euclid releases.

    Each band is scaled to sky-noise units, the white point is set from a common high percentile so colours stay
    physical, and an asinh stretch with `soften` (white point / soften = linear-to-log knee) brings out faint
    outskirts without burning the centre. Colour is then boosted by `saturation` around the luminance.
    """
    weights = weights or {b: 1.0 for b in bands}
    chans = []
    noise = {}
    ref = np.asarray(images['VIS'], float) if 'VIS' in images else None
    filled = {}
    for b in bands:
        img = images[b]
        if ref is not None and b != 'VIS':
            img = fill_from_reference(img, ref)
        filled[b] = fill_small_holes(img)
    if flatten_arcsec:
        filled = flatten_colour(filled, pixscale, sigma_arcsec=flatten_arcsec, own_colour=flatten_own_colour)
    if clip_y and all(b in filled for b in ('H', 'Y', 'VIS')):
        filled['Y'] = clip_middle_band(filled)
    for b in bands:
        img = filled[b] * weights[b]
        if not np.isfinite(img).any():
            raise ValueError(f'no valid pixels in {b}')
        _, med, std = sigma_clipped_stats(img[np.isfinite(img)], sigma=3, maxiters=5)
        noise[b] = std
        chans.append(img - med)
    stack = np.stack(chans)
    # put bands on comparable footing: normalise by sky noise ratio to the bluest band would change colours, so
    # keep flux ratios (uJy) and set one black/white point for all channels.
    sky = np.median(list(noise.values()))
    black = -black_sigma * sky
    finite = stack[np.isfinite(stack)]
    white = np.percentile(finite, white_percentile)
    x = np.clip((stack - black) / (white - black), 0, None)
    y = np.arcsinh(x * soften) / np.arcsinh(soften)
    lum = y.mean(axis=0, keepdims=True)
    # Background-subtraction residuals differ between bands and show up as colour blotches in faint outskirts, so
    # fade colour to grey below colour_sigma[1] x sky noise (fully grey below colour_sigma[0]).
    if colour_sigma is not None:
        lin = np.nanmean(stack, axis=0) / sky
        from scipy.ndimage import gaussian_filter
        lin = gaussian_filter(np.nan_to_num(lin), 1.0)
        lo, hi = colour_sigma
        f = np.clip((np.log10(np.clip(lin, 1e-3, None)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0, 1)[None]
    else:
        f = 1.0
    y = lum + saturation * f * (y - lum)
    y = np.clip(y, 0, 1) ** (1 / gamma)
    y = np.nan_to_num(y, nan=0.0)
    return np.moveaxis(y, 0, -1)


def colour_on_grey(images, rgb, pixscale, grey_smooth_arcsec=0.6, grey_black_sigma=12.0, grey_q=2.0, sky_level=0.82,
                   blend_smooth_arcsec=3.0, blend_range=(0.12, 0.30)):
    """
    Colour galaxy on an inverted-greyscale sky (Dragonfly / Legacy-Survey-LSB style).

    Grey layer: the band-summed image, lightly smoothed to bring out low surface brightness light, shown inverted:
    sky at `sky_level`, getting darker with flux and fully black at `grey_black_sigma` x (smoothed) sky noise.
    Colour layer: `rgb` (from make_rgb). They are blended on the colour image's brightness smoothed on
    `blend_smooth_arcsec` (so extended galaxy light turns to colour while compact sources stay in the grey layer),
    ramping from grey to colour across `blend_range`. The transition runs through black, giving the dark rim.
    """
    from scipy.ndimage import gaussian_filter
    L = np.nanmean(np.stack([fill_small_holes(images[b]) / REF_COLOUR[b] for b in BANDS]), axis=0)
    L = np.nan_to_num(L, nan=0.0)
    Ls = gaussian_filter(L, max(1.0, grey_smooth_arcsec / pixscale))
    _, med, sig = sigma_clipped_stats(Ls[Ls != 0], sigma=3, maxiters=5)
    x = (Ls - med) / sig
    v = np.arcsinh(np.clip(x, 0, None) / grey_q) / np.arcsinh(grey_black_sigma / grey_q)
    grey = sky_level * (1 - np.clip(v, 0, 1))
    grey += np.clip(-x / 3, 0, 1) * (1 - sky_level) * 0.6      # slightly lighter where below sky
    grey = np.where(L == 0, 1.0, grey)                          # no data: white

    y = gaussian_filter(rgb.mean(axis=-1), max(1.0, blend_smooth_arcsec / pixscale))
    a0, a1 = blend_range
    t = np.clip((y - a0) / (a1 - a0), 0, 1)
    alpha = (t * t * (3 - 2 * t))[..., None]                    # smoothstep
    return alpha * rgb + (1 - alpha) * grey[..., None]


def file_label(name):
    return re.sub(r'[^A-Za-z0-9_+-]', '_', name.strip())
