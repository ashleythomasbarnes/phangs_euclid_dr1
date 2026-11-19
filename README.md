# Euclid DR1 RGB images of PHANGS galaxies

R = H_E, G = Y_E, B = I_E (VIS), from the DR1 MER background-subtracted mosaics, mosaicked across tiles and
binned for large galaxies. Environment: `conda run --no-capture-output -n astro ...`

| file | what |
|---|---|
| `euclid_tools.py` | tile footprints + coverage, cutout download, binning/reprojection/mosaicking, RGB stretch |
| `make_phangs_rgb.py` | batch over the PHANGS sample table (`--png-only` to restyle, `--overwrite`, or pass names) |
| `euclid_rgb_cutout.py` | same for any single target / coordinate |
| `data/euclid_dr1_mer_tiles.ecsv` | cached DR1 tile list (VIS/Y/H); `--refresh-tiles` to update |
| `data/phangs_euclid_dr1_coverage.ecsv` | coverage of every PHANGS galaxy (field fraction, centre covered, tiles) |
| `output/fits/<gal>_euclid_<band>.fits` | per-band mosaics on a common grid, uJy/pix |
| `output/png/<gal>_euclid_rgb.png` | RGB with axes; `_noaxes.png` is the bare image |

Notes
- Field half-size = 1.3 x R25 (1.5'-15'); pixel scale is 0.1" unless the field would exceed 4000 px.
- Coverage is computed locally from tile polygons: the archive `fov` column is an array, so ADQL `INTERSECTS` on it
  returns nothing. `Euclid.query_sia` (used by the first version of the script) is not in the installed astroquery.
- Partially covered fields are made anyway; the PNG title gives the covered fraction. Tile footprints include
  zero-padded areas, so galaxies with < 2% real data are reported as not covered (`output/summary.ecsv`).
- `output/phangs_euclid_dr1_gallery.png`: all galaxies in one figure (`--gallery-only` to rebuild).
- Second style, `_rgbgrey` PNGs and `output/phangs_euclid_dr1_gallery_rgbgrey.png`: colour galaxy on an
  inverted-greyscale sky that shows low surface brightness light (`et.colour_on_grey`, settings in `GREY_KW`).
  It also shows the MER background artefacts plainly (tile/exposure steps around NGC 1291, over-subtracted moats).

Presentation figures: `make_poster.py` (run after `make_phangs_rgb.py`) writes `output/figures/`
- `grid_*` 4x3 multipanels (black colour, black colour-on-grey, white colour-on-grey, `_clean` without overlays,
  `grid_wide_*` 6x2), `hero_*` (NGC 1566 large + 11 around it), `slide16x9_*` (3840x2160 with an M*-SFR panel),
  and `cards/<gal>_{rgb,rgbgrey}_card` single-galaxy figures. PNG + JPG, PDF for the main ones.
- Overlays from the sample table: R_eff dashed, R_25 dotted, projected with PHANGS inclination and PA; kpc bars;
  type, distance, log M*, SFR. Panels ordered by T-type. NGC 3031 excluded (`EXCLUDE`); hero galaxy is `HERO`.

Display-only corrections (the FITS are untouched; all applied in `et.make_rgb`)
- NISP bulge holes: DR1 MER masks bright bulges in Y/H (NGC 1291 loses its whole bulge). Filled with VIS scaled by
  the local H/VIS, Y/VIS ratio around each hole (`fill_from_reference`); small holes by interpolation.
- Colour flattening (`flatten_colour`, `FLATTEN_ARCSEC` in `make_phangs_rgb.py`): the per-band MER background
  subtraction leaves different over/under-subtraction under big galaxies, giving colour halos and casts. Residuals
  from a reference colour are smoothed on 30" and subtracted, anchored to each galaxy's own bright-region colour
  (clipped to the range of clean galaxies). Genuine colour gradients on > 30" scales are flattened too.
  Overrides: NGC 1291 (8", fixed reference) and NGC 7793 (fixed reference).
- Faint outskirts fade to grey (`colour_sigma`), for the same reason.
- Not fixable here: masked areas around bright stars (black patches), NGC 7793's over-subtracted VIS rectangle in the
  centre, NGC 3031's partial coverage.

## GECKOS (branch `geckos`)

Same images for the GECKOS edge-on sample (master catalogue `data/geckos_mastercat_v20260209.csv`, J. van de Sande).

  conda run --no-capture-output -n astro python make_geckos_rgb.py      # coverage, mosaics, PNGs
  conda run --no-capture-output -n astro python make_geckos_poster.py   # figures

- Field half-size = 4 R_e (r band; min 1.5'), since the catalogue has no R25. Coverage of all 36 galaxies in
  `data/geckos_euclid_dr1_coverage.ecsv`: 3 fall on DR1 tile footprints, but ESO 484-036 lies in zero padding, so
  2 have data (ESO 079-003, ESO 120-016; `output_geckos/summary.ecsv`).
- Colour flattening is off (`FLATTEN` in `make_geckos_rgb.py`): the 30" kernel is wider than these thin disks and
  left teal fringes along them. All other RGB settings as for PHANGS.
- Figures in `output_geckos/figures/`: `strips_*` rotate each disk horizontal by its PA and crop to a 3:1 strip
  (±4 R_e where the field allows) with a rotated N/E compass, ordered by M*; `slide16x9_*`; north-up `cards/`.
  Overlays: R_e with the catalogue ellipticity, kpc bar, log M*, log SFR.
- Not fixable here: the black masked patch beside ESO 120-016's disk (masked in all bands in MER).

Download robustness (`euclid_tools.py`, applies to PHANGS too): the cutout service sometimes returns a valid but
near-empty FITS, or a full-size cutout with a stretch of zeros (a different band each time). `fetch_cutout` retries
files much smaller than requested, and `build_band_mosaics` re-downloads any band that is more than 3% less complete
than the best band (`REFETCH_TRIES`, `REFETCH_SHORTFALL`).
