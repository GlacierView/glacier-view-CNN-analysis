"""The 19 bands of each exported Landsat scene, and how to read their values.

Each GeoTIFF is one Earth Engine Collection 2 Level-2 image with every band
exported, in Earth Engine's order (`EE_BANDS`). The six bands the model uses
keep short names (blue, green, red, nir, swir, thermal); the rest are named
after their Earth Engine ids.

Landsat 5 and 7 share a layout. Landsat 8 adds a coastal-aerosol band at index
0, which shifts the reflectance bands by one, so `swir` is index 4 on L5/L7 but
5 on L8. `preprocess.band_indexes` looks positions up in these maps, and
`preprocess.get_common_bands` checks a raster's band count against them. So each
map must list all 19 bands in file order.
"""

#: Earth Engine band ids, in file order.
EE_BANDS = {
    "l5": [
        "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "SR_ATMOS_OPACITY", "SR_CLOUD_QA",
        "ST_B6", "ST_ATRAN", "ST_CDIST", "ST_DRAD", "ST_EMIS", "ST_EMSD", "ST_QA", "ST_TRAD",
        "ST_URAD", "QA_PIXEL", "QA_RADSAT",
    ],
    "l8": [
        "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "SR_QA_AEROSOL", "ST_B10",
        "ST_ATRAN", "ST_CDIST", "ST_DRAD", "ST_EMIS", "ST_EMSD", "ST_QA", "ST_TRAD", "ST_URAD",
        "QA_PIXEL", "QA_RADSAT",
    ],
}
EE_BANDS["l7"] = EE_BANDS["l5"]

_SHORT_NAMES = {
    "l5": {"SR_B1": "blue", "SR_B2": "green", "SR_B3": "red", "SR_B4": "nir", "SR_B5": "swir",
           "SR_B7": "swir_2", "ST_B6": "thermal"},
    "l8": {"SR_B1": "coastal_aerosol", "SR_B2": "blue", "SR_B3": "green", "SR_B4": "red",
           "SR_B5": "nir", "SR_B6": "swir", "SR_B7": "swir_2", "ST_B10": "thermal"},
}
_SHORT_NAMES["l7"] = _SHORT_NAMES["l5"]

#: Satellite -> {band name: index}. Short names for the spectral bands, the
#: lower-cased Earth Engine id for everything else.
landsat_bands = {
    sat: {_SHORT_NAMES[sat].get(ee_id, ee_id.lower()): i for i, ee_id in enumerate(ids)}
    for sat, ids in EE_BANDS.items()
}

# --------------------------------------------------------------------------- #
# Value encoding (Earth Engine's Collection 2 Level-2 band table)
# --------------------------------------------------------------------------- #

#: Reflectance = DN * scale + offset, for every SR_B* band.
SR_SCALE, SR_OFFSET = 2.75e-05, -0.2
#: Surface temperature (K) = DN * scale + offset, for ST_B6 / ST_B10.
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
#: Valid DNs for the reflectance bands. 0 is fill; Landsat 5/7 write 65535 where
#: a visible band saturates, which is common over snow.
SR_VALID_DN = (1, 65455)


def reflectance_to_dn(reflectance: float) -> float:
    return (reflectance - SR_OFFSET) / SR_SCALE


def kelvin_to_dn(kelvin: float) -> float:
    return (kelvin - ST_OFFSET) / ST_SCALE


# --------------------------------------------------------------------------- #
# Suggested cut-offs (notebooks/exploration/band_distributions.ipynb)
# --------------------------------------------------------------------------- #

#: Reflectance 0 to 1. Below 0 is atmospheric over-correction, mostly dark SWIR
#: over ice. Above 1 is valid over sunlit snow but clipping there gave better
#: scaled values, and it also clips the saturation pile-up.
SR_CUTOFF_DN = (round(reflectance_to_dn(0.0)), round(reflectance_to_dn(1.0)))
#: 200-320 K, a generous range for glacier surfaces.
ST_CUTOFF_DN = (round(kelvin_to_dn(200.0)), round(kelvin_to_dn(320.0)))
#: (low, high) in DN for each band the model uses. The current pipeline scales to
#: these (`preprocess.scale_to_cutoffs`) in training and at inference alike.
CUTOFFS_DN = {
    **dict.fromkeys(("blue", "green", "red", "nir", "swir"), SR_CUTOFF_DN),
    "thermal": ST_CUTOFF_DN,
}
