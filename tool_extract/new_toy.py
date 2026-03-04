from __future__ import annotations

# Entry point that re-exports the refactored modules for backward compatibility.

from scripts import config
from scripts.config import *  # noqa: F401,F403
from scripts.geometry import (
    RegionInfo,
    build_regions_from_svg,
    compute_region_colors,
    render_color_regions,
    write_log,
    write_png,
    write_zones_png,
)
from scripts.packing import (
    _log_step,
    _rotate_pts,
    build_bleed,
    compute_scene,
    main,
    pack_regions,
    write_pack_log,
    write_pack_outline_png,
    write_pack_png,
    write_pack_svg,
)
from scripts.svg_utils import (
    _find_embedded_image,
    _get_canvas_size,
    _invert_transform_point,
    _iter_geometry,
    _normalize_name,
    _ops_to_matrix,
    _parse_points,
    _parse_transform,
    _read_image_any,
    _write_svg_paths,
    _write_svg_paths_fill,
    _write_svg_paths_fill_stroke,
    write_region_svg,
    write_zone_outline_svg,
    write_zone_svg,
)
from scripts.zones import (
    _label_pos_for_zone,
    _label_pos_outside,
    _remap_zones_by_area,
    build_zone_boundaries,
    build_zone_geoms,
    build_zone_polys,
    build_zones,
    load_zones_cache,
    save_zones_cache,
    write_zone_outline_png,
    write_zones_log,
)


if __name__ == "__main__":
    main()
