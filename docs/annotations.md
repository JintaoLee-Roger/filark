# Annotation

FiLark annotations are editable labels stored in source coordinates. They are
designed for browsing-time event labeling and for reviewing detector output.

## Supported Geometry

| Type | Use case | Stored data |
|---|---|---|
| `bbox` | compact time-channel events | `t0`, `t1`, `c0`, `c1` |
| `polyline` | arrivals, moveout curves, tracks | `points = [(t, c), ...]` |
| `mask` | irregular or dense detector regions | `t0`, `t1`, `c0`, `c1`, `mask` |

Coordinates are semantic time-channel coordinates. They do not depend on zoom,
window size, display scale, or pixel position.

## Draw And Edit

Open the Annotation panel before drawing, and turn off **Drag to select** in the
ROI panel.

Current canvas interaction:

- `BBox`: hold `Ctrl` and drag.
- `Polyline`: `Ctrl+Left-click` to start; click more points; use `Enter`,
  `Right-click`, or `Esc` to finish/cancel.
- `Polyline edit`: select a polyline, drag a vertex, `Shift+click` a segment to
  insert a point, `Alt/Option+click` a vertex to delete it.
- `Mask Brush`: choose `Mask Brush`, then `Ctrl+drag` to paint. Use brush action
  `Erase` or hold `Shift` while painting to erase.
- `Mask undo/redo`: `Ctrl+Z` and `Ctrl+Y`.

The mask brush radius is measured in screen pixels, so it remains visually
consistent after fit-all or zoom changes. The saved mask is still projected into
time-channel coordinates.

## Labels And Metadata

Each annotation has:

- label;
- visibility;
- label style color;
- optional `confidence`;
- optional `reviewed`;
- optional note and custom attributes.

The side panel list is the persistent annotation store for the current session.
Canvas overlays are a rendering of that store.

## Detection Overlays Are Not Automatically Annotations

Monitor plugins can render bbox, polyline, and mask overlays. Those overlays are
owned by the monitor until they are saved/imported as annotation JSON. This
keeps temporary detector output separate from reviewed labels.

## Export Format

Annotation export is JSON. The payload contains:

- `version`
- `source`: source metadata such as path, files, shape, dims, fs, dx, unit, and
  optional gauge length/start time;
- `label_styles`
- `items`

Mask arrays are encoded as compressed base64 payloads by
`filark.utils.annotations.encode_annotation_data`.

Compact example:

```json
{
  "version": 1,
  "source": {
    "kind": "H5Tape",
    "path": "/abs/path/to/example.h5",
    "dims": "nt_nc",
    "shape": [12000, 2048],
    "fs": 1550.0,
    "dx": 5.0,
    "dx_unit": "m"
  },
  "items": [
    {
      "uid": 1,
      "label": "event",
      "anno_type": "bbox",
      "data": {"t0": 1000, "t1": 1200, "c0": 40, "c1": 90}
    }
  ]
}
```

## Use Annotations From Python

```python
from filark.utils.annotations import annotation_items, annotation_bounds

items = annotation_items("annotations.json")
item = items[0]
t0, t1, c0, c1 = annotation_bounds(item, pad_t=100, pad_c=8)
```

To reopen the referenced source:

```python
from filark.utils.annotations import open_annotation_source

source = open_annotation_source("annotations.json")
```

## Mask Resolution

Mask texture resolution does not have to equal the projected data extent. A
small mask can be displayed over a large `t0:t1, c0:c1` box. This is useful for
detector output and for keeping exported annotation files compact. If a workflow
needs pixel-level labels, disable or avoid downsampled mask exports in that
workflow.
