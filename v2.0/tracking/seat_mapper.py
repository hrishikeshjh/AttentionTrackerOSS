"""
SeatMapper — assigns tracked persons to pre-configured seat zones.
(ARCHITECTURE.md §3 Tracking Layer + §3 Seat Occupancy Detection)

Seat zones are polygons in normalised [0,1] frame coordinates.
A (track, seat) pair is a candidate when the person-box centre lies inside
the seat polygon, or else when IoU(person_bbox, seat_polygon) >=
SEAT_IOU_THRESHOLD.

Assignment is ONE-TO-ONE: candidates are taken greedily in descending order
of (centre-contained, IoU), and each seat and each track is used at most
once. Previously each seat independently picked its best track, so one
large box spanning two seats occupied both and a second person could lose a
seat to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Point, Polygon
from shapely.geometry import box as shapely_box

from server.config import INPUT_HEIGHT, INPUT_WIDTH, SEAT_IOU_THRESHOLD


@dataclass
class SeatStatus:
    seat_id: str
    occupied: bool
    track_id: int | None = None


class SeatMapper:
    """
    Manages seat zone → track_id assignment for one room.

    seat_map: { "seat_1": [[x1,y1],[x2,y2],...], ... }
              coordinates normalised [0,1]
    """

    def __init__(self, seat_map: dict[str, list[list[float]]]):
        # Convert normalised polygons to pixel-space Shapely Polygons
        self._seat_polygons: dict[str, Polygon] = {}
        for seat_id, pts in seat_map.items():
            pixel_pts = [
                (p[0] * INPUT_WIDTH, p[1] * INPUT_HEIGHT) for p in pts
            ]
            self._seat_polygons[seat_id] = Polygon(pixel_pts)

    def assign(
        self,
        tracks: list,   # list of Track from byte_tracker
    ) -> list[SeatStatus]:
        """
        Return the occupancy status of every configured seat (in seat_map
        order). Each track occupies at most one seat and each seat holds at
        most one track.
        """
        candidates: list[tuple[int, float, int, str, int]] = []
        for t_idx, track in enumerate(tracks):
            x1, y1, x2, y2 = track.bbox_xyxy
            person_box = shapely_box(x1, y1, x2, y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centre = Point(cx, cy)

            for s_idx, (seat_id, seat_poly) in enumerate(self._seat_polygons.items()):
                union = person_box.union(seat_poly).area
                iou = person_box.intersection(seat_poly).area / union if union > 0 else 0.0
                # Primary: centre-point containment (works for both CCTV and
                # close-up webcam where IoU of a large bbox is low).
                contained = 1 if seat_poly.contains(centre) else 0
                if not contained and iou < SEAT_IOU_THRESHOLD:
                    continue
                candidates.append((contained, iou, t_idx, seat_id, s_idx))

        # Best first; ties broken deterministically by track / seat order.
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[4]))
        seat_track: dict[str, int] = {}
        used_tracks: set[int] = set()
        for _contained, _iou, t_idx, seat_id, _s_idx in candidates:
            if seat_id in seat_track or t_idx in used_tracks:
                continue
            seat_track[seat_id] = tracks[t_idx].track_id
            used_tracks.add(t_idx)

        return [
            SeatStatus(
                seat_id=sid,
                occupied=sid in seat_track,
                track_id=seat_track.get(sid),
            )
            for sid in self._seat_polygons
        ]
