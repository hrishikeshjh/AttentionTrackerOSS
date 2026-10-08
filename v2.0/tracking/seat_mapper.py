"""
SeatMapper — assigns tracked persons to pre-configured seat zones.
(ARCHITECTURE.md §3 Tracking Layer + §3 Seat Occupancy Detection)

Seat zones are polygons in normalised [0,1] frame coordinates.
A person is assigned to a seat when IoU(person_bbox, seat_polygon) > threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon
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
        Return the occupancy status of every configured seat.
        If two persons overlap the same seat, the one with higher IoU wins.
        """
        # Build seat → best_track mapping
        seat_best: dict[str, tuple[float, int]] = {}  # seat_id → (iou, track_id)

        for track in tracks:
            x1, y1, x2, y2 = track.bbox_xyxy
            person_box = shapely_box(x1, y1, x2, y2)
            # Centre of the person's bounding box in pixel space
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            for seat_id, seat_poly in self._seat_polygons.items():
                # Primary: centre-point containment (works for both CCTV and
                # close-up webcam where IoU of a large bbox is low).
                from shapely.geometry import Point
                if seat_poly.contains(Point(cx, cy)):
                    score = 1.0  # prefer centroid-matched seats
                else:
                    # Fallback: IoU for partial overlaps (e.g. person at seat edge)
                    intersection = person_box.intersection(seat_poly).area
                    union = person_box.union(seat_poly).area
                    score = intersection / union if union > 0 else 0.0
                    if score < SEAT_IOU_THRESHOLD:
                        continue

                prev_score, _ = seat_best.get(seat_id, (0.0, -1))
                if score > prev_score:
                    seat_best[seat_id] = (score, track.track_id)

        return [
            SeatStatus(
                seat_id=sid,
                occupied=sid in seat_best,
                track_id=seat_best[sid][1] if sid in seat_best else None,
            )
            for sid in self._seat_polygons
        ]
