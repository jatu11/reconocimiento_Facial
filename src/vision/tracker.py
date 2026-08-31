from __future__ import annotations

import numpy as np
import supervision as sv
from src.utils.config import TRACKER_BUFFER, TRACKER_MATCH_THRESH
from src.vision.frame_context import FrameContext


class FaceTracker:
    """
    Tracker basado en ByteTrack.

    Trabaja directamente sobre FrameContext.
    Únicamente asigna Track IDs a cada DetectedFace.
    """

    def __init__(self):

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=TRACKER_BUFFER,
            minimum_matching_threshold=TRACKER_MATCH_THRESH,
            frame_rate=30,
        )

        self.generated_ids: set[int] = set()

    def update(self, context: FrameContext) -> FrameContext:

        if not context.faces:
            self.tracker.update_with_detections(sv.Detections.empty())

            return context

        xyxy = np.array(
            [
                [
                    face.left,
                    face.top,
                    face.right,
                    face.bottom,
                ]
                for face in context.faces
            ],
            dtype=np.float32,
        )

        confidence = np.array(
            [max(face.score, 0.5) for face in context.faces],
            dtype=np.float32,
        )

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=np.zeros(len(context.faces), dtype=np.int32),
        )

        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked) == 0:
            return context

        for face, track_id in zip(
            context.faces,
            tracked.tracker_id,
        ):
            face.track_id = int(track_id)

            self.generated_ids.add(face.track_id)

        return context

    def get_total_ids(self) -> int:

        return len(self.generated_ids)
