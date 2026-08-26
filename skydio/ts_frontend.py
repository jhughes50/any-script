"""
    @author Jason Hughes
    @date May 2026

    @about parse the TS file
"""
import traceback
import time
import av 
import klvdata
from klvdata.misb0601 import UASLocalMetadataSet
#from klvdata.misb0903 import VMTILocalSet

import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

from typing import Optional, List, Dict, Tuple, Generator, Any, Iterator

UASLocalMetadataSet.parsers.pop(b'\x4a', None)

"""
    @author Jason Hughes
    @date May 2026

    @about helper objects and functions for x10 processing
"""
import utm
import numpy as np

from datetime import datetime
from dataclasses import dataclass, field

from typing import Optional, List, Dict, Tuple, Generator, Any

FIELD_MAP = {
    "PrecisionTimeStamp":           "timestamp",
    "PlatformHeadingAngle":         "platform_heading",
    "PlatformPitchAngle":           "platform_pitch",
    "PlatformRollAngle":            "platform_roll",
    "SensorLatitude":               "sensor_lat",
    "SensorLongitude":              "sensor_lon",
    "SensorTrueAltitude":           "sensor_alt",
    "FrameCenterLatitude":          "frame_center_lat",
    "FrameCenterLongitude":         "frame_center_lon",
    "FrameCenterElevation":         "frame_center_elev",
    "PlatformGroundSpeed":          "platform_ground_speed",
    "SensorNorthVelocity":          "sensor_north_velocity",
    "SensorEastVelocity":           "sensor_east_velocity",
    "PlatformPitchAngleFull":       "platform_pitch_angle_full",
    "PlatformRollAngleFull":        "platform_roll_angle_full",
    "SensorRelativeAzimuthAngle":   "sensor_relative_azimuth_angle",
    "SensorRelativeElevationAngle": "sensor_relative_elevation_angle",
    "SensorRelativeRollAngle":      "sensor_relative_roll_angle",
    "SlantRange":                   "slant_range",
    "MissionID":                    "mission_id",
    "PlatformDesignation":          "platform_designation",
    "PlatformCallSign":             "platform_callsign",
    "ImageCoordinateSystem":        "image_coordinate_system",
    "CornerLatitudePoint1Full":     "corner_latitude_point_1",
    "CornerLongitudePoint1Full":    "corner_longitude_point_1",
    "CornerLatitudePoint2Full":     "corner_latitude_point_2",
    "CornerLongitudePoint2Full":    "corner_longitude_point_2",
    "CornerLatitudePoint3Full":     "corner_latitude_point_3",
    "CornerLongitudePoint3Full":    "corner_longitude_point_3",
    "CornerLatitudePoint4Full":     "corner_latitude_point_4",
    "CornerLongitudePoint4Full":    "corner_longitude_point_4",
    "SensorHorizontalFieldOfView":  "sensor_horizontal_fov",
    "SensorVerticalFieldOfView":    "sensor_vertical_fov",
    "ImageSourceSensor":            "image_source_sensor",
    "UASLSVersionNumber":           "uas_ls_version_number",
}



@dataclass
class FrameMetadata:
    timestamp: Optional[datetime] = None          
    platform_heading: Optional[float] = None      
    platform_pitch: Optional[float] = None        
    platform_roll: Optional[float] = None         
    sensor_lat: Optional[float] = None            
    sensor_lon: Optional[float] = None            
    sensor_alt: Optional[float] = None            
    frame_center_lat: Optional[float] = None      
    frame_center_lon: Optional[float] = None      
    frame_center_elev: Optional[float] = None     
    packet_time: Optional[float] = None
    platform_ground_speed: Optional[float] = None
    sensor_north_velocity: Optional[float] = None
    sensor_east_velocity: Optional[float] = None
    platform_pitch_angle_full: Optional[float] = None
    platform_roll_angle_full: Optional[float] = None
    sensor_relative_azimuth_angle: Optional[float] = None
    sensor_relative_elevation_angle: Optional[float] = None
    sensor_relative_roll_angle: Optional[float] = None
    slant_range: Optional[float] = None
    mission_id: Optional[str] = None
    platform_designation: Optional[str] = None
    platform_callsign: Optional[str] = None
    image_coordinate_system: Optional[str] = None
    corner_latitude_point_1: Optional[float] = None
    corner_longitude_point_1: Optional[float] = None
    corner_latitude_point_2: Optional[float] = None
    corner_longitude_point_2: Optional[float] = None
    corner_latitude_point_3: Optional[float] = None
    corner_longitude_point_3: Optional[float] = None
    corner_latitude_point_4: Optional[float] = None
    corner_longitude_point_4: Optional[float] = None
    sensor_horizontal_fov: Optional[float] = None
    sensor_vertical_fov: Optional[float] = None
    image_source_sensor: Optional[str] = None
    uas_ls_version_number: Optional[float] = None
    raw: dict = field(default_factory=dict) 

    @classmethod
    def from_klv(cls, klv_packet, packet_time=None) -> "FrameMetadata":
        fm = cls(packet_time=packet_time)
        for item in klv_packet.items.values():
            value = getattr(item.value, "value", item.value)
            attr = FIELD_MAP.get(item.name)
            if attr:
                setattr(fm, attr, value)
            else:
                fm.raw[item.name] = value
        return fm

    @property
    def corners(self) -> np.ndarray | None:
        corners = [self.corner_latitude_point_1,
                   self.corner_longitude_point_1,
                   self.corner_longitude_point_2,
                   self.corner_latitude_point_2,
                   self.corner_latitude_point_3,
                   self.corner_longitude_point_3,
                   self.corner_latitude_point_4,
                   self.corner_longitude_point_4]

        if None in corners: return None

        e1, n1, _, _ = utm.from_latlon(self.corner_latitude_point_1, self.corner_longitude_point_1)
        e2, n2, _, _ = utm.from_latlon(self.corner_latitude_point_2, self.corner_longitude_point_2)
        e3, n3, _, _ = utm.from_latlon(self.corner_latitude_point_3, self.corner_longitude_point_3)
        e4, n4, _, _ = utm.from_latlon(self.corner_latitude_point_4, self.corner_longitude_point_4)

        return np.array([[e1,n1],[e2,n2],[e3,n3],[e4,n4]])

    @property
    def fov(self) -> np.ndarray:
        return np.array([self.sensor_horizontal_fov, self.sensor_vertical_fov])



@dataclass
class FramePair:
    frame : np.ndarray
    frame_time : float
    metadata : Optional[FrameMetadata]
    metadata_age : Optional[float]

    @property
    def has_metadata(self) -> bool:
        return self.metadata is not None


def dump_raw_klv(data: bytes) -> None:
    uas_key = bytes(UASLocalMetadataSet.key)

    print("\n[RAW KLV]")
    print("packet_len:", len(data))
    print("first_64:", data[:64].hex(" "))
    print("uas_key_at:", data.find(uas_key))
    print("starts_with_uas_key:", data.startswith(uas_key))

    try:
        for outer_i, (outer_key, outer_value) in enumerate(KLVParser(data, key_length=16)):
            print(
                "\n[OUTER]",
                "i=", outer_i,
                "key=", outer_key.hex(" "),
                "value_len=", len(outer_value),
                "is_uas=", outer_key == uas_key,
            )

            if outer_key != uas_key:
                continue

            dump_raw_uas_local_set(outer_value)
    except BaseException as exc:
        print("[RAW KLV] outer parse failed")
        print("type:", type(exc).__name__)
        print("repr:", repr(exc))
        print("args:", getattr(exc, "args", None))
        traceback.print_exc()

class TSFrontEnd(object):

    def __init__(self, path : str) -> None:
        self.container_ = av.open(path)

        self.video_ = self.container_.streams.video[0]
        self.klv_ = next((s for s in self.container_.streams if s.type == "data"), None)

        self.max_metadata_age_ = 1.0

    def __del__(self) -> None:
        if self.container_:
            self.container_.close()

    def iterFrames(self) -> Generator[Any, Any, Any]:
        # iter through frames and meta data as generator 
        latest_meta: Optional[FrameMetadata] = None
        latest_meta_time: Optional[float] = None

        streams = [self.video_] + ([self.klv_] if self.klv_ else [])

        for packet in self.container_.demux(*streams):
            if packet.size == 0:
                continue

            if self.klv_ is not None and packet.stream.index == self.klv_.index:
                data= bytes(packet)
                pkt_time = (
                    float(packet.pts * self.klv_.time_base)
                    if packet.pts is not None else latest_meta_time
                )
                try:
                    for klv_packet in klvdata.StreamParser(bytes(packet)):
                        latest_meta = FrameMetadata.from_klv(klv_packet, pkt_time)
                        latest_meta_time = pkt_time
                except ValueError as e:
                    print("\n[KLV] typed parse failed")
                    print("stream:", packet.stream.index)
                    print("pts:", packet.pts)
                    print("pkt_time:", pkt_time)
                    print("packet_size:", packet.size)
                    print("exception_type:", type(e).__name__)
                    print("exception_repr:", repr(e))
                    print("exception_args:", getattr(e, "args", None))
                    traceback.print_exc()
                    continue

            if packet.stream.index == self.video_.index:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    ft = float(frame.pts * self.video_.time_base)

                    meta = latest_meta
                    age = None
                    if meta is not None and latest_meta_time is not None:
                        age = ft - latest_meta_time
                        if self.max_metadata_age_ is not None and age > self.max_metadata_age_:
                            meta = None  
                            if packet.size == 0: age = None

                    yield FramePair(
                        frame = frame.to_ndarray(format="rgb24"),
                        frame_time = ft,
                        metadata = meta,
                        metadata_age = age
                    )
    def iterFramesRealTime(self, speed : float = 1.0) -> Iterator:
        """simulate realtime klv stream"""
        wall_start = None
        pts_start = None

        for pair in self.iterFrames():
            if wall_start is None:
                wall_start = time.monotonic()
                pts_start = pair.frame_time

            target_wall = wall_start + (pair.frame_time - pts_start) / speed
            delay = target_wall - time.monotonic()

            if delay > 0: time.sleep(delay)

            yield pair
