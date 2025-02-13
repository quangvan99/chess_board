import cv2
import threading
from seg.process import YOLOModel
from seg.yolo.xla import find_horizontal_vertical_lines_and_intersections
from seg.yolo.det_onnx import YOLOv8ONNXModel
from cfg.src import source
from config.config import cfg
import math
import time
from typing import Dict, Tuple, List
import copy
lock = threading.Lock()

class VideoProcessor:
    def __init__(self, source_id: int, video_path: str):
        self.source_id = source_id
        self.video_path = video_path
        self.corners = None
        self.grid_points = None
        self.is_processed = False
        self.lock = threading.Lock()

    def is_valid_detection(self, corners: List, grid_points: List) -> bool:
        if corners is None or len(corners) != 4:
            return False
        
        if grid_points is None or len(grid_points) < 81:
            return False
        
        return True

    def update_detection(self, corners: List, grid_points: List) -> bool:
        with self.lock:
            if not self.is_processed and self.is_valid_detection(corners, grid_points):
                self.corners = copy.deepcopy(corners)
                self.grid_points = copy.deepcopy(grid_points)
                self.is_processed = True
                return True
            return False

class ResultManager:
    def __init__(self, result_file_path: str):
        self.result_file_path = result_file_path
        self.lock = threading.Lock()
        self.processors: Dict[int, VideoProcessor] = {}
        self.results_cache: Dict[int, Dict] = {}

    def initialize_processor(self, source_id: int, video_path: str):
        self.processors[source_id] = VideoProcessor(source_id, video_path)
        self.results_cache[source_id] = {
            "video_path": video_path,
            "corners": None,
            "grid_points": None
        }

    def update_single_result(self, source_id: int, corners=None, grid_points=None):
        with self.lock:
            self.results_cache[source_id].update({
                "corners": copy.deepcopy(corners) if corners is not None else None,
                "grid_points": copy.deepcopy(grid_points) if grid_points is not None else None
            })
            self._write_all_results()

    def _write_all_results(self):
        with open(self.result_file_path, "w") as file:
            for source_id in sorted(self.results_cache.keys()):
                result = self.results_cache[source_id]
                file.write(f"[source_id_{source_id}]\n")
                file.write(f"video_path = {result['video_path']}\n")
                
                if result["corners"] is not None and result["grid_points"] is not None:
                    formatted_corners = ', '.join([f"{int(coord)}" for corner in result["corners"] for coord in corner])
                    file.write(f"corners = {formatted_corners}\n")
                    formatted_grid_points = ', '.join([f"({x}, {y})" for x, y in result["grid_points"]])
                    file.write(f"grid_points = {formatted_grid_points}\n")
                else:
                    file.write(f"corners = None\n")
                    file.write(f"grid_points = None\n")
                
                file.write("\n")
                
def sort_grid_points(grid_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(grid_points) != 90:
        return grid_points  # Không đủ 90 điểm, không xử lý
    
    # Sắp xếp theo tọa độ y trước
    grid_points.sort(key=lambda p: p[1])
    
    # Chia thành 10 hàng
    sorted_grid = []
    for i in range(10):
        row = grid_points[i * 9:(i + 1) * 9]
        row.sort(key=lambda p: p[0])  # Sắp xếp theo tọa độ x trong từng hàng
        sorted_grid.extend(row)
    
    return sorted_grid

def process_video(video_path: str, source_id: int, stop_event: threading.Event, result_manager: ResultManager):
    processor = result_manager.processors[source_id]
    
    yolo_model = YOLOModel()
    det_inters = YOLOv8ONNXModel(
        path=cfg.model['intersection_detection']['model_path'],
        class_names=0,
        conf_threshold=cfg.model['intersection_detection']['conf_threshold'],
    )
    det_pieces = YOLOv8ONNXModel(
        path=cfg.model['piece_detection']['model_path'],
        class_names=0,
        conf_threshold=cfg.model['piece_detection']['conf_threshold'],
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print(f"Started processing video {video_path} (source_id {source_id})")

    while cap.isOpened() and not stop_event.is_set() and not processor.is_processed:
        ret, frame = cap.read()
        if not ret:
            break

        corners, warped_image = yolo_model.detect_chessboard(frame)
        if len(corners) > 0:
            bbox, sco, cls = det_pieces(warped_image)
            pos_0 = None
            pos_7 = None
            for i, class_id in enumerate(cls):
                if class_id == 0:
                    pos_0 = bbox[i]
                elif class_id == 7:
                    pos_7 = bbox[i]

            if pos_0 is not None and pos_7 is not None:
                x0, y0 = (pos_0[0] + pos_0[2]) / 2, (pos_0[1] + pos_0[3]) / 2
                x7, y7 = (pos_7[0] + pos_7[2]) / 2, (pos_7[1] + pos_7[3]) / 2

                dx, dy = x7 - x0, y7 - y0
                angle = math.degrees(math.atan2(dx, dy))
                rotate = 0

                if abs(angle) < 30 or abs(angle) > 150:
                    if y0 < y7:
                        warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)
                        rotate = 180
                else:
                    if x7 > x0:
                        warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rotate = -90
                    else:
                        warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)
                        rotate = 90

                boxes_inter, score_inter, cls_inter = det_inters(warped_image)
                if len(boxes_inter) == 90:
                    points = [( (x1 + x2) / 2, (y1 + y2) / 2 ) for x1, y1, x2, y2 in boxes_inter]
                    grid_points = [(int(p[0]), int(p[1])) for p in points]
                    grid_points = sort_grid_points(grid_points)
                else:
                    warped_image = cv2.resize(warped_image, (500, 500))
                    grid_points, rotate_90 = find_horizontal_vertical_lines_and_intersections(warped_image)

                if grid_points is not None and len(grid_points) > 0:
                    scale_x = 1280 / warped_image.shape[1]
                    scale_y = 1280 / warped_image.shape[0]
                    grid_points = [(int(x * scale_x), int(y * scale_y)) for x, y in grid_points]
                    
                    # Điều chỉnh corners dựa trên góc xoay
                    corners_copy = copy.deepcopy(corners)
                    if rotate == 90:
                        corners_copy = [corners_copy[3], corners_copy[0], corners_copy[1], corners_copy[2]]
                    elif rotate == -90:
                        corners_copy = [corners_copy[1], corners_copy[2], corners_copy[3], corners_copy[0]]
                    elif rotate == 180:
                        corners_copy = [corners_copy[2], corners_copy[3], corners_copy[0], corners_copy[1]]
                    
                    # Cập nhật kết quả nếu hợp lệ
                    if processor.update_detection(corners_copy, grid_points):
                        result_manager.update_single_result(source_id, corners_copy, grid_points)
                        print(f"Successfully detected valid corners and grid points for video {video_path} (source_id {source_id})")
                        break

    cap.release()
    print(f"Finished processing video {video_path} (source_id {source_id})")

def process_multiple_videos(video_paths: List[str], result_file_path: str):
    result_manager = ResultManager(result_file_path)
    threads = []
    stop_events = []

    # Khởi tạo result manager cho tất cả các video
    for source_id, video_path in enumerate(video_paths):
        result_manager.initialize_processor(source_id, video_path)

    # Tạo và chạy các thread xử lý video
    for source_id, video_path in enumerate(video_paths):
        stop_event = threading.Event()
        stop_events.append(stop_event)
        thread = threading.Thread(
            target=process_video,
            args=(video_path, source_id, stop_event, result_manager)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    video_paths = source["source"]["properties"]["urls"]
    video_paths = [path.replace("file://", "") for path in video_paths]
    result_file_path = "cfg/chessboard_detection_results.txt"
    process_multiple_videos(video_paths, result_file_path)