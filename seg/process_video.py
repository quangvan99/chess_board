import cv2
import threading
from seg.process import YOLOModel
from seg.yolo.xla import find_horizontal_vertical_lines_and_intersections
from cfg.src import source

lock = threading.Lock()

def update_result_file(result_file_path, source_id, video_path, corners=None, grid_points=None):
    with lock: 
        with open(result_file_path, "r") as result_file:
            lines = result_file.readlines()

        updated_lines = []
        found_source = False

        for line in lines:
            if line.strip() == f"[source_id_{source_id}]":
                found_source = True
                updated_lines.append(line) 
                updated_lines.append(f"video_path = {video_path}\n")
                if corners is not None and grid_points is not None:
                    formatted_corners = ', '.join([f"{int(coord)}" for corner in corners for coord in corner])
                    updated_lines.append(f"corners = {formatted_corners}\n")
                    updated_lines.append(f"grid_points = [{grid_points}]\n\n")
                else:
                    updated_lines.append(f"corners = None\n")
                    updated_lines.append(f"grid_points = None\n\n")
            elif found_source and line.startswith("video_path"):
                continue  
            elif found_source and (line.startswith("corners") or line.startswith("grid_points")):
                continue  
            else:
                updated_lines.append(line)

        if not found_source:
            updated_lines.append(f"[source_id_{source_id}]\n")
            updated_lines.append(f"video_path = {video_path}\n")
            updated_lines.append(f"corners = None\n")
            updated_lines.append(f"grid_points = None\n\n")

        with open(result_file_path, "w") as result_file:
            result_file.writelines(updated_lines)


def process_video(video_path, source_id, stop_event, result_file_path):
    yolo_model = YOLOModel()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    corners_found = False
    grid_points_found = False
    
    while cap.isOpened():
        if stop_event.is_set():
            print(f"Stopping video: {video_path}")
            break

        ret, frame = cap.read()
        if not ret:
            break

        corners, warped_image = yolo_model.detect_chessboard(frame)
        
        if len(corners) > 0:
            grid_points = find_horizontal_vertical_lines_and_intersections(warped_image)
            
            if grid_points is not None and len(grid_points) > 0:
                update_result_file(result_file_path, source_id, video_path, corners, grid_points)
                stop_event.set()
                corners_found = True
                grid_points_found = True
                break
            else:
                print(f"Grid points not found yet in video: {video_path}")
    
    if not corners_found or not grid_points_found:
        update_result_file(result_file_path, source_id, video_path)


def process_multiple_videos(video_paths, result_file_path):
    threads = []
    stop_events = []

    with open(result_file_path, "w") as file:
        for source_id, video_path in enumerate(video_paths):
            file.write(f"[source_id_{source_id}]\n")
            file.write(f"video_path = {video_path}\n")
            file.write(f"corners = None\n")
            file.write(f"grid_points = None\n\n")

    for source_id, video_path in enumerate(video_paths):
        stop_event = threading.Event()
        stop_events.append(stop_event)
        thread = threading.Thread(target=process_video, args=(video_path, source_id, stop_event, result_file_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    video_paths = source["source"]["properties"]["urls"]
    video_paths = [path.replace("file://", "") for path in video_paths]
    result_file_path = "cfg/chessboard_detection_results.txt"
    process_multiple_videos(video_paths, result_file_path)
