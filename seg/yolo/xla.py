import cv2
import numpy as np
from scipy.spatial import KDTree

def filter_close_points(points, distance_threshold=20):
    """ 
    Hàm lọc các điểm gần nhau dựa vào ngưỡng khoảng cách, giữ lại trung bình của những điểm gần nhau.
    """
    filtered_points = []
    while points:
        current_point = points.pop(0)
        close_points = [current_point]

        points_to_remove = []
        for point in points:
            if np.linalg.norm(np.array(current_point) - np.array(point)) < distance_threshold:
                close_points.append(point)
                points_to_remove.append(point)

        # Tính điểm trung bình và thêm vào danh sách kết quả
        average_point = np.mean(close_points, axis=0).astype(int)
        filtered_points.append(tuple(average_point))

        # Loại bỏ các điểm gần nhau ra khỏi danh sách points bằng cách so sánh mảng Numpy
        points = [p for p in points if not any(np.array_equal(p, remove_p) for remove_p in points_to_remove)]

    return filtered_points

def filter_close_points_optimized(points, distance_threshold=20):
    """
    Optimized version of the filter_close_points function using KDTree for efficiency.
    """
    if not points:
        return []

    points = np.array(points)
    tree = KDTree(points)
    filtered_points = []
    visited = np.zeros(len(points), dtype=bool)  # Track visited points

    for i, point in enumerate(points):
        if visited[i]:
            continue

        # Find all points within the distance_threshold
        idxs = tree.query_ball_point(point, distance_threshold)
        close_points = points[idxs]

        # Mark these points as visited
        visited[idxs] = True

        # Compute the mean of the close points
        average_point = np.mean(close_points, axis=0).astype(int)
        filtered_points.append(tuple(average_point))

    return filtered_points

def remove_outside_points(points, x_min, x_max, y_min, y_max):
    """
    Lọc các điểm ra ngoài khu vực bàn cờ (giới hạn bởi x_min, x_max, y_min, y_max).
    """
    filtered_points = [point for point in points if x_min < point[0] < x_max and y_min < point[1] < y_max]
    return filtered_points
def expand_grid(points):
    """
    Mở rộng lưới từ các điểm hiện tại, thêm một vòng điểm bao ngoài.
    """
    if len(points) == 56:
        points = np.array(points)
        xs, ys = points[:, 0], points[:, 1]
        
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        
        unique_xs = sorted(np.unique(xs))
        unique_ys = sorted(np.unique(ys))

        x_spacing = 0
        for i in range(1, len(unique_xs)):
            spacing = unique_xs[i] - unique_xs[i - 1]
            if spacing >= 20:
                x_spacing = spacing
                break

        y_spacing = 0
        for i in range(1, len(unique_ys)):
            spacing = unique_ys[i] - unique_ys[i - 1]
            if spacing >= 20:
                y_spacing = spacing
                break

        if x_spacing == 0 or y_spacing == 0:
            raise ValueError("Không tìm được khoảng cách hợp lệ để mở rộng lưới.")

        new_points = []

        for x in unique_xs:
            new_points.append((x, min_y - y_spacing))  # Dòng trên
            new_points.append((x, max_y + y_spacing))  # Dòng dưới

        for y in unique_ys:
            new_points.append((min_x - x_spacing, y))  # Cột trái
            new_points.append((max_x + x_spacing, y))  # Cột phải

        new_points.append((min_x - x_spacing, min_y - y_spacing))  # Góc trên trái
        new_points.append((max_x + x_spacing, min_y - y_spacing))  # Góc trên phải
        new_points.append((min_x - x_spacing, max_y + y_spacing))  # Góc dưới trái
        new_points.append((max_x + x_spacing, max_y + y_spacing))  # Góc dưới phải

        # Hợp nhất các điểm cũ và mới
        expanded_grid = list(points) + new_points
        return expanded_grid

    else:
        print("Không đủ 56 điểm để mở rộng lưới!")
        return points

def find_horizontal_vertical_lines_and_intersections(image):
    rotate_90 = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    h_lines = []
    v_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if abs(theta) < np.pi / 36 or abs(theta - np.pi) < np.pi / 36:
                h_lines.append((rho, theta))
            elif abs(theta - np.pi / 2) < np.pi / 36: 
                v_lines.append((rho, theta))

    intersections = find_intersections(h_lines, v_lines)

    x_min, x_max = 60, image.shape[1] - 60
    y_min, y_max = 60, image.shape[0] - 60 
    filtered_intersections = remove_outside_points(intersections, x_min, x_max, y_min, y_max)
    filtered_intersections = filter_close_points_optimized(filtered_intersections)
    threshold = 12  # Ngưỡng thay đổi
    first_increase_index = None  # Khởi tạo biến để lưu chỉ số đầu tiên

    # Lặp qua các phần tử và kiểm tra sự thay đổi của x so với phần tử đầu tiên
    for i in range(1, len(filtered_intersections)):
        x_diff = abs(filtered_intersections[i][0] - filtered_intersections[0][0])  # Tính độ chênh lệch của x với phần tử đầu tiên
        if x_diff > threshold:
            first_increase_index = i  # Lưu chỉ số đầu tiên
            break  # Dừng lại sau khi tìm thấy phần tử đầu tiên

    # In ra kết quả
    if first_increase_index is not None:
        print(f"First element index with significant increase: {first_increase_index}, Value: {filtered_intersections[first_increase_index]} (x: {filtered_intersections[first_increase_index][0]})")
    else:
        print("No significant increase found.")
    if first_increase_index is not None and first_increase_index == 7:
        # Xoay ảnh 90 độ
        rotate_90 = True
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # Cập nhật tọa độ
        height, width = image.shape[:2]
        updated_points = []
        for (x, y) in filtered_intersections:
            # Cập nhật tọa độ mới
            new_x = y  # Xoay 90 độ, y trở thành x
            new_y = width - x  # Cập nhật y
            updated_points.append((new_x, new_y))
        filtered_intersections = updated_points    
    if len(filtered_intersections) == 63:
        filtered_intersections.sort(key=lambda x: x[1])
        # print(filtered_intersections)
        filtered_intersections = np.array(filtered_intersections)
        
        mid_start = len(filtered_intersections) // 2 - 3  
        mid_end = len(filtered_intersections) // 2 + 4    

        filtered_intersections = np.delete(filtered_intersections, slice(mid_start, mid_end), axis=0)

    expanded_grid_points = expand_grid(filtered_intersections)
    expanded_grid_points = filter_close_points_optimized(expanded_grid_points)

    if len(expanded_grid_points) != 90:
        return None
    return expanded_grid_points

def find_intersections(h_lines, v_lines):
    intersections = []
    
    for rho_h, theta_h in h_lines:
        for rho_v, theta_v in v_lines:
            A = np.array([
                [np.cos(theta_h), np.sin(theta_h)],
                [np.cos(theta_v), np.sin(theta_v)]
            ])
            b = np.array([[rho_h], [rho_v]])
            
            try:
                x, y = np.linalg.solve(A, b)
                intersections.append((int(np.round(x)), int(np.round(y))))
            except np.linalg.LinAlgError:
                pass

    return intersections