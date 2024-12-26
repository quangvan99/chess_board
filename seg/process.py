import cv2
from seg.yolo.yoloseg import YOLOSeg
import numpy as np
def warp_image(image, src_points, dst_size=(500, 500)):
    """
    Thực hiện phép biến đổi phối cảnh (warp perspective) cho ảnh dựa trên các điểm nguồn và kích thước đích.

    Parameters:
        image (np.ndarray): Ảnh nguồn cần warp.
        src_points (np.ndarray): Mảng các điểm nguồn (4 điểm) cần warp.
        dst_size (tuple): Kích thước (width, height) của ảnh đích sau khi warp. Mặc định là (500, 500).

    Returns:
        np.ndarray: Ảnh sau khi thực hiện phép biến đổi phối cảnh.
        np.ndarray: Ma trận biến đổi phối cảnh đã sử dụng.
    """
    width, height = dst_size
    # Định nghĩa các điểm đích (destination points)
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Tính toán ma trận biến đổi phối cảnh
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Thực hiện warp perspective
    warped_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped_image, matrix
def find_chessboard_quadrilateral(image, mask_maps):
    """
    Tìm tứ giác của bàn cờ trên ảnh và thực hiện warp ảnh để có góc nhìn trực diện.

    Parameters:
        image (np.ndarray): Ảnh nguồn chứa bàn cờ.
        mask_maps (list hoặc np.ndarray): Danh sách các mask để xác định vùng bàn cờ.

    Returns:
        tuple or None: (warped_image, warp_matrix) nếu tìm thấy tứ giác, ngược lại trả về None.
    """
    # Kiểm tra xem có mask nào không
    if isinstance(mask_maps, list):
        mask_maps = np.array(mask_maps)

    if mask_maps.shape[0] == 0:
        print("Không có mask nào được cung cấp!")
        return [], None

    # Kết hợp tất cả các mask thành một mask duy nhất
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(mask_maps.shape[0]):
        mask = mask_maps[i]
        mask = (mask > 0.5).astype(np.uint8) * 255  # Chuyển đổi sang định dạng nhị phân
        full_mask = cv2.bitwise_or(full_mask, mask)  # Kết hợp mask

    # Tìm các đường viền từ mask nhị phân
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Không tìm thấy đường viền nào!")
        return [], None

    # Chọn đường viền lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)

    # Tìm tứ giác bằng cách xấp xỉ đa giác
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Kiểm tra xem có phải là tứ giác không
    if len(approx) != 4:
        print("Không tìm thấy tứ giác chính xác!")
        return [], None

    board_corners = approx.reshape(4, 2)

    # Sắp xếp lại các góc của tứ giác theo thứ tự: trên-left, trên-right, dưới-right, dưới-left
    board_corners = sort_corners(board_corners)

    warped_image, warp_matrix = warp_image(image, board_corners)

    return board_corners, warped_image

def sort_corners(corners):
    """
    Sắp xếp các góc của tứ giác theo thứ tự: trên-left, trên-right, dưới-right, dưới-left.

    Parameters:
        corners (np.ndarray): Mảng các điểm góc (4 điểm).

    Returns:
        np.ndarray: Mảng các điểm góc đã được sắp xếp.
    """
    # Tính tổng và hiệu của các điểm để xác định góc
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]

    return np.float32([top_left, top_right, bottom_right, bottom_left])


class YOLOModel:
    """Quản lý mô hình YOLO để phát hiện bàn cờ và quân cờ"""
    def __init__(self):
        self.aligner = YOLOSeg(
            path= "seg/weights/board_mask.onnx",
            conf_thres = 0.95,
            iou_thres = 0.6,
            use_gpu= False,
            num_masks= 32
        )
    def detect_chessboard(self, frame):
        boxes, scores, class_ids, masks = self.aligner(frame)
        if masks is not None:
            corners, warped_image = find_chessboard_quadrilateral(frame, masks)
            return corners, warped_image
        else:
            return [], None