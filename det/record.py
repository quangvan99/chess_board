class MoveRecorder:
    """Ghi nhận và xử lý các nước đi cờ"""
    def __init__(self):
        self.previous_board = None
        self.move_history = []

    def record_move(self, current_board):
        if self.previous_board is None:
            self.previous_board = current_board  # First time, update directly
            return None

        if not self.is_board_change_valid(self.previous_board, current_board):
            print("Board change is too large, keeping the old state.")
            return None

        move = self.detect_move_with_capture(self.previous_board, current_board)
        if move:
            self.previous_board = current_board  # Update current board state
        
        return move

    def is_board_change_valid(self, old_board, new_board, max_changes=2):
        changes = 0
        for i in range(len(old_board)):
            for j in range(len(old_board[i])):
                if old_board[i][j] != new_board[i][j]:
                    changes += 1
                if changes > max_changes:
                    return False
        return True

    def detect_move(self, old_board, new_board):
        start_position = None
        end_position = None
        moved_piece = None

        for i in range(len(old_board)):
            for j in range(len(old_board[i])):
                old_piece = old_board[i][j]
                new_piece = new_board[i][j]

                if old_piece != new_piece:
                    if old_piece != '' and new_piece == '':
                        start_position = (i, j)
                        moved_piece = old_piece
                    if old_piece == '' and new_piece != '':
                        end_position = (i, j)
                    else:
                        end_position = (i, j)
        print(start_position, end_position, moved_piece)
        if start_position and end_position and moved_piece:
            return (start_position, end_position, moved_piece)
        return None
    
    def detect_move_with_capture(self, old_board, new_board):
        start_position = None
        end_position = None
        moved_piece = None
        captured_piece = None

        for i in range(len(old_board)):
            for j in range(len(old_board[i])):
                old_piece = old_board[i][j]
                new_piece = new_board[i][j]

                if old_piece != new_piece:
                    if old_piece != '' and new_piece == '':
                        # Vị trí bắt đầu (quân cờ di chuyển từ đây)
                        start_position = (i, j)
                        moved_piece = old_piece
                    elif old_piece == '' and new_piece != '':
                        # Vị trí đích (quân cờ di chuyển tới đây)
                        end_position = (i, j)
                    elif old_piece != '' and new_piece != '' and old_piece[0] != new_piece[0]:
                        # Phát hiện bắt quân: quân cờ đối thủ bị thay thế bởi quân cờ mới
                        # start_position = (i, j)
                        end_position = (i, j)
                        captured_piece = old_piece
                        moved_piece = new_piece
        # if captured_piece and start_position:
        #     end_position = start_position
        #     for i in range(len(old_board)):
        #         for j in range(len(old_board[i])):
        #             if old_board[i][j] == captured_piece:
        #                 end_position = (i, j)
        #                 break
        # Kết quả bao gồm vị trí bắt đầu, vị trí đích, quân cờ đã di chuyển, và quân cờ bị ăn (nếu có)
        if start_position and end_position and moved_piece:
            return (start_position, end_position, moved_piece)
        return None