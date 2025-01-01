import subprocess
from game.game_state import GameState, Player
from config.config import cfg
import requests
class ChessEngine:
    MAX_RESTART_TIMES = 10
    def __init__(self):
        self.engine_path = cfg.model['engine']['engine_path']
        self._engine_process = self.open_engine(self.engine_path)
        self.current_fen = "startpos"
        self.current_time_limit = 2000
        self.engine_restart_times = 0

    @property
    def engine_process(self):
        """Returns the engine process. If it is not running, it will be started."""
        if self._engine_process is None or self._engine_process.poll() is not None:
            if self._engine_process.poll() is not None:
                print("There was an error with the engine or there was a wrong move.")
                # error = self._engine_process.stderr.read().decode()
                # print(error)
            print("Engine is not running, starting it...")
            if self.engine_restart_times >= self.MAX_RESTART_TIMES:
                raise Exception("Engine is not running.")
            self._engine_process = self.open_engine(self.engine_path)
            self._engine_process.stdin.write(f"position fen {self.current_fen}\n".encode())
            self._engine_process.stdin.write(f"go movetime {self.current_time_limit}\n".encode())
            self._engine_process.stdin.flush()
            self.engine_restart_times += 1
        return self._engine_process

    @staticmethod
    def open_engine(engine_path):
        """Prepares the engine for use."""
        engine_process = subprocess.Popen(
            engine_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return engine_process

    def engine_write(self, command):
        """Sends a command to the engine."""
        command = f"{command}\n".encode()
        self.engine_process.stdin.write(command)
        self.engine_process.stdin.flush()

    def engine_read(self, timeout=1000):
        """Reads the engine output."""
        text = ""
        while True:
            if self.engine_process.poll() is None:
                text = self.engine_process.stdout.read().decode()
                if text != "":
                    break
        return text
    def close_engine(self):
        """Terminates the engine process."""
        if self._engine_process and self._engine_process.poll() is None:
            self._engine_process.terminate()
            self._engine_process = None

    @staticmethod
    def board_array_to_fen(board_array, next_player):
        """Converts a board array to a FEN string for Xiangqi."""
        fen = ""
        
        for row in board_array:
            empty = 0  # Số ô trống liên tiếp
            for square in row:
                if square == "":  # Nếu ô trống
                    empty += 1
                else:
                    if empty != 0:  # Nếu có ô trống trước đó
                        fen += str(empty)
                        empty = 0
                    # Kiểm tra xem quân cờ là đỏ hay đen và chuyển thành ký tự FEN
                    is_red = square[0] == "r"
                    fen += square[1].upper() if is_red else square[1].lower()
            if empty != 0:  # Thêm số lượng ô trống cuối cùng nếu có
                fen += str(empty)
            fen += "/"  # Kết thúc mỗi hàng với "/"
        
        # Xác định người chơi tiếp theo
        player_str = "w" if next_player == Player.RED else "b"
        return fen[:-1] + f" {player_str} - - 0 1"

    CDB_API_ENDPOINT = "http://www.chessdb.cn/chessdb.php"

    def query_cdb(self, action, params):
        """
        Gửi yêu cầu tới Xiangqi Cloud Database API.
        
        Args:
            action (str): Loại hành động, ví dụ: 'queryall', 'querybest'.
            params (dict): Các tham số yêu cầu.

        Returns:
            dict hoặc str: Phản hồi từ API.
        """
        params['action'] = action
        response = requests.get(self.CDB_API_ENDPOINT, params=params)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error querying CDB: {response.status_code}, {response.text}")

    def query_best_move(self, fen):
        params = {
            'board': fen,
            'learn': 1  # Bật chế độ học tự động
        }
        response = self.query_cdb("querybest", params)
        if response.startswith("bestmove"):
            return response.split(":")[1].strip()
        elif response == "nobestmove":
            return None
        elif response == "invalid board":
            raise ValueError("Invalid FEN provided to CDB.")
        else:
            print(f"Unexpected response: {response}")
            return None

    def query_all_moves(self, fen):
        params = {
            'board': fen,
            'showall': 1  # Hiển thị tất cả nước đi, kể cả nước chưa biết
        }
        response = self.query_cdb("queryall", params)
        if response == "unknown":
            return []
        elif response == "invalid board":
            raise ValueError("Invalid FEN provided to CDB.")
        else:
            return [move.split(",")[0].split(":")[1] for move in response.split("|")]

    # (các phương thức hiện tại giữ nguyên)
    # def get_move(self, game_state, time_limit=2000):
    #     """Returns the best move for the given board array."""
    #     board_array = game_state.board
    #     next_player = game_state.next_player
    #     fen = self.board_array_to_fen(board_array, next_player)
    #     print(fen)
    #     self.engine_write(f"position fen {fen}")
    #     self.engine_write(f"go movetime {time_limit}")
    #     self.current_fen = fen
    #     self.current_time_limit = time_limit

    #     best_move = None
    #     while True:
    #         if self.engine_process.poll() is None:
    #             text = self.engine_process.stdout.readline().decode()
    #             if text.startswith("bestmove"):
    #                 best_move = text.split(" ")[1]
    #                 break
    #         else:
    #             print("Engine is not running...")
    #             break
    #     return best_move
    def get_moves(self, game_state, time_limit=2000, use_cdb=False, get_all=False):
        """Returns the best move for the given board array."""
        board_array = game_state.board
        next_player = game_state.next_player
        fen = self.board_array_to_fen(board_array, next_player)
        print(f"FEN: {fen}")

        if use_cdb:
            if get_all:
                # Sử dụng CDB để lấy tất cả các nước đi
                all_moves = self.query_all_moves(fen)
                if all_moves:
                    print(f"All moves from CDB: {all_moves}")
                    return all_moves[:5]
                else:
                    print("No moves from CDB. Falling back to local engine.")
            else:
                # Sử dụng CDB để lấy nước đi tốt nhất
                best_move = self.query_best_move(fen)
                if best_move:
                    print(f"Best move from CDB: {best_move}")
                    return best_move
                else:
                    print("No best move from CDB. Falling back to local engine.")

        # Sử dụng engine nếu không có dữ liệu từ CDB
        self.engine_write(f"position fen {fen}")
        if get_all:
            self.engine_write("d")  # Yêu cầu danh sách tất cả nước đi từ engine
            moves = []
            while True:
                text = self.engine_process.stdout.readline().decode()
                if "Legal moves" in text:
                    moves = text.split(":")[1].strip().split()
                    break
            print(f"All moves from engine: {moves}")
            return moves
        else:
            self.engine_write(f"go movetime {time_limit}")
            self.current_fen = fen
            self.current_time_limit = time_limit

            best_move = None
            while True:
                if self.engine_process.poll() is None:
                    text = self.engine_process.stdout.readline().decode()
                    if text.startswith("bestmove"):
                        best_move = text.split(" ")[1]
                        break
                else:
                    print("Engine is not running...")
                    break
            print(f"Best move from engine: {best_move}")
            return best_move