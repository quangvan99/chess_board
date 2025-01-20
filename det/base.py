import os
import math
import copy
import sys  # Add this import

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

from det.ops import PERF_DATA
import cv2
import configparser
import ast
from game.simple_engine.simple_engine import SimpleEngine
from game.game_state import GameState, Player
from game.chess_engine import ChessEngine
# Fix bug deepstream 7.0
os.system("rm -rf ~/.cache/gstreamer-1.0/registry.x86_64.bin")
class BoardStateBuffer:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.board_buffer = []
        self.current_stable_board = None
        
    def add_board(self, board):
        """Thêm một board state mới vào buffer"""
        self.board_buffer.append(board)
        if len(self.board_buffer) > self.buffer_size:
            self.board_buffer.pop(0)
                
    def get_stable_board(self, confidence_threshold=0.7, labels=None):
        """
        Tính toán board state ổn định từ buffer
        Returns: board state nếu đủ độ tin cậy, None nếu chưa
        """
        if len(self.board_buffer) < self.buffer_size:
            return None

        if labels is None:
            raise ValueError("Danh sách nhãn (labels) phải được cung cấp.")

        height, width = self.board_buffer[0].shape
        probability_board = {label: np.zeros((height, width)) for label in labels}

        for board in self.board_buffer:
            for i in range(height):
                for j in range(width):
                    label = board[i][j]
                    if label in labels:
                        probability_board[label][i, j] += 1

        for label in labels:
            probability_board[label] /= self.buffer_size

        new_board = np.full((height, width), "", dtype=object)
        for i in range(height):
            for j in range(width):
                max_prob = 0
                selected_label = ""
                for label, prob_matrix in probability_board.items():
                    if prob_matrix[i, j] > max_prob:
                        max_prob = prob_matrix[i, j]
                        selected_label = label
                if max_prob >= confidence_threshold:
                    new_board[i, j] = selected_label

        return new_board

class MoveRecorder:
    """Ghi nhận và xử lý các nước đi cờ"""
    def __init__(self):
        self.previous_board = None
        self.move_history = []
        self.board_buffer = BoardStateBuffer(buffer_size=10)
        self.simple_engine = SimpleEngine(None)
    def record_move(self, current_board):
        # Thêm board hiện tại vào buffer
        self.board_buffer.add_board(current_board)
        
        # Lấy board ổn định từ buffer
        labels_list = ["rk", "ra", "rb", "rr", "rc", "rn", "rp", "bk", "ba", "bb", "br", "bc", "bn", "bp"]
        stable_board = self.board_buffer.get_stable_board(0.7 ,labels_list)
        if stable_board is None:
            return None
            
        if self.previous_board is None:
            self.previous_board = stable_board
            return None
        print("stable_board", stable_board)
        print("previous_board", self.previous_board)
        if not self.is_board_change_valid(self.previous_board, stable_board):
            print("Board change is too large, keeping the old state.")
            return None
            
        move = self.detect_move_with_capture(self.previous_board, stable_board)
        if move:
            to_pos, from_pos, piece = move
            print(f"Detected move: {piece} from {from_pos} to {to_pos}")
            if self.simple_engine.move(self.previous_board, from_pos, to_pos):
                stable_board[to_pos[0], to_pos[1]] = piece
                self.previous_board = stable_board
            else:
                print("Invalid move, keeping the old state.")
                return None
            
        return move

    def is_board_change_valid(self, old_board, new_board, max_changes=2):
        changes = 0
        for i in range(len(old_board)):
            for j in range(len(old_board[i])):
                if old_board[i][j] != new_board[i][j]:
                    changes += 1
                if changes > max_changes:
                    print(f"Too many changes: {changes}")
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
                        start_position = (i, j)
                        moved_piece = old_piece
                    elif old_piece == '' and new_piece != '':
                        end_position = (i, j)
                    elif old_piece != '' and new_piece != '' and old_piece[0] != new_piece[0]:
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
            return (end_position, start_position, moved_piece)
        return None
class SourceState:
    """Maintain state for each video source"""
    def __init__(self):
        self.move_record = MoveRecorder()
        self.current_arrow = None
        self.previous_arrow = None
        self.arrow_source_id = None 
        self.move_history = []
        self.previous_board_processed = False
        self.current_player = None
        self.current_suggestion = None  # Thêm biến lưu suggestion hiện tại
        self.previous_suggest = None

class BasePipeline:
    def __init__(self):
        self.count = 0
        self.mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        self.source_states = {}
        self.config_point_path = "/home/project/chess_board/cfg/chessboard_detection_results.txt"
        self.chess_engine = ChessEngine()
        self.pipeline = None
        self.loop = None
        self.n_sources = 0
        self.perf_data = None  # Sẽ được khởi tạo trong create_pipeline_from_cfg
        self.str_pipe = None
        self.inputs = None

    def get_element(self, name):
        return self.pipeline.get_by_name(name)

    def cb_newpad(self, decodebin, pad, source_id):
        print("In cb_newpad\n")
        caps = pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()

        if gstname.find("video") != -1:
            pad_name = "sink_%u" % source_id
            sm = self.get_element("sm")
            # print(f"Attempting to link decodebin to streammux: {sm}\n")
            sinkpad = sm.get_static_pad(pad_name)
            if sinkpad:
                print(f"Pad '{pad_name}' exists")
            else:
                print(f"Pad '{pad_name}' doesn't exist")
                sinkpad = sm.get_request_pad(pad_name)
            
            if sinkpad and pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                print("Decodebin linked to pipeline successfully")
            else:
                print("Failed to link decodebin to pipeline")
                exit()

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        # Connect callback to internal decodebin signal
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)

        if "source" in name:
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property('drop-on-latency') is not None:
                Object.set_property("drop-on-latency", True)

    def create_source_bin(self, index, uri):
        # bin_name="source-bin-%02d"%index
        bin_name = f'src_{index}'
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", bin_name)
        if not uri_decode_bin:
            print(" Unable to create uri decode bin \n")
            exit()
        uri_decode_bin.set_property("uri",uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, index)
        uri_decode_bin.connect("child-added",self.decodebin_child_added, None)
        return uri_decode_bin
    
    def bus_call(self, bus, message, loop):
        
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error: %s: %s\n" % (err, debug))
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print("Pipeline state changed from {0:s} to {1:s}".format(
                    Gst.Element.state_get_name(old_state),
                    Gst.Element.state_get_name(new_state)))
        
        return True
    def intersection_area(self, box1, box2):
        """Calculate intersection area of two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def read_grid_points(self, config_point_path, section, key):
        config = configparser.ConfigParser()
        config.read(config_point_path)
        if not config.has_option(section, key):
            return []
        points_str = config.get(section, key)
        try:
            points = ast.literal_eval(points_str)
            return [tuple(map(int, point)) for point in points]  
        except Exception as e:
            print(f"Error parsing points: {e}")
            return []

    def create_display_meta(self, batch_meta, suggest_move, suggest_arrow, start_pos, end_pos, text, move_history=None):
        """Create display metadata for visualization"""
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        # Arrow configuration
        display_meta.num_arrows = 2
        arrow = display_meta.arrow_params[0]
        arrow.x1, arrow.y1 = start_pos
        arrow.x2, arrow.y2 = end_pos
        arrow.arrow_width = 4
        arrow.arrow_head = pyds.NvOSD_Arrow_Head_Direction.START_HEAD
        if text.startswith('r'):
            arrow.arrow_color.set(1.0, 0.0, 0.0, 0.5)
        else:
            arrow.arrow_color.set(0.0, 0.0, 0.0, 1.0)     
        if suggest_arrow:
            start_pos_suggest ,end_pos_suggest = suggest_arrow
            arrow = display_meta.arrow_params[1]
            arrow.x1, arrow.y1 = start_pos_suggest
            arrow.x2, arrow.y2 = end_pos_suggest
            arrow.arrow_width = 4
            arrow.arrow_head = pyds.NvOSD_Arrow_Head_Direction.START_HEAD
            arrow.arrow_color.set(1.0, 1.0, 1.0, 1.0)
        # Circle configuration
        display_meta.num_circles = 4
        circles = [
            (start_pos, 25, (0.0, 1.0, 0.0, 0.3)),  # Start outer
            (start_pos, 15, (0.0, 1.0, 0.0, 0.8)),  # Start inner
            (end_pos, 25, (1.0, 0.0, 0.0, 0.3)),    # End outer
            (end_pos, 15, (1.0, 0.0, 0.0, 0.8))     # End inner
        ]
        
        for i, (pos, radius, color) in enumerate(circles):
            circle = display_meta.circle_params[i]
            circle.xc, circle.yc = pos
            circle.radius = radius
            circle.circle_color.set(*color)

        # Tính toán số lượng text labels cần thiết
        num_texts = 2  # Start/End labels
        if move_history:
            num_texts += len(move_history)  # Move history
        if suggest_move:
            num_texts += 1  # Suggestion text
        
        display_meta.num_labels = num_texts
        text_params = []
        history_list = []
        # Add Start/End labels
        text_params.extend([
            {
                "text": "End",
                "pos": (start_pos[0] - 40, start_pos[1] - 55),
                "color": (0.0, 1.0, 0.0, 1.0),
                "size": 14
            },
            {
                "text": "Start",
                "pos": (end_pos[0] - 40, end_pos[1] - 55),
                "color": (1.0, 0.0, 0.0, 1.0),
                "size": 14
            }
        ])

        if move_history is not None:
            while len(move_history) > 10:
                move_history.pop(0)  

            history_list = [
                f"BUOC {move[3]}: {move[2]} {move[1]} -> {move[0]}"
                for move in move_history
            ]
        else:
            history_list = []
        combined_text = "\n".join(history_list)
        nvosdpadding = self.pipeline.get_by_name("nvosdpadding")

        if nvosdpadding:
            nvosdpadding.set_property("num-sources", suggest_move)
            nvosdpadding.set_property("padding-text", combined_text)
        for i, params in enumerate(text_params):
            text_param = display_meta.text_params[i]
            text_param.display_text = params["text"]
            text_param.x_offset, text_param.y_offset = params["pos"]
            text_param.font_params.font_size = params["size"]
            text_param.font_params.font_color.set(*params["color"])
            
        return display_meta
            

    def process_game_state(self, state, current_board, xc_yc_list, source_id):
        """
        Process game state and return display information
        Args:
            state: SourceState object
            current_board: Current board state
            xc_yc_list: List of coordinates
            source_id: ID of the video source
        """
        display_info = {
            'arrow': None,
            'suggest_move': state.current_suggestion,  
            'move_history': state.move_history,
            'suggest_arrow': None
        }

        # Xử lý nước đi mới
        if state.current_arrow is not None:
            start_pos, end_pos, text = state.current_arrow
            state.current_player = 'r' if text.startswith('b') else 'b'
            print(state.current_player)
            start_coords = xc_yc_list[start_pos[0] * 9 + start_pos[1]]
            end_coords = xc_yc_list[end_pos[0] * 9 + end_pos[1]]
            
            state.previous_arrow = state.current_arrow
            state.move_history.append((end_pos, start_pos, text, len(state.move_history) + 1))
            state.arrow_source_id = source_id
            state.previous_board_processed = False
            display_info['arrow'] = (start_coords, end_coords, text)

        # Tính toán gợi ý nước đi tiếp theo
        else:
            if not state.previous_board_processed and state.current_player is not None:
                # print("---------------------------current arrow:", state.current_arrow)
                game_state = GameState(state.move_record.previous_board)
                game_state.next_player = Player.RED if state.current_player == 'r' else Player.BLACK
                
                best_move = self.chess_engine.get_moves(game_state)
                if best_move:
                    state.current_suggestion, start_pos_suggest, end_pos_suggest = self._format_suggestion(best_move, state)
                    state.previous_suggest = (end_pos_suggest, start_pos_suggest)
                    start_coords_suggest = xc_yc_list[start_pos_suggest[0] * 9 + start_pos_suggest[1]]
                    end_coords_suggest = xc_yc_list[end_pos_suggest[0] * 9 + end_pos_suggest[1]]
                    display_info['suggest_move'] = state.current_suggestion
                    display_info['suggest_arrow'] = (start_coords_suggest, end_coords_suggest)
                
                state.arrow_source_id = source_id
                # state.current_player = 'b' if state.current_player == 'r' else 'r'
                state.previous_board_processed = True

        # Hiển thị arrow hiện tại
        if state.previous_arrow is not None and state.arrow_source_id == source_id:
            start_pos, end_pos, text = state.previous_arrow
            start_coords = xc_yc_list[start_pos[0] * 9 + start_pos[1]]
            end_coords = xc_yc_list[end_pos[0] * 9 + end_pos[1]]
            display_info['arrow'] = (start_coords, end_coords, text)
            if state.previous_suggest:
                start_pos_suggest, end_pos_suggest = state.previous_suggest
                start_coords_suggest = xc_yc_list[start_pos_suggest[0] * 9 + start_pos_suggest[1]]
                end_coords_suggest = xc_yc_list[end_pos_suggest[0] * 9 + end_pos_suggest[1]]
                display_info['suggest_arrow'] = (start_coords_suggest, end_coords_suggest)

        return display_info

    def tiler_sink_pad_buffer_probe(self, pad, info, u_data):
        """Callback function for processing each frame"""
        if self.perf_data is None:
            print("Warning: perf_data not initialized")
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                source_id = frame_meta.source_id
                
                if source_id not in self.source_states:
                    self.source_states[source_id] = SourceState()
                
                state = self.source_states[source_id]
                
                # Process objects in frame
                objects = self._process_objects(frame_meta)
                
                # Update performance data
                stream_index = f"stream{source_id}"
                self.perf_data.update_fps(stream_index)
                
                # Get grid points for this source
                xc_yc_list = self._read_grid_points(source_id)
                
                if xc_yc_list and objects:
                    # Create board from detected objects
                    board = self._create_board_from_objects(objects, xc_yc_list)
                    
                    # Record move
                    state.current_arrow = state.move_record.record_move(board)
                    
                    # Process game state and get display info
                    display_info = self.process_game_state(state, board, xc_yc_list, source_id)
                    
                    # Draw visualization if there's an arrow to display
                    if display_info['arrow']:
                        print("---------------------display info:", display_info)
                        start_coords, end_coords, text = display_info['arrow']
                        display_meta = self.create_display_meta(
                            batch_meta,
                            display_info['suggest_move'],
                            display_info['suggest_arrow'],
                            start_coords,
                            end_coords,
                            text,
                            display_info['move_history']
                        )
                        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                
                l_frame = l_frame.next
                
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _process_objects(self, frame_meta):
        """Extract objects from frame metadata"""
        objects = []
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                objects.append(obj_meta)
                obj_meta.text_params.display_text = f"{obj_meta.class_id}:{obj_meta.confidence*100:.0f}"
                l_obj = l_obj.next
            except StopIteration:
                break
        return objects

    def _read_grid_points(self, source_id):
        """Read grid points for specific source from config"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_point_path)
            section = f"source_id_{source_id}"
            if section in config:
                points_str = config[section]["grid_points"]
                return ast.literal_eval(points_str)
        except Exception as e:
            print(f"Error reading grid points: {e}")
        return None

    def create_pipeline_from_cfg(self, str_pipe, name_pipe="pipeline"):
        """Create and configure the GStreamer pipeline"""
        Gst.init(None)

        self.str_pipe = str_pipe
        self.inputs = str_pipe['source']['properties']['urls']
        self.n_sources = len(self.inputs)
        self.perf_data = PERF_DATA(self.n_sources)  # Khởi tạo PERF_DATA sau khi có n_sources
        
        self.pipeline = Gst.Pipeline.new(name_pipe)

        # Create elements 
        elements = {}
        for el in str_pipe:
            if el == 'source':
                continue
            elements[el] = Gst.ElementFactory.make(str_pipe[el]['plugin'], el)
            for k,v in str_pipe[el]['properties'].items():
                if k == "batch-size":
                    elements[el].set_property(k, self.n_sources)
                elif k == "rows":
                    elements[el].set_property(k, int(math.sqrt(self.n_sources)))
                elif k == "columns":
                    elements[el].set_property(k, int(math.ceil((1.0*self.n_sources)/int(math.sqrt(self.n_sources)))))
                elif k == "nvbuf-memory-type":
                    elements[el].set_property(k, self.mem_type)
                elif k == "caps": 
                    elements[el].set_property(k, Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
                else:
                    elements[el].set_property(k, v)

        # add source and streammux
        self.pipeline.add(elements["sm"])
        for i, location in enumerate(self.inputs):
            source_bin = self.create_source_bin(i, location)
            self.pipeline.add(source_bin)

        # add elements
        names = list(elements.keys())
        for i in range(len(names)):
            if names[i].startswith(("src", "sm")):
                continue
            self.pipeline.add(elements[names[i]])

        # link elements
        for i in range(len(names) - 1):
            elements[names[i]].link(elements[names[i+1]])
            print(f"Linking {names[i]} to {names[i+1]}")
        
        return self.pipeline

    def run(self, pipeline):
        if not isinstance(pipeline, Gst.Pipeline):
            self.pipeline = self.create_pipeline_from_cfg(pipeline)
        self.loop = GLib.MainLoop() # Create a mainloop
        bus = self.pipeline.get_bus() # Retrieve the bus from the pipeline
        bus.add_signal_watch() # Add a watch for new messages on the bus
        bus.connect("message", self.bus_call, self.loop) # Connect the loop to the callback function

        tiler_sink_pad = self.pipeline.get_by_name("nvtracker").get_static_pad("src")
        if not tiler_sink_pad:
            print(" Unable to get src pad \n")
        else:
            ##
            tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)
            GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except:
            pass
        # cleaning up as the pipeline comes to an end
        self.pipeline.set_state(Gst.State.NULL)
    
    def release_pipeline(self):
        self.pipeline.set_state(Gst.State.NULL)
        del self.pipeline
        del self.perf_data
        del self.inputs
        del self.n_sources
        del self.str_pipe

    def _create_board_from_objects(self, objects, xc_yc_list):
        """
        Tạo bảng cờ từ các object được phát hiện
        Args:
            objects: Danh sách các object được phát hiện
            xc_yc_list: Danh sách tọa độ các ô cờ
        Returns:
            numpy.ndarray: Bảng cờ với các quân cờ được đặt đúng vị trí
        """
        # Khởi tạo bảng cờ trống
        board = np.full((10, 9), "", dtype=object)
        
        # Xử lý từng object
        for obj_meta in objects:
            # Lấy tọa độ tâm của object
            rect_params = obj_meta.rect_params
            center_x = rect_params.left + rect_params.width / 2
            center_y = rect_params.top + rect_params.height / 2
            
            # Tìm ô cờ gần nhất
            min_distance = float('inf')
            closest_cell = None
            
            for i, (xc, yc) in enumerate(xc_yc_list):
                distance = math.sqrt((center_x - xc)**2 + (center_y - yc)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_cell = i
            
            if closest_cell is not None:
                # Chuyển đổi index 1D thành tọa độ 2D trên bàn cờ
                row = closest_cell // 9
                col = closest_cell % 9
                
                # Lấy tên class của object
                obj_label = obj_meta.obj_label
                
                # Cập nhật bảng cờ
                board[row][col] = obj_label
        
        return board

    def _format_suggestion(self, best_move, state):
        """
        Format nước đi gợi ý thành text
        Args:
            best_move: Nước đi tốt nhất từ chess engine
            state: Trạng thái hiện tại
        Returns:
            str: Text mô tả nước đi gợi ý
        """
        try:
            from_x = ord(best_move[0]) - ord("a")
            from_y = 9 - int(best_move[1])
            to_x = ord(best_move[2]) - ord("a")
            to_y = 9 - int(best_move[3])

            piece = state.move_record.previous_board[from_y][from_x]
            return f"BUOC: {piece} from {from_y, from_x} to {to_y, to_x}", (from_y, from_x), (to_y, to_x)
        except Exception as e:
            print(f"Error formatting suggestion: {e}")
            return ""