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
    def __init__(self):
        self.move_record = MoveRecorder()
        self.current_arrow = None
        self.previous_arrow = None
        self.arrow_source_id = None 
        self.move_history = []

class BasePipeline:
    def __init__(self):
        self.count = 0
        self.mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        # Dictionary để lưu trữ trạng thái cho từng source
        self.source_states = {}
        self.config_point_path =  "/home/project/chess_board/cfg/chessboard_detection_results.txt"

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

    def tiler_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id
            if source_id not in self.source_states:
                self.source_states[source_id] = SourceState()
            state = self.source_states[source_id]

            surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            objects = []
            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    objects.append(obj_meta)
                    obj_meta.text_params.display_text = f"{obj_meta.class_id}:{obj_meta.confidence*100:.0f}"

                except StopIteration:
                    break

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                stream_index = f"stream{source_id}"
                self.perf_data.update_fps(stream_index)

                xc_yc_list = self.read_grid_points(self.config_point_path, 
                                                f"source_id_{source_id}", 
                                                "grid_points")

                if xc_yc_list is not None and len(xc_yc_list) > 0:
                    board = []
                    for rect in xc_yc_list:
                        is_inside_bbox = False
                        x_center, y_center = rect
                        for obj in objects:
                            rect_params = obj.rect_params
                            if (rect_params.left < x_center < rect_params.left + rect_params.width and 
                                rect_params.top < y_center < rect_params.top + rect_params.height):
                                class_id = obj.class_id
                                class_name = obj.obj_label
                                is_inside_bbox = True
                                break
                        board.append(class_name if is_inside_bbox else "")

                    board = np.array(board).reshape(10, 9)
                    state.current_arrow = state.move_record.record_move(board)
                    # print("previous_board", state.move_record.previous_board)
                    # print("current_board", board)
                    
                    if state.current_arrow is not None:
                        
                        start_pos, end_pos, text = state.current_arrow
                        start_pos = (xc_yc_list[start_pos[0] * 9 + start_pos[1]])
                        end_pos = (xc_yc_list[end_pos[0] * 9 + end_pos[1]])
                        state.previous_arrow = state.current_arrow
                        state.arrow_source_id = source_id  
                    elif state.previous_arrow is not None and state.arrow_source_id == source_id:
                        print("***********************************************************************************************************")
                        print("previous_arrow", state.previous_arrow)
                        print("current_arrow", state.current_arrow)
                        start_pos, end_pos, text = state.previous_arrow
                        start_pos = (xc_yc_list[start_pos[0] * 9 + start_pos[1]])
                        end_pos = (xc_yc_list[end_pos[0] * 9 + end_pos[1]])

                    if ('start_pos' in locals() and 'end_pos' in locals() and 
                        state.arrow_source_id == source_id):
                        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                        current_move = (end_pos, start_pos)  # end_pos là start thực tế
                        if not state.move_history or state.move_history[-1] != current_move:
                            state.move_history.append(current_move)
                        
                        # Arrow params
                        display_meta.num_arrows = 1
                        display_meta.arrow_params[0].x1 = start_pos[0]
                        display_meta.arrow_params[0].y1 = start_pos[1] 
                        display_meta.arrow_params[0].x2 = end_pos[0]
                        display_meta.arrow_params[0].y2 = end_pos[1]
                        display_meta.arrow_params[0].arrow_width = 4 
                        display_meta.arrow_params[0].arrow_head = pyds.NvOSD_Arrow_Head_Direction.START_HEAD  
                        display_meta.arrow_params[0].arrow_color.set(1.0, 1.0, 1.0, 0.8)
                        
                        # Circle params
                        display_meta.num_circles = 4  
                        
                        display_meta.circle_params[0].xc = start_pos[0]
                        display_meta.circle_params[0].yc = start_pos[1]
                        display_meta.circle_params[0].radius = 25 
                        display_meta.circle_params[0].circle_color.set(0.0, 1.0, 0.0, 0.3)
                        
                        display_meta.circle_params[1].xc = start_pos[0]
                        display_meta.circle_params[1].yc = start_pos[1]
                        display_meta.circle_params[1].radius = 15 
                        display_meta.circle_params[1].circle_color.set(0.0, 1.0, 0.0, 0.8)
                        
                        display_meta.circle_params[2].xc = end_pos[0]
                        display_meta.circle_params[2].yc = end_pos[1]
                        display_meta.circle_params[2].radius = 25 
                        display_meta.circle_params[2].circle_color.set(1.0, 0.0, 0.0, 0.3)
                        
                        display_meta.circle_params[3].xc = end_pos[0]
                        display_meta.circle_params[3].yc = end_pos[1]
                        display_meta.circle_params[3].radius = 15 
                        display_meta.circle_params[3].circle_color.set(1.0, 0.0, 0.0, 0.8)

                        # Text params for move history
                        padding_x_offset = 10
                        padding_y_offset = 20
                        line_spacing = 30

                        # Số lượng text = 2 (Start/End) + số lượng moves trong history
                        display_meta.num_labels = len(state.move_history) + 2
                        
                        # Text "End" và "Start" cho current move
                        display_meta.text_params[0].display_text = "End"
                        display_meta.text_params[0].x_offset = start_pos[0] - 20
                        display_meta.text_params[0].y_offset = start_pos[1] - 35
                        display_meta.text_params[0].font_params.font_size = 14
                        display_meta.text_params[0].font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                        
                        display_meta.text_params[1].display_text = "Start"
                        display_meta.text_params[1].x_offset = end_pos[0] - 20
                        display_meta.text_params[1].y_offset = end_pos[1] - 35
                        display_meta.text_params[1].font_params.font_size = 14
                        display_meta.text_params[1].font_params.font_color.set(1.0, 0.0, 0.0, 1.0)

                        frame_width = frame_meta.source_frame_width
                        print("*********************frame_width", frame_width)
                        # Hiển thị lịch sử moves trong padding
                        for idx, (hist_start, hist_end) in enumerate(state.move_history):
                            move_text = f"Move {idx + 1}: Start({hist_start[0]:.1f}, {hist_start[1]:.1f}) -> End({hist_end[0]:.1f}, {hist_end[1]:.1f})"
                            display_meta.text_params[idx + 2].display_text = move_text
                            display_meta.text_params[idx + 2].x_offset = 640 + padding_x_offset 
                            display_meta.text_params[idx + 2].y_offset = padding_y_offset + (idx * line_spacing)
                            display_meta.text_params[idx + 2].font_params.font_size = 14
                            display_meta.text_params[idx + 2].font_params.font_color.set(0.0, 0.0, 0.0, 1.0);


                        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                # # Vẽ grid points chỉ cho source_id hiện tại
                # MAX_CIRCLES = 16
                # display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                # num_points = min(len(xc_yc_list), MAX_CIRCLES)
                # display_meta.num_circles = num_points

                # for i in range(num_points):
                #     x, y = xc_yc_list[i]
                #     display_meta.circle_params[i].xc = x
                #     display_meta.circle_params[i].yc = y
                #     display_meta.circle_params[i].radius = 3
                #     display_meta.circle_params[i].circle_color.red = 0.0
                #     display_meta.circle_params[i].circle_color.green = 1.0
                #     display_meta.circle_params[i].circle_color.blue = 0.0
                #     display_meta.circle_params[i].circle_color.alpha = 1.0

                # pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            except StopIteration:
                break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def create_pipeline_from_cfg(self, str_pipe, name_pipe="pipeline"):
        Gst.init(None)

        self.str_pipe = str_pipe
        self.inputs = str_pipe['source']['properties']['urls']
        self.n_sources = len(self.inputs)
        self.perf_data = PERF_DATA(self.n_sources)
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
