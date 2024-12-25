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
# Fix bug deepstream 7.0
os.system("rm -rf ~/.cache/gstreamer-1.0/registry.x86_64.bin")
class MoveRecorder:
    """Ghi nhận và xử lý các nước đi cờ"""
    def __init__(self):
        self.previous_board = None
        self.move_history = []

    def record_move(self, current_board):
        if self.previous_board is None:
            self.previous_board = current_board  # First time, update directly
            return None
        print("Previous board", self.previous_board)
        print("Current board", current_board)
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
                    if old_piece != 'O' and new_piece == 'O':
                        start_position = (i, j)
                        moved_piece = old_piece
                    elif old_piece == 'O' and new_piece != 'O':
                        end_position = (i, j)
                    elif old_piece != 'O' and new_piece != 'O' and old_piece[0] != new_piece[0]:
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
class BasePipeline:
    def __init__(self):
        self.count = 0
        self.mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        self.previous_board = None  
        self.current_arrow = None   
        self.move_record = MoveRecorder()  
        self.previous_arrow = None
        self.config_point_path = "/home/project/chess_board/cfg/point.txt"
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
            

            surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            # # Chuyển đổi NvBufSurface thành numpy array
            # frame_image = np.array(surface, copy=True, order='C')
            # print(frame_image.shape)
            objects = []
            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    objects.append(obj_meta)
                    # print(objects)
                    obj_meta.text_params.display_text = f"{obj_meta.class_id}:{obj_meta.confidence*100:.0f}"
                    # # exit()
                    xc_yc_list = self.read_grid_points(self.config_point_path, "grid_chess_points", "grid_points")
                except StopIteration:
                    break

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # update frame rate through this probe
            stream_index = "stream{0}".format(frame_meta.source_id)
            self.perf_data.update_fps(stream_index)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            board = []
            class_id = None
            for rect in xc_yc_list:
                is_inside_bbox = False
                x_center, y_center = rect
                for obj in objects:
                    rect_params = obj.rect_params
                    left = rect_params.left
                    top = rect_params.top
                    width = rect_params.width
                    height = rect_params.height
                    # class_id = obj.class_id
                    # print(f"Class ID: {class_id}")
                    
                    if left < x_center < left + width and top < y_center < top + height:
                        class_id = obj.class_id
                        is_inside_bbox = True
                        break
                board.append(class_id if is_inside_bbox else "O")
            board = np.array(board).reshape(10, 9)
            self.current_arrow = self.move_record.record_move(board)
            print("Current arrow", self.current_arrow)
            if self.current_arrow is not None:
                if self.current_arrow != self.previous_arrow:
                    start_pos, end_pos, text = self.current_arrow
                    start_pos = (xc_yc_list[start_pos[0] * 9 + start_pos[1]])
                    end_pos = (xc_yc_list[end_pos[0] * 9 + end_pos[1]])
                    self.previous_arrow = self.current_arrow 
                else:
                    start_pos, end_pos, text = self.previous_arrow
                    start_pos = (xc_yc_list[start_pos[0] * 9 + start_pos[1]])
                    end_pos = (xc_yc_list[end_pos[0] * 9 + end_pos[1]])
                print("start_pos", start_pos)
                print("end_pos", end_pos)
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_arrows = 1
                display_meta.arrow_params[0].x1 = start_pos[0]
                display_meta.arrow_params[0].y1 = start_pos[1]
                display_meta.arrow_params[0].x2 = end_pos[0]
                display_meta.arrow_params[0].y2 = end_pos[1]
                display_meta.arrow_params[0].arrow_width = 2
                # display_meta.arrow_params[0].arrow_color = pyds.NvOSD_ColorParams(1.0, 0.0, 0.0, 1.0) 
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            else:
                if self.previous_arrow is not None:
                    start_pos, end_pos, text = self.previous_arrow
                    start_pos = (xc_yc_list[start_pos[0] * 9 + start_pos[1]])
                    end_pos = (xc_yc_list[end_pos[0] * 9 + end_pos[1]])
                    
                    print("No new arrow, using previous arrow:")
                    print("start_pos", start_pos)
                    print("end_pos", end_pos)
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    display_meta.num_arrows = 1
                    display_meta.arrow_params[0].x1 = start_pos[0]
                    display_meta.arrow_params[0].y1 = start_pos[1]
                    display_meta.arrow_params[0].x2 = end_pos[0]
                    display_meta.arrow_params[0].y2 = end_pos[1]
                    display_meta.arrow_params[0].arrow_width = 2
                    # display_meta.arrow_params[0].arrow_color = pyds.NvOSD_ColorParams(1.0, 0.0, 0.0, 1.0)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            # if self.previous_board is None:
            #     self.previous_board = board

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

        tiler_sink_pad = self.pipeline.get_by_name("tiler").get_static_pad("sink")
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
