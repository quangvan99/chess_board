import os
import math
import copy
import sys  # Add this import

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

from ops import PERF_DATA
import cv2
# Fix bug deepstream 7.0
os.system("rm -rf ~/.cache/gstreamer-1.0/registry.x86_64.bin")

class BasePipeline:
    def __init__(self):
        self.count = 0
        self.mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    
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
            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    obj_meta.text_params.display_text = f"{obj_meta.class_id}:{obj_meta.confidence*100:.0f}"
                    # exit()
                    # Lấy bounding box từ object meta
                    rect_params = obj_meta.rect_params
                    left = rect_params.left
                    top = rect_params.top
                    width = rect_params.width
                    height = rect_params.height
                    
                    

                    # In ra tọa độ bounding box của object
                    # print(f"Stream: {frame_meta.pad_index}, Object ID: {obj_meta.object_id}, "
                    #     f"Bounding Box: Left={left}, Top={top}, Width={width}, Height={height}")
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
