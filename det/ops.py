import time

from threading import Lock

start_time=time.time()

fps_mutex = Lock()

class GETFPS:
    def __init__(self,stream_id):
        global start_time
        self.start_time=start_time
        self.is_first=True
        self.frame_count=0
        self.stream_id=stream_id

    def update_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        else:
            global fps_mutex
            with fps_mutex:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        end_time = time.time()
        with fps_mutex:
            stream_fps = float(self.frame_count/(end_time - self.start_time))
            self.frame_count = 0
        self.start_time = end_time
        return round(stream_fps, 2)

    def print_data(self):
        print('frame_count=',self.frame_count)
        print('start_time=',self.start_time)

class PERF_DATA:
    def __init__(self, num_streams=1):
        self.perf_dict = {}
        self.all_stream_fps = {}
        for i in range(num_streams):
            self.all_stream_fps["stream{0}".format(i)]=GETFPS(i)

    def perf_print_callback(self):
        self.perf_dict = {stream_index:stream.get_fps() for (stream_index, stream) in self.all_stream_fps.items()}
        print ("\n**PERF: ", self.perf_dict, "\n")
        return True
    
    def update_fps(self, stream_index):
        self.all_stream_fps[stream_index].update_fps()



import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
def create_source_bin(sm, index, uri):
    def cb_newpad(decodebin, pad, source_id):
        caps = pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()

        if gstname.find("video") != -1:
            pad_name = "sink_%u" % source_id
            sinkpad = sm.get_static_pad(pad_name)
            if not sinkpad:
                sinkpad = sm.get_request_pad(pad_name)
            
            if sinkpad and pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                print("Decodebin linked to pipeline successfully")
            else:
                print("Failed to link decodebin to pipeline")
                exit()

    def decodebin_child_added(child_proxy, Object, name, user_data):
        # Connect callback to internal decodebin signal
        if name.find("decodebin") != -1:
            Object.connect("child-added", decodebin_child_added, user_data)

        if "src" in name:
            source_element = child_proxy.get_by_name("src")
            if source_element.find_property('drop-on-latency') is not None:
                Object.set_property("drop-on-latency", True)

    bin_name = f'src_{index}'
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", bin_name)
    if not uri_decode_bin:
        print(" Unable to create uri decode bin \n")
        exit()
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added", cb_newpad, index)
    uri_decode_bin.connect("child-added",decodebin_child_added, None)
    return uri_decode_bin


def bus_call(bus, message, loop, pipeline):
    
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
        if message.src == pipeline:
            old_state, new_state, pending_state = message.parse_state_changed()
            print("Pipeline state changed from {0:s} to {1:s}".format(
                Gst.Element.state_get_name(old_state),
                Gst.Element.state_get_name(new_state)))
    return True

def log_state_return(state_return, plugin=None):
    if state_return == Gst.StateChangeReturn.SUCCESS:
        print("STATE CHANGE SUCCESS\n")

    elif state_return == Gst.StateChangeReturn.FAILURE:
        print("STATE CHANGE FAILURE\n")
    
    elif state_return == Gst.StateChangeReturn.ASYNC:
        if plugin:
            state_return = plugin.get_state(Gst.CLOCK_TIME_NONE)

    elif state_return == Gst.StateChangeReturn.NO_PREROLL:
        print("STATE CHANGE NO PREROLL\n")
    return state_return