import configparser

# Initialize the ConfigParser
config = configparser.ConfigParser()
# Read the configuration file
config.read("/home/project/weights/dslaunchpad_tracker_config.txt")
# Get all sections in the configuration file
config.sections()

# Extract tracker configuration values
for key in config['tracker']:
    if key == 'tracker-width':
        tracker_width = config.getint('tracker', key)
    if key == 'tracker-height':
        tracker_height = config.getint('tracker', key)
    if key == 'gpu-id':
        tracker_gpu_id = config.getint('tracker', key)
    if key == 'll-lib-file':
        tracker_ll_lib_file = config.get('tracker', key)
    if key == 'll-config-file':
        tracker_ll_config_file = config.get('tracker', key)
        
sm = {"sm": {
        "plugin": "nvstreammux",
        "properties": {
            "width": 1280,
            "height": 1280,
            "batch-size": 1,
            # "batched-push-timeout": 4000000,
            "nvbuf-memory-type": None
                    }
    }
}   

# Detection and tracking configuration
detection_tracking = {
    "nvvideoconvert1": {
        "plugin": "nvvideoconvert",
        "properties": {
            "nvbuf-memory-type": None
        }
    },
    
    "capsfilter": {
        "plugin": "capsfilter",
        "properties": {
            "caps": None
        }
    },

    "nvinfer": {
        "plugin": "nvinfer",
        "properties": {
            "config-file-path": "/home/project/weights/config_infer_primary_yoloV8.txt",
            "batch-size": None
        }
    },

    "nvtracker": {
        "plugin": "nvtracker",
        "properties": {
            "tracker-width": tracker_width,
            "tracker-height": tracker_height,
            "gpu-id": tracker_gpu_id,
            "ll-lib-file": tracker_ll_lib_file,
            "ll-config-file": tracker_ll_config_file,
        }
    },
    "nvvideoconvert2": {
        "plugin": "nvvideoconvert",
        "properties": {
            "nvbuf-memory-type": None
        }
    },
    "tiler": {
        "plugin": 'nvmultistreamtiler',
        "properties": {
            "rows": 2,
            "columns": 2,
            "width": 640,
            "height": 640,
        }
    },
}

# dewarper = {
#         "nvdewarper": {
#             "plugin": "nvdewarper",
#             "properties": {
#                 "config-file": "/home/project/weights/config_dewarper.txt",
#             },
#             "next": "",
#         },
#     }

# Visualization configuration
viz = {
    "nvvideoconvert3": {
        "plugin": "nvvideoconvert",
        "properties": {
            "nvbuf-memory-type": None
        }
    },
    "nvosdpadding": {
        "plugin": 'nvosdpadding',
        "properties": {
            'process-mode': 0,
            'enable-padding': True,  
            'padding-size': 500,      
            'padding-color': "255,255,255,255"  
        }
    },
    "nvdsosd": {
        "plugin": 'nvdsosd',
        "properties": {
            'process-mode': 0,
            'display-text': 1
        }
    },
}

# Displaysink configuration
diplaysink = {
    "nvvideoconvert4": {
        "plugin": "nvvideoconvert",
        "properties": {
            "nvbuf-memory-type": None
        }
    },
    "fpsdisplaysink": {
        "plugin": 'fpsdisplaysink',
        "properties": {
            "sync": False
        }
    },
}

# Fakesink configuration
fake_sink = {
    "fakesink": {
        "plugin": "fakesink",
        "properties": {
        }
    }
}

# File sink configuration
filesink = {
    "nvvideoconvert5": {
        "plugin": "nvvideoconvert",
        "properties": {
            "nvbuf-memory-type": None
        }
    },
    "encoder": {
        "plugin": 'nvv4l2h264enc',
        "properties": {
        }
    },
    "h264_parser": {
        "plugin": 'h264parse',
        "properties": {
        }
    },
    "muxer": {
        "plugin": 'qtmux',
        "properties": {
        }
    },
    "nvvideo_renderer": {
        "plugin": 'filesink',
        "properties": {
            "location": "/home/project/videos/out.mp4",
            "sync": 0
        }
    },
}
segment = {
    "nvvideoconvert1": {
        "plugin": "nvvideoconvert",
        "properties": {
            "nvbuf-memory-type": None
        }
    },
    
    "capsfilter": {
        "plugin": "capsfilter",
        "properties": {
            "caps": None
        }
    },

    "nvinfer": {
        "plugin": "nvinfer",
        "properties": {
            "config-file-path": "/home/project/weights/config_infer_primary_yoloV8_seg.txt",
            "batch-size": None
        }
    },
    "nvtracker": {
        "plugin": "nvtracker",
        "properties": {
            "tracker-width": tracker_width,
            "tracker-height": tracker_height,
            "gpu-id": tracker_gpu_id,
            "ll-lib-file": tracker_ll_lib_file,
            "ll-config-file": tracker_ll_config_file,
        }
    },

}