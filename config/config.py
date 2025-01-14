import os
import yaml
import cv2
from typing import Any, Dict

class Config:
    def __init__(self, config_file: str = None):
        """Initialize configuration
        
        Args:
            config_file: Path to yaml config file. If None, load default config
        """
        # Load default config
        default_config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
        with open(default_config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Override with custom config if provided
        if config_file is not None:
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._config = self._merge_configs(self._config, custom_config)

        # Process special configs
        self._process_configs()

    def _merge_configs(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge custom config into default config"""
        merged = default.copy()
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _process_configs(self):
        """Process special config values that need conversion"""
        # Convert aruco dict type string to cv2 constant if it exists
        if 'aruco' in self._config.get('board', {}):
            try:
                # Try to import aruco from cv2.aruco (OpenCV 3.x)
                import cv2.aruco as aruco
                aruco_dict_type = self._config['board']['aruco']['dict_type']
                self._config['board']['aruco']['dict_type'] = getattr(aruco, aruco_dict_type)
            except ImportError:
                try:
                    # Try to import aruco from cv2 (OpenCV 4.x)
                    aruco_dict_type = self._config['board']['aruco']['dict_type']
                    self._config['board']['aruco']['dict_type'] = getattr(cv2.aruco, aruco_dict_type)
                except AttributeError:
                    print("Warning: cv2.aruco not available. Using default dictionary type.")
                    self._config['board']['aruco']['dict_type'] = 250  # Default value for DICT_4X4_250

        # Convert video codec string to fourcc code if it exists
        if 'video_codec' in self._config.get('visualization', {}):
            codec = self._config['visualization']['video_codec']
            self._config['visualization']['video_codec'] = cv2.VideoWriter_fourcc(*codec)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing config values as attributes"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    @property
    def device(self) -> str:
        """Get device type based on use_gpu setting"""
        return 'cuda' if self._config['model']['use_gpu'] else 'cpu'

# Create global config instance
cfg = Config()