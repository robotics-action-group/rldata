import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorStorage
from .modality import ModalityConfig

"""
How a data point looks like
    {
        "video": {
            "camera_eye": [...],
            "camera_left_wrist": [...]
        },
        "state": {
            "robot_state": [...],
            ...
        },
        "action": {
            "robot_action": [...],
            ...
        },
        "language": {
            "task_description": [...]
        }
    }
"""

class RLData():
    """
    Online buffer for LeRobot dataset.
    This class is used to store the online data collected during the simulation.
    It inherits from torch Dataset 
    """
    def __init__(self, data_config, buffer_size=1000):
        # data format
        self._modality_configs: dict[str, ModalityConfig] = data_config.modality_config()
        self.data = None  # Placeholder for data storage
        self.initialize()
        print("Data manager setup with modalities:")
        self.printModalityConfig()
        self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(buffer_size), batch_size=1)

    # def __add__(self

    def initialize(self):
        """
        Initialize the online buffer. Creates empty tensors for each modality.
        This method is called to set up the buffer before recording data.
        """
        self._key_to_modality = {}
        for modality_label, modality in self._modality_configs.items():
            for key in modality.modality_keys:
                # create dict maping key to modality_label
                self._key_to_modality[key] = modality_label
    
    def addZerosBuffer(self, batch_size=1):
        """
        Create a zero-initialized buffer for each modality.
        This method is called to set up the buffer before receiving data.
        """
        new_dict = {}
        current_length = len(self.replay_buffer)
        for modality_label, modality in self._modality_configs.items():
            if modality_label == "language":
                continue
            new_dict[modality_label] = {}
            for idx, key in enumerate(modality.modality_keys):
                if modality_label in ["video", "state", "action"]:
                    shape = [batch_size] + modality.shapes_list[idx]
                    new_dict[modality_label][key] = torch.zeros(shape, dtype=torch.float32)
        # self.printStructureDict(new_dict)
        self.replay_buffer.extend(new_dict)
        new_length = len(self.replay_buffer)

        zero_index_start = current_length
        zero_index_end = new_length
        return(zero_index_start, zero_index_end)
    
    def printStructureDict(self, dict_obj):
        """
        Print the structure of a dict, where values are also dicts of {str: torch.Tensor}.
        Args:
            dict_obj (dict): The dictionary to print.
        """
        print("Structure of the dictionary:")
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value.shape} ({sub_value.dtype})")
            else:
                print(f"{key}: {value.shape} ({value.dtype})")
    
    def printReplayBufferStructure(self):
        """
        Print the structure of the replay buffer.
        This method is called to print the structure of the replay buffer.
        """
        print("Structure of the replay buffer:")
        for modality_label, modality in self._modality_configs.items():
            if modality_label == "language":
                continue
            print(f"{modality_label}:")
            for key in modality.modality_keys:
                if key in self.replay_buffer[modality_label]:
                    value = self.replay_buffer[modality_label][key]
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: Not found in replay buffer")
    
    def printModalityConfig(self):
        """
        Print the modality configuration.
        This method is called to print the modality configuration.
        """
        print("Modality configuration:")
        for modality_label, modality in self._modality_configs.items():
            print(f"{modality_label}:")
            print(f"  Modality keys: {modality.modality_keys}")
            print(f"  Shapes list: {modality.shapes_list}")
            print(f"  Delta indices: {modality.delta_indices}")

    def process_obs_from_isaac(self, obs_dict): 
        """
        Process the observation dictionary from Isaac Gym to the format required by the model.

        Args:
            obs_dict (dict): The observation dictionary from Isaac Gym.
        Returns:
            dict: Processed observation dictionary compatible with the model.
        """
        obs = obs_dict["observations"]
        new_dict = {}
        for key, value in obs.items():
            # print(f"Processing key: {key}, value device: {value.device}, value type: {type(value)}")
            modality_label = self._key_to_modality.get(key, None)
            if new_dict.get(modality_label) is None:
                new_dict[modality_label] = {key: value}
            else:
                new_dict[modality_label][key] = value
        return new_dict

    def add_to_replay_buffer(self, processed_obs):
        """
        Add the processed observation dictionary to the replay buffer.

        Args:
            obs_dict (dict): The observation dictionary from Isaac Gym.
        """
        # processed_obs = self.process_obs_from_isaac(obs_dict)
        # print("Adding to replay buffer with keys: ", processed_obs.keys())
        # Check if the processed_obs is in the expected format
        if not isinstance(processed_obs, dict):
            raise ValueError("Processed observations must be a dictionary.")
        for modality_label, modality in self._modality_configs.items():
            if modality_label == "language":
                continue
            if modality_label not in processed_obs:
                raise ValueError(f"Processed observations missing modality: {modality_label}")
            for key in modality.modality_keys:
                if key not in processed_obs[modality_label]:
                    raise ValueError(f"Processed observations missing key: {key} in modality: {modality_label}")
                processed_obs[modality_label][key] = processed_obs[modality_label][key].cpu()
        # print("Processed observation keys: ", processed_obs.keys())
        self.replay_buffer.extend(processed_obs)

    def toGR00T(self):
        """
        Convert the online buffer data to GR00T format.
        
        """
        new_dict = {}
        for modality_label, modality in self._modality_configs.items():
            new_dict[modality_label] = {}
            for key in modality.modality_keys:
                if modality_label in ["video", "state", "action"]:
                    new_dict[modality_label][key] = self.replay_buffer[modality_label][key]
                elif modality_label == "language":
                    new_dict[modality_label][key] = "do the fucking task"
        return new_dict

    def emptyReplayBuffer(self):
        """
        Empty the replay buffer.
        
        This method is called to clear the replay buffer, typically at the end of an episode or when starting a new one.
        """
        self.replay_buffer.clear()
        # self.initialize()

    def makeVideo(self, key, video_dir, fps=30):
        """
        Create a video from the replay buffer data for a specific key.

        Args:
            key (str): The key for which to create the video.
            video_dir (str): The directory where the video will be saved.
            fps (int): Frames per second for the output video.
        """
        import cv2
        import os

        if key not in self._key_to_modality:
            raise ValueError(f"Key {key} not found in the modality configuration.")

        video_data = self.replay_buffer["video"][key]
        if video_data is None or len(video_data) == 0:
            raise ValueError(f"No data found for key {key} in the replay buffer.")

        # Create a directory for the video if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)

        # Define the video writer
        height, width = video_data[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
        # fourcc = cv2.VideoWriter_fourcc(*"VP80")  # VP8
        video_path = os.path.join(video_dir, f"{key}_video.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for frame in video_data:
            frame_np = frame.cpu().numpy()
            if frame_np.dtype != 'uint8':
                # If float, assume [0,1] and scale; otherwise, cast directly
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).clip(0, 255).astype('uint8')
                else:
                    frame_np = frame_np.astype('uint8')
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
    
    def makeGif(self, key, gif_dir, fps=30):
        """
        Create a GIF from the replay buffer data for a specific key.

        Args:
            key (str): The key for which to create the GIF.
            gif_dir (str): The directory where the GIF will be saved.
            fps (int): Frames per second for the output GIF.
        """
        import imageio
        import os

        if key not in self._key_to_modality:
            raise ValueError(f"Key {key} not found in the modality configuration.")

        video_data = self.replay_buffer["video"][key]
        if video_data is None or len(video_data) == 0:
            raise ValueError(f"No data found for key {key} in the replay buffer.")

        # Create a directory for the GIF if it doesn't exist
        os.makedirs(gif_dir, exist_ok=True)

        gif_path = os.path.join(gif_dir, f"{key}_video.gif")
        
        # Convert frames to numpy arrays and save as GIF
        frames = []
        for frame in video_data:
            frame_np = frame.cpu().numpy()
            if frame_np.dtype != 'uint8':
                # if frame_np.max() <= 1.0: # TODO: why dark image, triggers this if condition
                frame_np = (frame_np * 255).clip(0, 255).astype('uint8')
                # else:
                #     frame_np = frame_np.astype('uint8')
            frames.append(frame_np)
        imageio.mimsave(gif_path, frames, fps=fps)


