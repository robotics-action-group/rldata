import torch.distributed as dist
from rldata.base import *

class RLDataDistributed(RLData):
    """
    A distributed version of RLData that uses PyTorch's dist modules
    for communication between processes.
    """
    def __init__(self, name="agent", data_config=None, buffer_size=1000):
        """
        Initialize the online buffer for distributed training.
        This method is called to set up the buffer for distributed training provided by pytorch.
        
        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
        """
        super().__init__(data_config=data_config, buffer_size=buffer_size)
        world_size = 2
        if name == "agent":
            self.rank = 0
        elif name == "simulator":
            self.rank = 1

        self.AGENT_RANK = 0
        self.SIMULATOR_RANK = 1
        if self.rank == self.SIMULATOR_RANK:
            dist.init_process_group("gloo", rank=self.rank, world_size=world_size, init_method='tcp://10.5.0.2:8000')
        else:   
            dist.init_process_group("gloo", rank=self.rank, world_size=world_size, init_method='tcp://10.5.0.2:8000')
    
    def shutdown(self):
        """
        Shutdown the distributed process group.
        
        This method is called to clean up the distributed environment.
        """
        dist.destroy_process_group()
        print("[INFO] Distributed process group destroyed.")
    
    def sendBufferToAgent(self):
        """
        Send the data to the agent.        
        """
        # send the observation tensor to the agent
        op_list = []
        idx = 0
        length_of_buffer = len(self.replay_buffer)
        req = dist.isend(tensor=torch.tensor(length_of_buffer), dst=self.AGENT_RANK)
        req.wait()
        dist.barrier()  # Ensure all processes are synchronized before sending data
        print("length sent successfully")
        for modality_label, modality in self._modality_configs.items():
            if modality_label == "language":
                print("[DEBUG] Skipping language modality for sending.")
                continue
            for key in modality.modality_keys:
                print(key)
                # if modality_label not in self.replay_buffer:
                #     raise ValueError(f"Modality {modality_label} not found in the replay buffer.")
                # if key not in self.replay_buffer[modality_label]:
                #     raise ValueError(f"Key {key} not found in modality {modality_label} of the replay buffer.")
                print(f"[DEBUG] Sending {modality_label}:{key} shape={self.replay_buffer[modality_label][key].shape} dtype={self.replay_buffer[modality_label][key].dtype} device={self.replay_buffer[modality_label][key].device}")
                send_op = dist.P2POp(dist.isend, tensor=self.replay_buffer[modality_label][key], peer=self.AGENT_RANK)
                op_list.append(send_op)
        print("[INFO] Setting up send operations for modalities")
        reqs = dist.batch_isend_irecv(op_list)
        print("[INFO] Sending buffer to agent with length:", length_of_buffer)
        for req in reqs:
            req.wait()
        print("[INFO] Buffer sent to agent successfully.")

    def recvBufferFromSim(self):
        """
        Receive the data from the simulator.
        
        This method is called to receive the data from the simulator process.
        It blocks until the data is received.
        """
        # first receive the length of the buffer
        length_of_buffer_tensor = torch.tensor(0)
        req = dist.irecv(tensor=length_of_buffer_tensor, src=self.SIMULATOR_RANK)
        req.wait()
        dist.barrier()
        print("[INFO] Length of buffer received from simulator:", length_of_buffer_tensor.item())
        length_of_buffer = length_of_buffer_tensor.item()

        zero_start, zero_end = self.addZerosBuffer(length_of_buffer)
        print("[INFO] Created zero buffer with length:", length_of_buffer)
        self.printReplayBufferStructure()
        op_list = []
        for modality_label, modality in self._modality_configs.items():
            if modality_label == "language":
                print("[DEBUG] Skipping language modality for sending.")
                continue
            for key in modality.modality_keys:
                print(key)
                # if modality_label not in self.replay_buffer:
                #     raise ValueError(f"Modality {modality_label} not found in the replay buffer.")
                # if key not in self.replay_buffer[modality_label]:
                #     raise ValueError(f"Key {key} not found in modality {modality_label} of the replay buffer.")
                print(f"[DEBUG] Receiving {modality_label}:{key} shape={self.replay_buffer[modality_label][key].shape} dtype={self.replay_buffer[modality_label][key].dtype} device={self.replay_buffer[modality_label][key].device}")
                recv_op = dist.P2POp(dist.irecv, tensor=self.replay_buffer[modality_label][key][zero_start:zero_end], peer=self.SIMULATOR_RANK)
                op_list.append(recv_op)
        print("[INFO] Receiving buffer from simulator with length:", length_of_buffer)
        reqs = dist.batch_isend_irecv(op_list)
        for req in reqs:
            req.wait()

    def sendObject(self, obj, dst=1):
        """
        Send an object to a process in the distributed environment.
        
        Args:
            obj: The object to be sent.
            dst (int): The rank of the destination process.
        """
        dist.send_object_list([obj], dst=dst)
    
    def recvObject(self, src=0):
        """
        Receive an object from a process in the distributed environment.
        
        Args:
            src (int): The rank of the source process to receive the object from.
        
        Returns:
            The received object.
        """
        dump = [None]
        dist.recv_object_list(dump, src=src)
        return dump[0]

    def sendData(self, data: torch.Tensor, dst=1):
        """
        Send data to a process in the distributed environment.
        Args:
            data (torch.Tensor): The data to be sent.
        """
        dist.send(tensor=data, dst=dst)
    
    def recvData(self, src=0, trgt_tensor: torch.Tensor = torch.zeros(19, device="cpu")):
        """
        Receive data from a process in the distributed environment.
        
        Args:
            src (int): The rank of the source process to receive data from.
        
        Returns:
            torch.Tensor: The received data.
        """
        dist.recv(tensor=trgt_tensor, src=src)
        return trgt_tensor

    def sendDictOneByOne(self, data: dict, dst=1):
        """
        Send a dictionary of data to a process in the distributed environment.
        
        Args:
            data (dict): The dictionary containing data to be sent.
            dst (int): The rank of the destination process.
        """
        video = 0
        state = 1
        action = 2
        language = 3

        dst = self.AGENT_RANK
        for modality_label, modality in data.items():
            for key, value in modality.items():
                m = eval(modality_label)
                k = self._modality_configs[modality_label].modality_keys.index(key)
                # create a tensor [m,k]
                address_tensor = torch.tensor([m, k], device="cuda")
                # send the address tensor to the destination process
                self.sendData(address_tensor, dst=dst)
                
                # send the value tensor to the destination process
                self.sendData(value, dst=dst)

    def recvDictOneByOne(self, src=0):
        """
        Receive a dictionary of data from a process in the distributed environment.
        
        Args:
            src (int): The rank of the source process to receive data from.
        
        Returns:
            dict: The received dictionary containing data.
        """
        data = {}
        src = self.SIMULATOR_RANK
        while True:
            # receive the address tensor
            address_tensor = self.recvData(src=src, trgt_tensor=torch.zeros(2, device="cuda"))
            if address_tensor is None:
                break
            m, k = address_tensor.tolist()
            modality_label = list(self._modality_configs.keys())[int(m)]
            key = self._modality_configs[modality_label].modality_keys[int(k)]
            
            # receive the value tensor
            value_tensor = self.recvData(src=src, trgt_tensor=torch.zeros(19, device="cuda"))
            if modality_label not in data:
                data[modality_label] = {}
            data[modality_label][key] = value_tensor
        return data
                
    def sendToPolicy(self, obs: torch.Tensor):
        """
        Send data to the policy for processing.
        
        Args:
            data (torch.Tensor): The data to be sent to the policy.
        """
        # send data to the policy
        self.sendData(obs, dst=self.AGENT_RANK)
        self.action = self.recvData(trgt_tensor=self.action, src=self.AGENT_RANK)
        return self.action

    def waitForObs(self):
        """
        Wait for observations from the simulator.
        
        This method is called to wait for observations from the simulator process.
        It blocks until the data is received.
        
        Returns:
            torch.Tensor: The received observations.
        """
        # wait for data from the simulator
        self.observation = self.recvData(src=self.SIMULATOR_RANK, trgt_tensor=self.observation)
        return self.observation

    def sendAction(self, action: torch.Tensor):
        """
        Send action to the simulator process.
        
        Args:
            action (torch.Tensor): The action to be sent to the simulator.
        """
        # send action to the simulator
        self.sendData(data=action, dst=self.SIMULATOR_RANK)
    
    def recvAction(self):
        """
        Receive action from the agent process.
        
        This method is called to receive action from the agent process.
        It blocks until the data is received.
        
        Returns:
            torch.Tensor: The received action.
        """
        # wait for action from the agent
        self.action = self.recvData(src=self.AGENT_RANK, trgt_tensor=self.action)
        return self.action
