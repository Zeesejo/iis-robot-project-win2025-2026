"""
Finite State Machine (FSM) for Robot Task Execution
Module 7: Logic & State Management

States:
- IDLE: Waiting to start
- SEARCH: Looking for target object using vision
- NAVIGATE: Moving robot base toward target
- APPROACH: Fine positioning near target
- GRASP: Manipulating arm to grasp object
- LIFT: Lifting the grasped object
- SUCCESS: Task completed successfully
- FAILURE: Error occurred, recovery needed
"""

from enum import Enum, auto
import time


class RobotState(Enum):
    """Enumeration of all possible robot states"""
    IDLE = auto()
    SEARCH = auto()
    NAVIGATE = auto()
    APPROACH = auto()
    GRASP = auto()
    LIFT = auto()
    SUCCESS = auto()
    FAILURE = auto()


class RobotFSM:
    """
    Finite State Machine for coordinating robot behavior.
    Manages state transitions based on sensor feedback and task progress.
    """
    
    def __init__(self):
        self.state = RobotState.IDLE
        self.previous_state = None
        self.state_start_time = time.time()
        self.max_state_time = 30.0  # Max time in any state before timeout
        
        # Task data
        self.target_found = False
        self.target_position = None
        self.distance_to_target = float('inf')
        self.grasp_attempted = False
        self.object_grasped = False
        
        # Transition history for debugging
        self.state_history = []
        
        # Flag to prevent repeated printing in terminal states
        self.terminal_message_printed = False
        
    def transition_to(self, new_state):
        """
        Transition to a new state with logging.
        
        Args:
            new_state: RobotState to transition to
        """
        if self.state != new_state:
            self.previous_state = self.state
            self.state = new_state
            self.state_start_time = time.time()
            self.state_history.append((new_state, time.time()))
            self.terminal_message_printed = False  # Reset flag on state change
            print(f"[FSM] State transition: {self.previous_state.name} → {new_state.name}")
    
    def get_time_in_state(self):
        """Returns time spent in current state (seconds)"""
        return time.time() - self.state_start_time
    
    def check_timeout(self):
        """Check if current state has exceeded max time"""
        if self.get_time_in_state() > self.max_state_time:
            print(f"[FSM] Timeout in state {self.state.name}")
            self.transition_to(RobotState.FAILURE)
            return True
        return False
    
    def update(self, sensor_data):
        """
        Main FSM update loop. Called every simulation step.
        
        Args:
            sensor_data: Dict containing:
                - 'target_visible': bool
                - 'target_position': [x, y, z] or None
                - 'distance_to_target': float
                - 'collision_detected': bool
                - 'gripper_contact': bool
                
        Returns:
            Dict with control commands:
                - 'navigate': bool
                - 'approach': bool
                - 'grasp': bool
                - 'lift': bool
        """
        # Default control output
        control = {
            'navigate': False,
            'approach': False,
            'grasp': False,
            'lift': False
        }
        
        # Check for timeout in any state
        if self.state not in [RobotState.SUCCESS, RobotState.FAILURE]:
            self.check_timeout()
        
        # State machine logic
        if self.state == RobotState.IDLE:
            # Start the task
            print("[FSM] Starting task: Search for target object")
            self.transition_to(RobotState.SEARCH)
            
        elif self.state == RobotState.SEARCH:
            if sensor_data.get('target_visible', False):
                self.target_found = True
                self.target_position = sensor_data['target_position']
                print(f"[FSM] Target found at {self.target_position}")
                self.transition_to(RobotState.NAVIGATE)
            # Continue searching (vision processing happens elsewhere)
            
        elif self.state == RobotState.NAVIGATE:
            self.distance_to_target = sensor_data.get('distance_to_target', float('inf'))
            
            if self.distance_to_target < 1.5:  # Close enough to approach
                print(f"[FSM] Reached navigation waypoint (dist={self.distance_to_target:.2f}m)")
                self.transition_to(RobotState.APPROACH)
            elif sensor_data.get('collision_detected', False):
                print("[FSM] Collision during navigation!")
                self.transition_to(RobotState.FAILURE)
            else:
                control['navigate'] = True  # Keep moving toward target
                
        elif self.state == RobotState.APPROACH:
            self.distance_to_target = sensor_data.get('distance_to_target', float('inf'))
            
            if self.distance_to_target < 0.5:  # Within grasp range
                print("[FSM] Target within grasp range")
                self.transition_to(RobotState.GRASP)
            elif sensor_data.get('collision_detected', False):
                print("[FSM] Collision during approach!")
                self.transition_to(RobotState.FAILURE)
            else:
                control['approach'] = True  # Fine positioning
                
        elif self.state == RobotState.GRASP:
            if sensor_data.get('gripper_contact', False):
                self.object_grasped = True
                print("[FSM] Object grasped successfully!")
                self.transition_to(RobotState.LIFT)
            elif self.get_time_in_state() > 5.0:  # Grasp timeout
                print("[FSM] Grasp attempt failed (timeout)")
                self.transition_to(RobotState.FAILURE)
            else:
                control['grasp'] = True  # Attempt grasp
                
        elif self.state == RobotState.LIFT:
            if self.get_time_in_state() > 2.0:  # Lift for 2 seconds
                print("[FSM] Object lifted successfully!")
                self.transition_to(RobotState.SUCCESS)
            else:
                control['lift'] = True  # Hold lifted position
                
        elif self.state == RobotState.SUCCESS:
            if not self.terminal_message_printed:
                print("[FSM] Task completed successfully! 🎉")
                self.terminal_message_printed = True
            # Stay in success state
            
        elif self.state == RobotState.FAILURE:
            if not self.terminal_message_printed:
                print("[FSM] Task failed. Recovery needed.")
                self.terminal_message_printed = True
            # Recovery logic would go here
            
        return control
    
    def reset(self):
        """Reset FSM to initial state"""
        self.state = RobotState.IDLE
        self.previous_state = None
        self.state_start_time = time.time()
        self.target_found = False
        self.target_position = None
        self.distance_to_target = float('inf')
        self.grasp_attempted = False
        self.object_grasped = False
        self.state_history = []
        print("[FSM] Reset to IDLE state")
    
    def get_state_name(self):
        """Returns current state name as string"""
        return self.state.name
    
    def is_task_complete(self):
        """Check if task is finished (success or failure)"""
        return self.state in [RobotState.SUCCESS, RobotState.FAILURE]
