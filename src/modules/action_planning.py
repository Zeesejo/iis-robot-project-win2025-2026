"""
M7 - Action Planning: Finite State Machine
Manages the mission: INIT -> SEARCH -> NAVIGATE -> ALIGN -> GRASP -> DONE
Includes failure recovery transitions.
"""

import numpy as np


class State:
    INIT = 'INIT'
    SEARCH = 'SEARCH'
    NAVIGATE = 'NAVIGATE'
    ALIGN = 'ALIGN'
    GRASP = 'GRASP'
    LIFT = 'LIFT'
    DONE = 'DONE'
    RECOVER = 'RECOVER'
    FAILED = 'FAILED'


class MissionFSM:
    """
    Finite State Machine for the Navigate-to-Grasp mission.
    Transitions:
        INIT -> SEARCH (always)
        SEARCH -> NAVIGATE (target found)
        SEARCH -> SEARCH (target not found, rotate)
        NAVIGATE -> ALIGN (near table)
        NAVIGATE -> RECOVER (obstacle collision)
        ALIGN -> GRASP (arm aligned)
        GRASP -> LIFT (object grasped)
        LIFT -> DONE (object lifted)
        RECOVER -> NAVIGATE (after recovery)
        Any -> FAILED (timeout or repeated failure)
    """

    def __init__(self):
        self.state = State.INIT
        self.prev_state = None
        self._step_count = 0
        self._state_steps = 0
        self._max_search_steps = 2400   # 10 seconds at 240Hz
        self._max_nav_steps = 12000     # 50 seconds
        self._max_grasp_steps = 4800    # 20 seconds
        self._max_lift_steps = 2400
        self._recovery_steps = 480      # 2 seconds
        self._recovery_count = 0
        self._max_recoveries = 5

        # Mission data
        self.target_position = None       # Estimated (x, y, z) of target
        self.table_position = None        # Known from map
        self.current_waypoint_idx = 0
        self.waypoints = []
        self.nav_complete = False
        self.grasp_attempts = 0
        self._max_grasp_attempts = 3

    def transition(self, event, data=None):
        """
        Drive state machine forward based on event and data.
        event: string event name
        data: optional dict with extra info
        Returns: (new_state, action_command)
        """
        self._step_count += 1
        self._state_steps += 1

        prev = self.state

        if self.state == State.INIT:
            self.state = State.SEARCH
            self._state_steps = 0
            return self.state, {'cmd': 'rotate_search'}

        elif self.state == State.SEARCH:
            if event == 'target_found':
                self.target_position = data.get('target_pos')
                self.state = State.NAVIGATE
                self._state_steps = 0
                return self.state, {'cmd': 'plan_path',
                                     'goal': self.target_position}
            elif event == 'table_found':
                self.table_position = data.get('table_pos')
                self.state = State.NAVIGATE
                self._state_steps = 0
                return self.state, {'cmd': 'plan_path',
                                     'goal': self.table_position}
            elif self._state_steps > self._max_search_steps:
                self.state = State.FAILED
                return self.state, {'cmd': 'stop', 'reason': 'search_timeout'}
            return self.state, {'cmd': 'rotate_search'}

        elif self.state == State.NAVIGATE:
            if event == 'waypoint_reached':
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    self.state = State.ALIGN
                    self._state_steps = 0
                    return self.state, {'cmd': 'align_arm'}
                return self.state, {'cmd': 'next_waypoint',
                                     'waypoint': self.waypoints[
                                         self.current_waypoint_idx]}
            elif event == 'collision':
                if self._recovery_count < self._max_recoveries:
                    self._recovery_count += 1
                    self.state = State.RECOVER
                    self._state_steps = 0
                    return self.state, {'cmd': 'backup'}
                else:
                    self.state = State.FAILED
                    return self.state, {'cmd': 'stop',
                                         'reason': 'too_many_collisions'}
            elif event == 'at_table':
                self.state = State.ALIGN
                self._state_steps = 0
                return self.state, {'cmd': 'align_arm'}
            elif self._state_steps > self._max_nav_steps:
                self.state = State.FAILED
                return self.state, {'cmd': 'stop', 'reason': 'nav_timeout'}
            return self.state, {'cmd': 'continue_nav'}

        elif self.state == State.ALIGN:
            if event == 'arm_aligned':
                self.state = State.GRASP
                self._state_steps = 0
                return self.state, {'cmd': 'execute_grasp'}
            return self.state, {'cmd': 'align_arm'}

        elif self.state == State.GRASP:
            if event == 'grasp_success':
                self.state = State.LIFT
                self._state_steps = 0
                return self.state, {'cmd': 'lift_object'}
            elif event == 'grasp_fail':
                self.grasp_attempts += 1
                if self.grasp_attempts >= self._max_grasp_attempts:
                    self.state = State.FAILED
                    return self.state, {'cmd': 'stop',
                                         'reason': 'grasp_failed'}
                self.state = State.ALIGN
                return self.state, {'cmd': 'realign'}
            elif self._state_steps > self._max_grasp_steps:
                self.state = State.FAILED
                return self.state, {'cmd': 'stop', 'reason': 'grasp_timeout'}
            return self.state, {'cmd': 'execute_grasp'}

        elif self.state == State.LIFT:
            if event == 'lift_success':
                self.state = State.DONE
                return self.state, {'cmd': 'hold'}
            elif self._state_steps > self._max_lift_steps:
                self.state = State.DONE  # Accept partial success
                return self.state, {'cmd': 'hold'}
            return self.state, {'cmd': 'lift_object'}

        elif self.state == State.RECOVER:
            if self._state_steps > self._recovery_steps:
                self.state = State.NAVIGATE
                self._state_steps = 0
                self.current_waypoint_idx = max(0, self.current_waypoint_idx - 1)
                return self.state, {'cmd': 'continue_nav'}
            return self.state, {'cmd': 'backup'}

        elif self.state in (State.DONE, State.FAILED):
            return self.state, {'cmd': 'idle'}

        return self.state, {'cmd': 'idle'}

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    def is_terminal(self):
        return self.state in (State.DONE, State.FAILED)

    def __repr__(self):
        return f"<FSM state={self.state} step={self._step_count}>"
