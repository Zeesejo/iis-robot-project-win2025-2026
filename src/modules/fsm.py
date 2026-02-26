"""
Finite State Machine (FSM) for Robot Task Execution
Module 7: Logic & State Management

States:
- IDLE:     Waiting to start
- SEARCH:   Looking for target object using vision
- NAVIGATE: Moving robot base toward target
- APPROACH: Fine positioning near target
- GRASP:    Manipulating arm to grasp object
- LIFT:     Lifting the grasped object
- SUCCESS:  Task completed successfully
- FAILURE:  Error occurred, recovery needed

FIX: Timeouts are now measured in simulation steps (passed in from the
     Sense-Think-Act loop) instead of wall-clock time.  The old
     time.time()-based timer fired during PyBullet startup before the
     first sim step was ever executed, causing instant FAILURE.
"""

from enum import Enum, auto
import time


class RobotState(Enum):
    """Enumeration of all possible robot states"""
    IDLE     = auto()
    SEARCH   = auto()
    NAVIGATE = auto()
    APPROACH = auto()
    GRASP    = auto()
    LIFT     = auto()
    SUCCESS  = auto()
    FAILURE  = auto()


class RobotFSM:
    """
    Finite State Machine for coordinating robot behaviour.

    Timeouts are step-based (240 steps = 1 simulated second) so they are
    immune to the wall-clock time consumed by PyBullet initialisation.
    Call  fsm.tick()  once per Sense-Think-Act iteration.
    """

    # Sim runs at 240 Hz; define timeouts in simulated seconds.
    SIM_HZ = 240
    TIMEOUT_STEPS = {
        RobotState.SEARCH:   600 * SIM_HZ,   # 600 s  (robot may need to orbit far)
        RobotState.NAVIGATE: 300 * SIM_HZ,   # 300 s
        RobotState.APPROACH: 120 * SIM_HZ,   # 120 s
        RobotState.GRASP:    20  * SIM_HZ,   #  20 s  (multi-phase arm)
        RobotState.LIFT:     10  * SIM_HZ,   #  10 s
    }

    def __init__(self):
        self.state          = RobotState.IDLE
        self.previous_state = None

        # Step-based timer (replaces wall-clock time.time())
        self._step_counter      = 0   # global sim step counter (set by tick())
        self._state_start_step  = 0   # step at which current state was entered

        # Keep a wall-clock start as well (for get_time_in_state backward-compat)
        self._state_start_wall  = time.time()

        # Task data
        self.target_found        = False
        self.target_position     = None
        self.distance_to_target  = float('inf')
        self.grasp_attempted     = False
        self.object_grasped      = False

        # Failure recovery
        self.failure_count    = 0
        self.max_failures     = 5
        self.failure_reason   = None
        self.recovery_started = False

        # Debug
        self.state_history            = []
        self.terminal_message_printed = False

    # ── step counter ────────────────────────────────────────────────────────

    def tick(self):
        """Increment the internal step counter.  Call once per STA loop."""
        self._step_counter += 1

    def get_steps_in_state(self):
        """Steps elapsed since entering the current state."""
        return self._step_counter - self._state_start_step

    def get_time_in_state(self):
        """Simulated seconds in current state (steps / SIM_HZ)."""
        return self.get_steps_in_state() / self.SIM_HZ

    # ── transitions ─────────────────────────────────────────────────────────

    def transition_to(self, new_state):
        if self.state == new_state:
            return
        self.previous_state      = self.state
        self.state               = new_state
        self._state_start_step   = self._step_counter
        self._state_start_wall   = time.time()
        self.terminal_message_printed = False
        self.state_history.append((new_state, self._step_counter))

        # Reset failure count on forward progress
        order = {RobotState.SEARCH: 0, RobotState.NAVIGATE: 1,
                 RobotState.APPROACH: 2, RobotState.GRASP: 3,
                 RobotState.LIFT: 4, RobotState.SUCCESS: 5}
        if (new_state in order and self.previous_state in order
                and order[new_state] > order.get(self.previous_state, -1)):
            self.failure_count = 0

        print(f"[FSM] State transition: {self.previous_state.name} → {new_state.name}")

    # ── timeout ─────────────────────────────────────────────────────────────

    def check_timeout(self):
        max_steps = self.TIMEOUT_STEPS.get(self.state)
        if max_steps is None:
            return False
        steps_in = self.get_steps_in_state()
        if steps_in > max_steps:
            secs = steps_in / self.SIM_HZ
            max_secs = max_steps / self.SIM_HZ
            print(f"[FSM] Timeout in state {self.state.name} after "
                  f"{secs:.1f}s (max={max_secs:.0f}s)")
            self.failure_reason = f'{self.state.name.lower()}_timeout'
            self.transition_to(RobotState.FAILURE)
            return True
        return False

    # ── main update ─────────────────────────────────────────────────────────

    def update(self, sensor_data):
        """
        Main FSM update.  Called every STA iteration AFTER tick().

        sensor_data keys:
            target_visible    bool
            target_position   [x,y,z] or None
            distance_to_target float
            collision_detected bool
            gripper_contact   bool
            object_grasped    bool
            estimated_pose    [x,y,theta]
        """
        control = {'navigate': False, 'approach': False,
                   'grasp': False, 'lift': False}

        if self.state not in (RobotState.SUCCESS, RobotState.FAILURE):
            self.check_timeout()

        # ── IDLE ────────────────────────────────────────────────────────────
        if self.state == RobotState.IDLE:
            print("[FSM] Starting task: Search for target object")
            self.transition_to(RobotState.SEARCH)

        # ── SEARCH ──────────────────────────────────────────────────────────
        elif self.state == RobotState.SEARCH:
            if sensor_data.get('target_visible', False):
                self.target_found    = True
                self.target_position = sensor_data['target_position']
                print(f"[FSM] Target found at {self.target_position}")
                self.transition_to(RobotState.NAVIGATE)

        # ── NAVIGATE ────────────────────────────────────────────────────────
        elif self.state == RobotState.NAVIGATE:
            self.distance_to_target = sensor_data.get('distance_to_target',
                                                       float('inf'))
            # [F23] Lowered from 2.0 m to 1.2 m: the standoff is 0.65 m so
            # 1.2 m means the robot is ~0.55 m from the standoff point and
            # close enough to start fine visual approach.
            if self.distance_to_target < 1.2:
                print(f"[FSM] Reached navigation waypoint "
                      f"(dist={self.distance_to_target:.2f}m)")
                self.transition_to(RobotState.APPROACH)
            elif sensor_data.get('collision_detected', False):
                print("[FSM] Collision during navigation!")
                self.failure_reason = 'collision'
                self.transition_to(RobotState.FAILURE)
            else:
                control['navigate'] = True

        # ── APPROACH ────────────────────────────────────────────────────────
        elif self.state == RobotState.APPROACH:
            self.distance_to_target = sensor_data.get('distance_to_target',
                                                       float('inf'))
            if self.distance_to_target < 0.55:
                print(f"[FSM] Target within grasp range "
                      f"(dist={self.distance_to_target:.2f}m)")
                self.transition_to(RobotState.GRASP)
            elif sensor_data.get('collision_detected', False):
                print("[FSM] Collision during approach!")
                self.failure_reason = 'collision'
                self.transition_to(RobotState.FAILURE)
            else:
                control['approach'] = True

        # ── GRASP ───────────────────────────────────────────────────────────
        elif self.state == RobotState.GRASP:
            if sensor_data.get('gripper_contact', False):
                self.object_grasped = True
                print("[FSM] Object grasped successfully!")
                self.transition_to(RobotState.LIFT)
            elif self.get_time_in_state() > 20.0:
                print("[FSM] Grasp attempt failed (timeout)")
                self.failure_reason = 'grasp_failed'
                self.transition_to(RobotState.FAILURE)
            else:
                control['grasp'] = True

        # ── LIFT ────────────────────────────────────────────────────────────
        elif self.state == RobotState.LIFT:
            if self.get_time_in_state() > 2.0:
                print("[FSM] Object lifted successfully!")
                self.transition_to(RobotState.SUCCESS)
            else:
                control['lift'] = True

        # ── SUCCESS ─────────────────────────────────────────────────────────
        elif self.state == RobotState.SUCCESS:
            if not self.terminal_message_printed:
                print("[FSM] Task completed successfully! 🎉")
                self.terminal_message_printed = True

        # ── FAILURE / RECOVERY ──────────────────────────────────────────────
        elif self.state == RobotState.FAILURE:
            if not self.terminal_message_printed:
                print(f"[FSM] Task failed. Reason: {self.failure_reason}. "
                      f"Attempting recovery...")
                self.terminal_message_printed = True

            if self.failure_count >= self.max_failures:
                if not getattr(self, '_max_fail_printed', False):
                    print(f"[FSM] Max failures ({self.max_failures}) reached. "
                          f"Resetting and retrying...")
                    self._max_fail_printed = True
                self.failure_count    = 0
                self._max_fail_printed = False
                self.recovery_started  = False
                self.failure_reason    = None
                self.target_found      = False
                self.transition_to(RobotState.SEARCH)
                return control

            if not self.recovery_started:
                self.failure_count   += 1
                self.recovery_started = True
                print(f"[FSM] Recovery {self.failure_count}/{self.max_failures}: "
                      f"Starting recovery for {self.failure_reason}...")

            if self.failure_reason == 'collision':
                control['navigate'] = True
                if self.get_time_in_state() > 2.0:
                    print("[FSM] Restarting search from new position")
                    self.recovery_started = False
                    self.failure_reason   = None
                    self.transition_to(RobotState.SEARCH)

            elif self.failure_reason == 'grasp_failed':
                if self.get_time_in_state() > 1.5:
                    print("[FSM] Retrying approach phase")
                    self.recovery_started  = False
                    self.failure_reason    = None
                    self.grasp_attempted   = False
                    self.transition_to(RobotState.APPROACH)

            elif self.failure_reason and 'timeout' in self.failure_reason:
                if ('navigate' in self.failure_reason
                        or 'search' in self.failure_reason):
                    if self.get_time_in_state() > 0.5:
                        print("[FSM] Restarting search due to timeout")
                        self.recovery_started = False
                        self.failure_reason   = None
                        self.target_found     = False
                        self.transition_to(RobotState.SEARCH)
                else:
                    if self.get_time_in_state() > 2.0:
                        print("[FSM] Restarting from search after timeout")
                        self.recovery_started = False
                        self.failure_reason   = None
                        self.transition_to(RobotState.SEARCH)

            else:
                if self.get_time_in_state() > 0.5:
                    print(f"[FSM] Unknown failure, restarting search...")
                    self.recovery_started = False
                    self.failure_reason   = None
                    self.transition_to(RobotState.SEARCH)

        return control

    # ── utilities ───────────────────────────────────────────────────────────

    def reset(self):
        """Reset FSM to IDLE."""
        self.state               = RobotState.IDLE
        self.previous_state      = None
        self._state_start_step   = self._step_counter
        self._state_start_wall   = time.time()
        self.target_found        = False
        self.target_position     = None
        self.distance_to_target  = float('inf')
        self.grasp_attempted     = False
        self.object_grasped      = False
        self.failure_count       = 0
        self.failure_reason      = None
        self.recovery_started    = False
        self.state_history       = []
        print("[FSM] Reset to IDLE state")

    def get_state_name(self):
        return self.state.name

    def is_task_complete(self):
        return self.state in (RobotState.SUCCESS, RobotState.FAILURE)
