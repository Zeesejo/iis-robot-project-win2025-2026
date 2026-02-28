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

All timeouts are step-based (immune to wall-clock startup delay).
Collision is detected via LIDAR readings passed in sensor_data.
"""

from enum import Enum, auto
import time
import logging

logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Enumeration of all possible robot states"""
    IDLE     = auto()
    SEARCH   = auto()
    NAVIGATE = auto()
    APPROACH = auto()
    GRASP    = auto()
    LIFT     = auto()
    PLACE    = auto()
    SUCCESS  = auto()
    FAILURE  = auto()


class RobotFSM:
    """
    Finite State Machine for coordinating robot behaviour.
    Timeouts are step-based (240 steps = 1 simulated second).
    Call fsm.tick() once per Sense-Think-Act iteration.
    """

    SIM_HZ = 240
    TIMEOUT_STEPS = {
        RobotState.SEARCH:   600 * 240,
        RobotState.NAVIGATE: 300 * 240,
        RobotState.APPROACH: 180 * 240,
        RobotState.GRASP:    30  * 240,
        RobotState.LIFT:     15  * 240,
        RobotState.PLACE:    15  * 240,
    }

    # LIDAR front-sector threshold for collision detection (meters)
    COLLISION_LIDAR_THRESH = 0.12

    def __init__(self):
        self.state          = RobotState.IDLE
        self.previous_state = None

        self._step_counter      = 0
        self._state_start_step  = 0
        self._state_start_wall  = time.time()

        self.target_found        = False
        self.target_position     = None
        self.distance_to_target  = float('inf')
        self.grasp_attempted     = False
        self.object_grasped      = False

        self.failure_count    = 0
        self.max_failures     = 5
        self.failure_reason   = None
        self.recovery_started = False

        self.state_history            = []
        self.terminal_message_printed = False

    # ── step counter ────────────────────────────────────────────────

    def tick(self):
        """Increment internal step counter. Call once per STA loop."""
        self._step_counter += 1

    def get_steps_in_state(self):
        return self._step_counter - self._state_start_step

    def get_time_in_state(self):
        return self.get_steps_in_state() / self.SIM_HZ

    # ── transitions ─────────────────────────────────────────────────

    def transition_to(self, new_state):
        if self.state == new_state:
            return
        self.previous_state      = self.state
        self.state               = new_state
        self._state_start_step   = self._step_counter
        self._state_start_wall   = time.time()
        self.terminal_message_printed = False
        self.state_history.append((new_state, self._step_counter))

        order = {RobotState.SEARCH: 0, RobotState.NAVIGATE: 1,
                 RobotState.APPROACH: 2, RobotState.GRASP: 3,
                 RobotState.LIFT: 4, RobotState.SUCCESS: 5}
        if (new_state in order and self.previous_state in order
                and order[new_state] > order.get(self.previous_state, -1)):
            self.failure_count = 0

        msg = f"[FSM] {self.previous_state.name} -> {new_state.name}  (step={self._step_counter})"
        print(msg)
        logger.info(msg)

    # ── timeout ─────────────────────────────────────────────────────

    def check_timeout(self):
        max_steps = self.TIMEOUT_STEPS.get(self.state)
        if max_steps is None:
            return False
        steps_in = self.get_steps_in_state()
        if steps_in > max_steps:
            secs     = steps_in / self.SIM_HZ
            max_secs = max_steps / self.SIM_HZ
            msg = (f"[FSM] TIMEOUT in {self.state.name} after {secs:.1f}s "
                   f"(max={max_secs:.0f}s)")
            print(msg)
            logger.warning(msg)
            self.failure_reason = f'{self.state.name.lower()}_timeout'
            self.transition_to(RobotState.FAILURE)
            return True
        return False

    # ── lidar collision helper ───────────────────────────────────────

    @staticmethod
    def _lidar_collision(lidar, thresh=None):
        """Return True if any front-sector lidar ray is closer than thresh."""
        if thresh is None:
            thresh = RobotFSM.COLLISION_LIDAR_THRESH
        if lidar is None:
            return False
        try:
            n = len(lidar)
            if n == 0:
                return False
            # front sector: indices ±3 around 0
            front = [lidar[i % n] for i in range(-3, 4)]
            return min(front) < thresh
        except (TypeError, ValueError):
            return False

    # ── main update ─────────────────────────────────────────────────

    def update(self, sensor_data):
        """
        Main FSM update. Called every STA iteration AFTER tick().

        sensor_data keys:
            target_visible    bool
            target_position   [x,y,z] or None
            distance_to_target float
            lidar             list of floats (raw lidar array) or None
            gripper_contact   bool
            object_grasped    bool
            estimated_pose    [x,y,theta]
        """
        control = {'navigate': False, 'approach': False,
                   'grasp': False, 'lift': False}

        lidar = sensor_data.get('lidar')

        if self.state not in (RobotState.SUCCESS, RobotState.FAILURE):
            self.check_timeout()

        # ── IDLE ────────────────────────────────────────────────────
        if self.state == RobotState.IDLE:
            print("[FSM] IDLE -> starting task")
            self.transition_to(RobotState.SEARCH)

        # ── SEARCH ──────────────────────────────────────────────────
        elif self.state == RobotState.SEARCH:
            if sensor_data.get('target_visible', False):
                self.target_found    = True
                self.target_position = sensor_data['target_position']
                print(f"[FSM] Target found at {self.target_position}")
                self.transition_to(RobotState.NAVIGATE)
            elif sensor_data.get('table_near', False):
                print("[FSM] Close to table -> NAVIGATE")
                self.transition_to(RobotState.NAVIGATE)

        # ── NAVIGATE ────────────────────────────────────────────────
        elif self.state == RobotState.NAVIGATE:
            self.distance_to_target = sensor_data.get('distance_to_target', float('inf'))
            collision = self._lidar_collision(lidar)
            if collision:
                print("[FSM] Collision detected during NAVIGATE")
                self.failure_reason = 'collision'
                self.transition_to(RobotState.FAILURE)
            elif self.distance_to_target < 1.5:
                print(f"[FSM] Reached nav waypoint (dist={self.distance_to_target:.2f}m)")
                self.transition_to(RobotState.APPROACH)
            else:
                control['navigate'] = True

        # ── APPROACH ────────────────────────────────────────────────
        elif self.state == RobotState.APPROACH:
            self.distance_to_target = sensor_data.get('distance_to_target', float('inf'))
            collision = self._lidar_collision(lidar, thresh=0.07)
            if collision:
                print("[FSM] Collision detected during APPROACH")
                self.failure_reason = 'collision'
                self.transition_to(RobotState.FAILURE)
            elif self.distance_to_target < 0.50:
                print(f"[FSM] In grasp range (dist={self.distance_to_target:.2f}m) -> GRASP")
                self.transition_to(RobotState.GRASP)
            else:
                control['approach'] = True

        # ── GRASP ───────────────────────────────────────────────────
        elif self.state == RobotState.GRASP:
            if sensor_data.get('gripper_contact', False):
                self.object_grasped = True
                print("[FSM] Gripper contact confirmed -> LIFT")
                self.transition_to(RobotState.LIFT)
            elif self.get_steps_in_state() > self.TIMEOUT_STEPS[RobotState.GRASP]:
                print("[FSM] Grasp timed out")
                self.failure_reason = 'grasp_failed'
                self.transition_to(RobotState.FAILURE)
            else:
                control['grasp'] = True

        # ── LIFT ────────────────────────────────────────────────────
        elif self.state == RobotState.LIFT:
            if self.get_steps_in_state() > 3 * self.SIM_HZ:   # 3 s
                print("[FSM] Object lifted successfully -> PLACE")
                self.transition_to(RobotState.PLACE)
            else:
                control['lift'] = True

        # ── PLACE ───────────────────────────────────────────────────
        elif self.state == RobotState.PLACE:
            if self.get_steps_in_state() > 3 * self.SIM_HZ:   # 3 s
                print("[FSM] Object placed successfully -> SUCCESS")
                self.transition_to(RobotState.SUCCESS)
            else:
                control['place'] = True

        # ── SUCCESS ─────────────────────────────────────────────────
        elif self.state == RobotState.SUCCESS:
            if not self.terminal_message_printed:
                msg = ("[FSM] *** TASK COMPLETED SUCCESSFULLY *** "
                       f"total_steps={self._step_counter} "
                       f"({self._step_counter/self.SIM_HZ:.1f}s sim time)")
                print(msg)
                logger.info(msg)
                self.terminal_message_printed = True

        # ── FAILURE / RECOVERY ──────────────────────────────────────
        elif self.state == RobotState.FAILURE:
            if not self.terminal_message_printed:
                msg = (f"[FSM] FAILURE reason={self.failure_reason} "
                       f"count={self.failure_count+1}/{self.max_failures}")
                print(msg)
                logger.warning(msg)
                self.terminal_message_printed = True

            if self.failure_count >= self.max_failures:
                if not getattr(self, '_max_fail_printed', False):
                    print(f"[FSM] Max failures reached — resetting to SEARCH")
                    self._max_fail_printed = True
                self.failure_count     = 0
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
                      f"{self.failure_reason}")

            if self.failure_reason == 'collision':
                control['navigate'] = True
                if self.get_steps_in_state() > 2 * self.SIM_HZ:
                    self.recovery_started = False
                    self.failure_reason   = None
                    self.transition_to(RobotState.SEARCH)

            elif self.failure_reason == 'grasp_failed':
                if self.get_steps_in_state() > int(1.5 * self.SIM_HZ):
                    self.recovery_started  = False
                    self.failure_reason    = None
                    self.grasp_attempted   = False
                    self.transition_to(RobotState.APPROACH)

            elif self.failure_reason and 'timeout' in self.failure_reason:
                if 'navigate' in self.failure_reason or 'search' in self.failure_reason:
                    if self.get_steps_in_state() > int(0.5 * self.SIM_HZ):
                        self.recovery_started = False
                        self.failure_reason   = None
                        self.target_found     = False
                        self.transition_to(RobotState.SEARCH)
                else:
                    if self.get_steps_in_state() > 2 * self.SIM_HZ:
                        self.recovery_started = False
                        self.failure_reason   = None
                        self.transition_to(RobotState.SEARCH)
            else:
                if self.get_steps_in_state() > int(0.5 * self.SIM_HZ):
                    self.recovery_started = False
                    self.failure_reason   = None
                    self.transition_to(RobotState.SEARCH)

        return control

    # ── utilities ───────────────────────────────────────────────────

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
        self.terminal_message_printed = False
        print("[FSM] Reset to IDLE")

    def get_state_name(self):
        return self.state.name

    def is_task_complete(self):
        return self.state in (RobotState.SUCCESS, RobotState.FAILURE)

    def is_success(self):
        return self.state == RobotState.SUCCESS
