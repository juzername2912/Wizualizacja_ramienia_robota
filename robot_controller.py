import pybullet as p
import pybullet_data
import time
import numpy as np
import pygame
from robot_controller import TeachAndPlayController

# ---------------------------------------------------------------------------
# Limity stawów w stopniach — spójne z wartościami w pliku simple_arm.urdf.
# Zmieniaj tutaj zamiast w URDF, jeśli chcesz ograniczyć zakres programowo.
#   Joint 0 (baza, oś Z):      ±150°  →  np. 30°–330° na skali 0–360°
#   Joint 1–3 (ramię, oś Y):   ±90°
# ---------------------------------------------------------------------------
JOINT_LIMITS_DEG = [
    (-150.0, 150.0),   # joint1 – obrót bazy
    ( -90.0,  90.0),   # joint2 – pierwsze ramię
    ( -90.0,  90.0),   # joint3 – drugie ramię
    ( -90.0,  90.0),   # joint_wrist – przegub
]
JOINT_LIMITS_RAD = [(np.deg2rad(lo), np.deg2rad(hi)) for lo, hi in JOINT_LIMITS_DEG]


class RobotSimulation:
    EE_INDEX      = 4   # indeks ogniwa end-effektora w URDF
    NUM_JOINTS    = 4   # liczba sterowanych stawów
    GRAB_THRESHOLD = 0.25  # maksymalny dystans [m] do chwytania kostki

    def __init__(self):
        self._setup_physics()
        self._load_models()
        self._setup_ui()
        self._setup_audio()

        self.controller = TeachAndPlayController(self.robot, self.EE_INDEX)

        self.current_angles   = [0.0] * self.NUM_JOINTS
        self.prev_angles      = [0.0] * self.NUM_JOINTS
        self.prev_slider_vals = [0.3, 0.0, 0.4]
        self.gripper_active   = False
        self.constraint_id    = None
        self.space_pressed    = False
        self.tick_counter     = 0  # do regulacji częstotliwości nagrywania

    # ------------------------------------------------------------------
    # Inicjalizacja
    # ------------------------------------------------------------------

    def _setup_physics(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def _load_models(self):
        self.plane = p.loadURDF("plane.urdf")
        self.cube  = p.loadURDF("cube.urdf",       [0.3, 0.0, 0.2], useFixedBase=False)
        self.robot = p.loadURDF("simple_arm.urdf", [0,   0,   0  ], useFixedBase=True)

    def _setup_ui(self):
        self.sliders = {
            'x': p.addUserDebugParameter("Ramię Cel X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Ramię Cel Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Ramię Cel Z",  0.1, 1.0, 0.4),
        }
        self.cube_sliders = {
            'x': p.addUserDebugParameter("Start Kostki X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Start Kostki Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Start Kostki Z",  0.05, 1.0, 0.2),
        }
        self.buttons = {
            'set_cube': p.addUserDebugParameter("USTAW KOSTKE NA STARCIE",  1, 0, 0),
            'record':   p.addUserDebugParameter(" NAGRYWAJ (START/STOP)", 1, 0, 0),
            'play':     p.addUserDebugParameter(" ODTWORZ SEKWENCJE",     1, 0, 0),
            'clear':    p.addUserDebugParameter(" WYCZUSC PAMIEC",        1, 0, 0),
        }
        self.btn_states = {k: 0 for k in self.buttons}

    def _setup_audio(self):
        pygame.mixer.init()
        try:
            self.motor_sound = pygame.mixer.Sound("motor.wav")
            self.motor_channel = pygame.mixer.Channel(0)
        except FileNotFoundError:
            print(" UWAGA: Brak pliku 'motor.wav'. Dźwięk nie będzie odtwarzany.")
            self.motor_sound = None

    # ------------------------------------------------------------------
    # Obsługa wejść użytkownika i Audio
    # ------------------------------------------------------------------

    def _check_button(self, btn_name: str) -> bool:
        """Zwraca True dokładnie raz po każdym kliknięciu przycisku."""
        current_clicks = p.readUserDebugParameter(self.buttons[btn_name])
        if current_clicks > self.btn_states[btn_name]:
            self.btn_states[btn_name] = current_clicks
            return True
        return False

    def _clamp_angles(self):
        """Przycina current_angles do limitów zdefiniowanych w JOINT_LIMITS_RAD."""
        for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            self.current_angles[i] = float(np.clip(self.current_angles[i], lo, hi))

    def handle_ik_sliders(self):
        """Sterowanie przez suwaki IK — aktualizuje kąty tylko przy zmianie."""
        vals = [p.readUserDebugParameter(self.sliders[k]) for k in ('x', 'y', 'z')]
        if any(abs(v - pv) > 0.001 for v, pv in zip(vals, self.prev_slider_vals)):
            ik_angles = p.calculateInverseKinematics(self.robot, self.EE_INDEX, vals)
            self.current_angles   = list(ik_angles)[:self.NUM_JOINTS]
            self.prev_slider_vals = vals
            self._clamp_angles()

    def handle_keyboard(self):
        """Sterowanie klawiaturą."""
        keys  = p.getKeyboardEvents()
        delta = 0.01  # krok kąta na klatkę [rad]

        key_map = {
            p.B3G_LEFT_ARROW:  (0, -delta),
            p.B3G_RIGHT_ARROW: (0, +delta),
            p.B3G_UP_ARROW:    (1, -delta),
            p.B3G_DOWN_ARROW:  (1, +delta),
            ord('z'):          (2, +delta),
            ord('x'):          (2, -delta),
            ord('c'):          (3, +delta),
            ord('v'):          (3, -delta),
        }

        for key, (joint_idx, d) in key_map.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                self.current_angles[joint_idx] += d

        self._clamp_angles()

        # Toggle chwytaka na pierwsze wciśnięcie spacji
        space_down = ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN
        if space_down and not self.space_pressed:
            self._toggle_gripper()
        self.space_pressed = space_down

    def _update_audio(self):
        if not self.motor_sound:
            return

        # Check if any joint is actually moving in the simulation (not just input change)
        is_moving = False
        for i in range(self.NUM_JOINTS):
            joint_state = p.getJointState(self.robot, i)
            joint_velocity = joint_state[1]  # Linear or angular velocity
            if abs(joint_velocity) > 0.01:  # Threshold for detectible motion
                is_moving = True
                break

        if is_moving:
            if not self.motor_channel.get_busy():
                self.motor_channel.play(self.motor_sound, loops=-1)
        else:
            if self.motor_channel.get_busy():
                self.motor_channel.stop()

    # ------------------------------------------------------------------
    # Logika chwytaka
    # ------------------------------------------------------------------

    def _toggle_gripper(self):
        self.gripper_active = not self.gripper_active

        if self.gripper_active and self.constraint_id is None:
            self._try_grab()
        elif not self.gripper_active and self.constraint_id is not None:
            self._release()

    def _try_grab(self):
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        ee_pos   = p.getLinkState(self.robot, self.EE_INDEX)[0]

        if np.linalg.norm(np.array(cube_pos) - np.array(ee_pos)) >= self.GRAB_THRESHOLD:
            self.gripper_active = False
            return

        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 0)

        ee_pos,   ee_orn   = p.getLinkState(self.robot, self.EE_INDEX)[0:2]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        local_pos, local_orn   = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        self.constraint_id = p.createConstraint(
            self.robot, self.EE_INDEX, self.cube, -1,
            p.JOINT_FIXED, [0, 0, 0], local_pos, [0, 0, 0], local_orn
        )

    def _release(self):
        p.removeConstraint(self.constraint_id)
        self.constraint_id = None
        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 1)

    # ------------------------------------------------------------------
    # Narzędzia
    # ------------------------------------------------------------------

    def reset_cube_position(self):
        cx, cy, cz = [p.readUserDebugParameter(self.cube_sliders[k]) for k in ('x', 'y', 'z')]
        p.resetBasePositionAndOrientation(self.cube, [cx, cy, cz], [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

    def sync_state_after_play(self):
        if not self.controller.waypoints:
            return
        last = self.controller.waypoints[-1]
        self.current_angles = list(last["angles"])
        self.gripper_active  = last["gripper"]
        if not self.gripper_active and self.constraint_id is not None:
            self._release()
        
        # Zapobiega "czknięciu" audio po synchronizacji
        self.prev_angles = list(self.current_angles)

    # ------------------------------------------------------------------
    # Główna pętla
    # ------------------------------------------------------------------

    def run(self):
        print("====== SYMULACJA ROBOTA — TRYB NAGRYWANIA ======")
        print("Klawiatura:  baza |  ramię1 | Z/X ramię2 | C/V przegub | SPACJA chwytak")

        while True:
            self.tick_counter += 1

            # 1. Odczyt wejść
            self.handle_ik_sliders()
            self.handle_keyboard()

            # Aktualizacja dźwięku podczas sterowania ręcznego
            if not self.controller.is_playing:
                self._update_audio()

            # 2. Przyciski UI
            if self._check_button('set_cube'):
                self.reset_cube_position()

            if self._check_button('record'):
                self.controller.toggle_recording()

            if self._check_button('play'):
                self.reset_cube_position()
                self.controller.play_sequence(self.cube)
                self.sync_state_after_play()

            if self._check_button('clear'):
                self.controller.clear_sequence()

            # 3. Ciągłe nagrywanie w tle
            if self.controller.is_recording and self.tick_counter % 10 == 0:
                self.controller.record_frame(self.current_angles, self.gripper_active, self.cube)

            # 4. Fizyka
            if not self.controller.is_playing:
                for i, angle in enumerate(self.current_angles):
                    p.setJointMotorControl2(
                        self.robot, i,
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=200,
                        maxVelocity=1.5
                    )

            p.stepSimulation()
            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    app = RobotSimulation()
    app.run()
