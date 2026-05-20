"""
Social Incentive Delay (SID) PsychoPy Prototype
------------------------------------------------
Edit these sections near the top of this file:
- timings
- trial numbers
- cue symbols
- comments
- image folders
- icon files
- catch trial frequency
- key mappings

This is a clean, code-based PsychoPy prototype modeled after a social
feedback / social media SID task. It uses images from:
- stimuli/own_photos/
- stimuli/other_photos/

Output CSV files are saved in:
- data/

The script is intentionally written for readability and easy editing.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from psychopy import core, event, gui, logging, visual
from psychopy.hardware import keyboard


# ============================================================================
# EDITABLE PARAMETERS
# ============================================================================

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
STIMULI_DIR = BASE_DIR / "stimuli"
OWN_PHOTO_DIR = STIMULI_DIR / "own_photos"
OTHER_PHOTO_DIR = STIMULI_DIR / "other_photos"
ASSET_DIR = STIMULI_DIR / "asset"
DATA_DIR = BASE_DIR / "data"

# Feedback phase 1 icons.
# These can be changed to different files if needed.
REWARD_HIT_ICON_FILE = ASSET_DIR / "RewardHit.png"
REWARD_MISS_ICON_FILE = ASSET_DIR / "RewardMiss.png"
PUNISHMENT_HIT_ICON_FILE = ASSET_DIR / "PunishmentHit.png"
PUNISHMENT_MISS_ICON_FILE = ASSET_DIR / "PunishmentMiss.png"

# --- Session settings ---
DEFAULT_DEBUG_MODE = False
PRACTICE_TRIALS = 8
BLOCK_TRIALS = 30
TOTAL_BLOCKS = 4
TOTAL_TRIALS = BLOCK_TRIALS * TOTAL_BLOCKS
DEBUG_TRIALS = 12
RUN_NUMBER = 1
FULLSCREEN = False
WINDOW_SIZE = (1200, 900)
BACKGROUND_COLOR = "black"

# --- Debug trial filters ---
# Set one of these to True when you want to debug a specific trial subset.
DEBUG_REWARD_ONLY = False
DEBUG_PUNISHMENT_ONLY = False
DEBUG_CATCH_ONLY = False

# --- Timings (seconds) ---
PHOTO_DURATION = 2.0
CUE_DURATION_MIN = 0.4
CUE_DURATION_MAX = 0.4
ANTICIPATION_MIN = 0.6
ANTICIPATION_MAX = 1.6
TARGET_WINDOW = 0.5
CATCH_TARGET_WINDOW = 1.5
POST_RESPONSE_DELAY = 1.0
FEEDBACK1_DURATION = 1.0
FEEDBACK2_MIN_DURATION = 1.5
FEEDBACK2_MAX_DURATION = 1.5
ITI_DURATION = 0.3

# --- Catch trial settings ---
ENABLE_CATCH_TRIALS = True
PRACTICE_CATCH_TRIALS = 1
MAIN_CATCH_TRIALS_PER_BLOCK = 2
MIN_NONCATCH_GAP = 4

# Catch trial prompt text
CATCH_PROMPT_TEXT = "Catch trial\nR = Reward    P = Punishment"

# --- Key mappings ---
MAIN_RESPONSE_KEY = "space"
CATCH_REWARD_KEY = "r"
CATCH_PUNISHMENT_KEY = "p"
QUIT_KEY = "escape"

# --- Adaptive RT settings ---
INITIAL_RT_THRESHOLD_MS = 500.0
MIN_RT_THRESHOLD_MS = 250.0
MAX_RT_THRESHOLD_MS = 700.0
ADAPT_STEP_MS = 20.0
TARGET_HIT_RATE = 0.60
ADAPT_WINDOW_SIZE = 10

# --- Cue symbols ---
CUE_SHAPE_REWARD = "circle"
CUE_SHAPE_PUNISHMENT = "square"
CUE_SYMBOL_COLOR = "white"
CUE_SYMBOL_SIZE = 0.08

# --- Display layout ---
PHOTO_SIZE = 0.70
TARGET_SIZE = 0.08
FIXATION_SIZE = 0.05

# --- Comments for feedback phase 2 ---
POSITIVE_COMMENTS = [
    "Nice post!",
    "Love this!",
    "So good",
    "This is great!",
]

NEGATIVE_COMMENTS = [
    "Not interesting",
    "I don't like this",
    "Nothing special",
    "Boring post",
]

# --- EEG trigger settings ---
EEG_ENABLED = False
EEG_PORT_TYPE = "serial"  # currently "serial" or "none"
EEG_SERIAL_PORT = "COM3"  # edit for your setup, e.g. "COM3" or "/dev/ttyUSB0"
EEG_SERIAL_BAUDRATE = 115200
EEG_PULSE_WIDTH_SEC = 0.005

# Trigger codes are easy to edit here.
TRIGGERS = {
    "trial_start": 1,
    "photo_own": 10,
    "photo_other": 11,
    "cue_reward": 20,
    "cue_punishment": 21,
    "anticipation": 30,
    "target": 40,
    "catch_prompt": 41,
    "response_space": 50,
    "response_reward_key": 51,
    "response_punishment_key": 52,
    "hit": 60,
    "miss": 61,
    "feedback1": 70,
    "feedback2": 71,
    "task_abort": 99,
}


ACTIVE_KEYBOARD: Optional[keyboard.Keyboard] = None


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class TrialSpec:
    trial_number: int
    photo_type: str
    cue_type: str
    image_path: Path
    catch_trial: bool = False
    trial_phase: str = "main"
    run_number: int = RUN_NUMBER


@dataclass
class TaskConfig:
    participant_id: str
    debug_mode: bool
    date_time: str
    total_trials: int
    output_csv: Path


@dataclass
class AdaptiveThreshold:
    current_ms: float = INITIAL_RT_THRESHOLD_MS
    recent_hits: List[int] = field(default_factory=list)

    def update(self, hit: Optional[bool]) -> None:
        if hit is None:
            return
        self.recent_hits.append(int(hit))
        if len(self.recent_hits) > ADAPT_WINDOW_SIZE:
            self.recent_hits.pop(0)
        observed_hit_rate = sum(self.recent_hits) / len(self.recent_hits)
        if observed_hit_rate > TARGET_HIT_RATE:
            self.current_ms = max(MIN_RT_THRESHOLD_MS, self.current_ms - ADAPT_STEP_MS)
        elif observed_hit_rate < TARGET_HIT_RATE:
            self.current_ms = min(MAX_RT_THRESHOLD_MS, self.current_ms + ADAPT_STEP_MS)


class TaskAbort(Exception):
    """Raised when the user safely quits the task."""


class DataLogger:
    def __init__(self, output_csv: Path):
        self.output_csv = output_csv
        self.rows: List[Dict[str, object]] = []
        self.fieldnames = [
            "participant_id",
            "date_time",
            "trial_number",
            "trial_phase",
            "run_number",
            "photo_type",
            "image_filename",
            "cue_type",
            "cue_symbol",
            "catch_trial",
            "catch_response",
            "catch_accuracy",
            "target_presented",
            "response_key",
            "reaction_time",
            "hit",
            "adaptive_threshold",
            "feedback_phase1_label",
            "feedback_phase1_top_text",
            "feedback_phase1_icon_type",
            "feedback_phase2_type",
            "comment_text",
        ]

    def append(self, row: Dict[str, object]) -> None:
        self.rows.append(row)
        self.save()

    def save(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


class EEGTrigger:
    def __init__(self) -> None:
        self.enabled = EEG_ENABLED
        self.port = None
        if not self.enabled or EEG_PORT_TYPE == "none":
            return
        if EEG_PORT_TYPE != "serial":
            logging.warning("Unsupported EEG port type '%s'. EEG disabled.", EEG_PORT_TYPE)
            self.enabled = False
            return
        try:
            import serial  # type: ignore

            self.port = serial.Serial(
                port=EEG_SERIAL_PORT,
                baudrate=EEG_SERIAL_BAUDRATE,
                timeout=0,
            )
            logging.info("EEG trigger connected via serial port %s", EEG_SERIAL_PORT)
        except Exception as exc:
            logging.warning("Could not initialize EEG trigger port: %s", exc)
            self.enabled = False

    def send(self, code_name: str) -> None:
        if not self.enabled or self.port is None:
            return
        code = TRIGGERS.get(code_name)
        if code is None:
            logging.warning("Unknown trigger code name: %s", code_name)
            return
        try:
            self.port.write(bytes([code]))
            core.wait(EEG_PULSE_WIDTH_SEC, hogCPUperiod=EEG_PULSE_WIDTH_SEC)
            self.port.write(bytes([0]))
        except Exception as exc:
            logging.warning("Failed to send EEG trigger '%s': %s", code_name, exc)

    def close(self) -> None:
        if self.port is not None:
            try:
                self.port.close()
            except Exception:
                pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def check_for_quit(data_logger: DataLogger, eeg: EEGTrigger) -> None:
    event_keys = event.getKeys([QUIT_KEY])
    keyboard_keys = []
    if ACTIVE_KEYBOARD is not None:
        keyboard_keys = ACTIVE_KEYBOARD.getKeys(
            keyList=[QUIT_KEY],
            waitRelease=False,
            clear=True,
        )
    if (QUIT_KEY in event_keys) or keyboard_keys:
        eeg.send("task_abort")
        data_logger.save()
        raise TaskAbort("Task interrupted by user.")


def show_for_duration(
    win: visual.Window,
    stimuli: Sequence[visual.BaseVisualStim],
    duration: float,
    data_logger: DataLogger,
    eeg: EEGTrigger,
) -> None:
    timer = core.Clock()
    while timer.getTime() < duration:
        check_for_quit(data_logger, eeg)
        for stim in stimuli:
            stim.draw()
        win.flip()


def collect_image_paths(folder: Path) -> List[Path]:
    valid_suffixes = {".jpg", ".png", ".jpeg"}
    if not folder.exists():
        logging.warning("Missing stimulus folder: %s", folder)
        return []
    paths = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in valid_suffixes
    ]
    if not paths:
        logging.warning("No image files found in %s", folder)
    return paths


def choose_image_for_type(photo_type: str, pools: Dict[str, List[Path]], counters: Dict[str, int]) -> Path:
    pool = pools[photo_type]
    if not pool:
        raise RuntimeError(f"No images available for photo_type '{photo_type}'.")
    path = pool[counters[photo_type] % len(pool)]
    counters[photo_type] += 1
    return path


def make_balanced_trial_list(
    total_trials: int,
    image_pools: Dict[str, List[Path]],
    enable_catch: bool = True,
    trial_phase: str = "main",
    start_index: int = 1,
    run_number: int = RUN_NUMBER,
    catch_trials_override: Optional[int] = None,
) -> List[TrialSpec]:
    conditions = [
        ("own", "reward"),
        ("own", "punishment"),
        ("other", "reward"),
        ("other", "punishment"),
    ]
    if DEBUG_REWARD_ONLY and not DEBUG_PUNISHMENT_ONLY:
        conditions = [cond for cond in conditions if cond[1] == "reward"]
    elif DEBUG_PUNISHMENT_ONLY and not DEBUG_REWARD_ONLY:
        conditions = [cond for cond in conditions if cond[1] == "punishment"]

    if not conditions:
        raise RuntimeError("No trial conditions available. Check debug filter settings.")

    repeats = math.ceil(total_trials / len(conditions))
    base_conditions = (conditions * repeats)[:total_trials]
    random.shuffle(base_conditions)

    image_counters = {"own": 0, "other": 0}
    for photo_type in image_counters:
        random.shuffle(image_pools[photo_type])

    catch_indices = set()
    if total_trials > 0 and (DEBUG_CATCH_ONLY or (ENABLE_CATCH_TRIALS and enable_catch)):
        if DEBUG_CATCH_ONLY:
            n_catch = total_trials
        elif catch_trials_override is not None:
            n_catch = catch_trials_override
        else:
            n_catch = 0
        catch_indices = choose_catch_indices(total_trials, n_catch, MIN_NONCATCH_GAP)

    trials: List[TrialSpec] = []
    for idx, (photo_type, cue_type) in enumerate(base_conditions, start=start_index):
        image_path = choose_image_for_type(photo_type, image_pools, image_counters)
        trials.append(
            TrialSpec(
                trial_number=idx,
                photo_type=photo_type,
                cue_type=cue_type,
                image_path=image_path,
                catch_trial=((idx - start_index) in catch_indices),
                trial_phase=trial_phase,
                run_number=run_number,
            )
        )
    return trials


def choose_catch_indices(total_trials: int, n_catch: int, min_gap: int) -> set:
    if n_catch <= 0:
        return set()
    available = list(range(total_trials))
    random.shuffle(available)
    chosen: List[int] = []
    for idx in available:
        if all(abs(idx - existing) > min_gap for existing in chosen):
            chosen.append(idx)
            if len(chosen) >= n_catch:
                break
    return set(sorted(chosen))


def get_feedback_labels(cue_type: str, hit: bool) -> Tuple[str, str, str]:
    if cue_type == "reward" and hit:
        return "Like 1", "Hit", "heart"
    if cue_type == "reward" and not hit:
        return "Like 0", "Miss", "heart"
    if cue_type == "punishment" and hit:
        return "Dislike 0", "Hit", "thumbs_down"
    return "Dislike 1", "Miss", "thumbs_down"


def get_feedback_phase2(cue_type: str, hit: bool) -> Tuple[str, str]:
    if cue_type == "reward" and hit:
        return "positive_comment", random.choice(POSITIVE_COMMENTS)
    if cue_type == "reward" and not hit:
        return "empty_bubble", ""
    if cue_type == "punishment" and hit:
        return "empty_bubble", ""
    return "negative_comment", random.choice(NEGATIVE_COMMENTS)


def create_window() -> visual.Window:
    return visual.Window(
        size=WINDOW_SIZE,
        fullscr=FULLSCREEN,
        color=BACKGROUND_COLOR,
        units="height",
        allowGUI=False,
    )


def build_static_stimuli(win: visual.Window) -> Dict[str, visual.BaseVisualStim]:
    stimuli: Dict[str, visual.BaseVisualStim] = {}
    stimuli["fixation"] = visual.TextStim(
        win,
        text="+",
        color="white",
        height=FIXATION_SIZE,
    )
    stimuli["cue_reward"] = visual.Circle(
        win,
        radius=CUE_SYMBOL_SIZE / 2,
        lineColor=CUE_SYMBOL_COLOR,
        fillColor=None,
        lineWidth=4,
    )
    stimuli["cue_punishment"] = visual.Rect(
        win,
        width=CUE_SYMBOL_SIZE,
        height=CUE_SYMBOL_SIZE,
        lineColor=CUE_SYMBOL_COLOR,
        fillColor=None,
        lineWidth=4,
    )
    stimuli["cue_reward_label"] = visual.TextStim(
        win,
        text="REWARD",
        color="white",
        height=0.03,
        pos=(0, -0.08),
    )
    stimuli["cue_punishment_label"] = visual.TextStim(
        win,
        text="PUNISHMENT",
        color="white",
        height=0.03,
        pos=(0, -0.08),
    )
    stimuli["target"] = visual.Rect(
        win,
        width=TARGET_SIZE,
        height=TARGET_SIZE,
        lineColor="white",
        fillColor="white",
    )
    stimuli["photo_frame"] = visual.Rect(
        win,
        width=PHOTO_SIZE + 0.01,
        height=PHOTO_SIZE + 0.01,
        lineColor="white",
        fillColor=None,
        lineWidth=2,
    )
    stimuli["catch_prompt"] = visual.TextStim(
        win,
        text="Catch trial\nChoose the cue you just saw",
        color="white",
        height=0.04,
        pos=(0, 0.24),
        alignText="center",
        wrapWidth=0.9,
    )
    stimuli["catch_reward_symbol"] = visual.Circle(
        win,
        radius=CUE_SYMBOL_SIZE / 2,
        lineColor=CUE_SYMBOL_COLOR,
        fillColor=None,
        lineWidth=4,
        pos=(-0.12, 0.02),
    )
    stimuli["catch_punishment_symbol"] = visual.Rect(
        win,
        width=CUE_SYMBOL_SIZE,
        height=CUE_SYMBOL_SIZE,
        lineColor=CUE_SYMBOL_COLOR,
        fillColor=None,
        lineWidth=4,
        pos=(0.12, 0.02),
    )
    stimuli["catch_reward_key"] = visual.TextStim(
        win,
        text=f"{CATCH_REWARD_KEY.upper()}",
        color="white",
        height=0.04,
        pos=(-0.12, -0.10),
    )
    stimuli["catch_punishment_key"] = visual.TextStim(
        win,
        text=f"{CATCH_PUNISHMENT_KEY.upper()}",
        color="white",
        height=0.04,
        pos=(0.12, -0.10),
    )
    stimuli["catch_reward_name"] = visual.TextStim(
        win,
        text="Reward",
        color="white",
        height=0.028,
        pos=(-0.12, -0.16),
    )
    stimuli["catch_punishment_name"] = visual.TextStim(
        win,
        text="Punishment",
        color="white",
        height=0.028,
        pos=(0.12, -0.16),
    )
    return stimuli


def draw_photo(win: visual.Window, image_stim: visual.ImageStim, photo_frame: visual.Rect) -> None:
    image_stim.draw()
    photo_frame.draw()
    win.flip()


def draw_heart_icon(win: visual.Window) -> None:
    left = visual.Circle(win, radius=0.055, pos=(-0.04, 0.03), fillColor="#E53935", lineColor="white", lineWidth=3)
    right = visual.Circle(win, radius=0.055, pos=(0.04, 0.03), fillColor="#E53935", lineColor="white", lineWidth=3)
    bottom = visual.ShapeStim(
        win,
        vertices=[(-0.095, 0.03), (0.0, -0.09), (0.095, 0.03)],
        fillColor="#E53935",
        lineColor="white",
        lineWidth=3,
        closeShape=True,
    )
    left.draw()
    right.draw()
    bottom.draw()


def draw_thumbsdown_icon(win: visual.Window) -> None:
    palm = visual.Rect(
        win, width=0.10, height=0.12, pos=(0.02, 0.00),
        fillColor="#E53935", lineColor="white", lineWidth=3
    )
    wrist = visual.Rect(
        win, width=0.06, height=0.05, pos=(-0.06, 0.06),
        fillColor="#E53935", lineColor="white", lineWidth=3
    )
    thumb = visual.Rect(
        win, width=0.04, height=0.10, pos=(-0.07, -0.07),
        fillColor="#E53935", lineColor="white", lineWidth=3
    )
    finger_1 = visual.Line(win, start=(0.05, 0.06), end=(0.05, -0.06), lineColor="white", lineWidth=2)
    finger_2 = visual.Line(win, start=(0.02, 0.06), end=(0.02, -0.06), lineColor="white", lineWidth=2)
    finger_3 = visual.Line(win, start=(-0.01, 0.06), end=(-0.01, -0.06), lineColor="white", lineWidth=2)
    palm.draw()
    wrist.draw()
    thumb.draw()
    finger_1.draw()
    finger_2.draw()
    finger_3.draw()


def draw_feedback_phase1(
    win: visual.Window,
    top_text: str,
    label_text: str,
    icon_stim: Optional[visual.ImageStim],
    data_logger: DataLogger,
    eeg: EEGTrigger,
) -> None:
    top = visual.TextStim(win, text=top_text, pos=(0, 0.28), color="white", height=0.06)
    label = visual.TextStim(win, text=label_text, pos=(0, -0.22), color="white", height=0.05)
    timer = core.Clock()
    while timer.getTime() < FEEDBACK1_DURATION:
        check_for_quit(data_logger, eeg)
        top.draw()
        if icon_stim is not None:
            icon_stim.draw()
        label.draw()
        win.flip()


def draw_feedback_phase2(
    win: visual.Window,
    comment_text: str,
    duration: float,
    data_logger: DataLogger,
    eeg: EEGTrigger,
) -> None:
    bubble = visual.Rect(
        win,
        width=0.75,
        height=0.28,
        pos=(0, 0.05),
        fillColor="#F5F5F5",
        lineColor="white",
        lineWidth=2,
    )
    tail = visual.ShapeStim(
        win,
        vertices=[(-0.10, -0.09), (-0.02, -0.09), (-0.08, -0.17)],
        fillColor="#F5F5F5",
        lineColor="white",
        lineWidth=2,
        closeShape=True,
    )
    comment = visual.TextStim(
        win,
        text=comment_text,
        color="black",
        pos=(0, 0.05),
        height=0.04,
        wrapWidth=0.65,
    )
    timer = core.Clock()
    while timer.getTime() < duration:
        check_for_quit(data_logger, eeg)
        bubble.draw()
        tail.draw()
        comment.draw()
        win.flip()


def make_optional_icon(win: visual.Window, path_value: Optional[str]) -> Optional[visual.ImageStim]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        logging.warning("Icon file not found: %s", path)
        return None
    return visual.ImageStim(win, image=str(path), size=(0.22, 0.22))


def load_feedback_icons(win: visual.Window) -> Dict[str, Optional[visual.ImageStim]]:
    return {
        "reward_hit": make_optional_icon(win, str(REWARD_HIT_ICON_FILE)),
        "reward_miss": make_optional_icon(win, str(REWARD_MISS_ICON_FILE)),
        "punishment_hit": make_optional_icon(win, str(PUNISHMENT_HIT_ICON_FILE)),
        "punishment_miss": make_optional_icon(win, str(PUNISHMENT_MISS_ICON_FILE)),
    }


def get_feedback_icon_key(cue_type: str, hit: bool) -> str:
    if cue_type == "reward" and hit:
        return "reward_hit"
    if cue_type == "reward" and not hit:
        return "reward_miss"
    if cue_type == "punishment" and hit:
        return "punishment_hit"
    return "punishment_miss"


def collect_catch_response(
    cue_type: str,
    win: visual.Window,
    static_stimuli: Dict[str, visual.BaseVisualStim],
    data_logger: DataLogger,
    eeg: EEGTrigger,
) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    clock = core.Clock()
    event.clearEvents(eventType="keyboard")
    while True:
        check_for_quit(data_logger, eeg)
        static_stimuli["catch_prompt"].draw()
        static_stimuli["catch_reward_symbol"].draw()
        static_stimuli["catch_punishment_symbol"].draw()
        static_stimuli["catch_reward_key"].draw()
        static_stimuli["catch_punishment_key"].draw()
        static_stimuli["catch_reward_name"].draw()
        static_stimuli["catch_punishment_name"].draw()
        win.flip()
        keys = event.getKeys(keyList=[CATCH_REWARD_KEY, CATCH_PUNISHMENT_KEY, QUIT_KEY])
        if keys:
            key_name = keys[0]
            if key_name == CATCH_REWARD_KEY:
                eeg.send("response_reward_key")
            elif key_name == CATCH_PUNISHMENT_KEY:
                eeg.send("response_punishment_key")
            expected = CATCH_REWARD_KEY if cue_type == "reward" else CATCH_PUNISHMENT_KEY
            return key_name, int(key_name == expected), clock.getTime() * 1000.0


def run_trial(
    trial: TrialSpec,
    config: TaskConfig,
    win: visual.Window,
    kb: keyboard.Keyboard,
    static_stimuli: Dict[str, visual.BaseVisualStim],
    adaptive: AdaptiveThreshold,
    data_logger: DataLogger,
    eeg: EEGTrigger,
    feedback_icons: Dict[str, Optional[visual.ImageStim]],
) -> None:
    eeg.send("trial_start")

    image_stim = visual.ImageStim(
        win,
        image=str(trial.image_path),
        size=(PHOTO_SIZE, PHOTO_SIZE),
    )

    eeg.send("photo_own" if trial.photo_type == "own" else "photo_other")
    show_for_duration(win, [image_stim, static_stimuli["photo_frame"]], PHOTO_DURATION, data_logger, eeg)

    cue_name = "cue_reward" if trial.cue_type == "reward" else "cue_punishment"
    cue_label_name = "cue_reward_label" if trial.cue_type == "reward" else "cue_punishment_label"
    cue_duration = random.uniform(CUE_DURATION_MIN, CUE_DURATION_MAX)
    eeg.send("cue_reward" if trial.cue_type == "reward" else "cue_punishment")
    show_for_duration(
        win,
        [static_stimuli[cue_name], static_stimuli[cue_label_name]],
        cue_duration,
        data_logger,
        eeg,
    )

    jitter = random.uniform(ANTICIPATION_MIN, ANTICIPATION_MAX)
    eeg.send("anticipation")
    show_for_duration(win, [static_stimuli["fixation"]], jitter, data_logger, eeg)

    response_key: Optional[str] = None
    reaction_time_ms: Optional[float] = None
    hit: Optional[bool] = None
    catch_response: Optional[str] = None
    catch_accuracy: Optional[int] = None
    target_presented = not trial.catch_trial

    if trial.catch_trial:
        eeg.send("catch_prompt")
        catch_response, catch_accuracy, reaction_time_ms = collect_catch_response(
            trial.cue_type, win, static_stimuli, data_logger, eeg
        )
        response_key = catch_response
    else:
        eeg.send("target")
        target_clock = core.Clock()
        event.clearEvents(eventType="keyboard")
        while target_clock.getTime() < TARGET_WINDOW:
            check_for_quit(data_logger, eeg)
            static_stimuli["target"].draw()
            win.flip()
            keys = event.getKeys(
                keyList=[MAIN_RESPONSE_KEY, QUIT_KEY],
                timeStamped=target_clock,
            )
            if keys:
                response_key = keys[0][0]
                reaction_time_ms = keys[0][1] * 1000.0
                hit = reaction_time_ms <= adaptive.current_ms
                eeg.send("response_space")
                break
        if response_key is None:
            hit = False
            reaction_time_ms = None

    adaptive_threshold_before_update = adaptive.current_ms
    if not trial.catch_trial:
        adaptive.update(hit)

    feedback_label = ""
    feedback_top_text = ""
    feedback_icon_type = ""
    feedback_phase2_type = "empty_bubble"
    comment_text = ""

    if trial.catch_trial:
        pass
    else:
        show_for_duration(win, [static_stimuli["fixation"]], POST_RESPONSE_DELAY, data_logger, eeg)
        feedback_label, feedback_top_text, feedback_icon_type = get_feedback_labels(trial.cue_type, bool(hit))
        feedback_phase2_type, comment_text = get_feedback_phase2(trial.cue_type, bool(hit))
        feedback_icon = feedback_icons.get(get_feedback_icon_key(trial.cue_type, bool(hit)))
        eeg.send("hit" if hit else "miss")
        eeg.send("feedback1")
        draw_feedback_phase1(
            win,
            top_text=feedback_top_text,
            label_text=feedback_label,
            icon_stim=feedback_icon,
            data_logger=data_logger,
            eeg=eeg,
        )
        feedback2_duration = random.uniform(FEEDBACK2_MIN_DURATION, FEEDBACK2_MAX_DURATION)
        eeg.send("feedback2")
        draw_feedback_phase2(win, comment_text, feedback2_duration, data_logger, eeg)

    row = {
        "participant_id": config.participant_id,
        "date_time": config.date_time,
        "trial_number": trial.trial_number,
        "trial_phase": trial.trial_phase,
        "run_number": trial.run_number,
        "photo_type": trial.photo_type,
        "image_filename": trial.image_path.name,
        "cue_type": trial.cue_type,
        "cue_symbol": CUE_SHAPE_REWARD if trial.cue_type == "reward" else CUE_SHAPE_PUNISHMENT,
        "catch_trial": int(trial.catch_trial),
        "catch_response": catch_response,
        "catch_accuracy": catch_accuracy,
        "target_presented": int(target_presented),
        "response_key": response_key,
        "reaction_time": reaction_time_ms,
        "hit": "" if trial.catch_trial else int(bool(hit)),
        "adaptive_threshold": adaptive_threshold_before_update,
        "feedback_phase1_label": feedback_label,
        "feedback_phase1_top_text": feedback_top_text,
        "feedback_phase1_icon_type": feedback_icon_type,
        "feedback_phase2_type": feedback_phase2_type,
        "comment_text": comment_text,
    }
    data_logger.append(row)

    show_for_duration(win, [static_stimuli["fixation"]], ITI_DURATION, data_logger, eeg)


def show_message_screen(
    win: visual.Window,
    message_text: str,
    data_logger: DataLogger,
    eeg: EEGTrigger,
    footer_text: Optional[str] = None,
) -> None:
    body = visual.TextStim(
        win,
        text=message_text,
        color="white",
        height=0.035,
        wrapWidth=1.0,
        pos=(0, 0.03),
    )
    footer = visual.TextStim(
        win,
        text=footer_text or f"Press {MAIN_RESPONSE_KEY.upper()} to continue",
        color="#BBBBBB",
        height=0.026,
        pos=(0, -0.34),
    )
    event.clearEvents(eventType="keyboard")
    while True:
        check_for_quit(data_logger, eeg)
        body.draw()
        footer.draw()
        win.flip()
        keys = event.getKeys()
        if MAIN_RESPONSE_KEY in keys:
            event.clearEvents(eventType="keyboard")
            break


def show_instructions(
    win: visual.Window,
    data_logger: DataLogger,
    eeg: EEGTrigger,
) -> None:
    instruction_pages = [
        (
            "Welcome\n\n"
            "In this task, you will see a photo, then a cue, then a target.\n\n"
            "Please look at the center of the screen and respond as quickly as possible."
        ),
        (
            "Own And Other Photos\n\n"
            "Some trials will show your own photos.\n"
            "Other trials will show other people's photos.\n\n"
            "Please pay attention to whose photo is shown."
        ),
        (
            "Cue Meaning\n\n"
            "A circle means REWARD.\n"
            "If you respond fast enough, you can get positive social feedback.\n\n"
            "A square means PUNISHMENT.\n"
            "If you respond fast enough, you can avoid negative social feedback."
        ),
        (
            "Main Response\n\n"
            "After a short delay, a white square target will appear.\n\n"
            f"Press {MAIN_RESPONSE_KEY.upper()} as quickly as possible when the target appears.\n\n"
            f"In the main task, you only need one key: {MAIN_RESPONSE_KEY.upper()}."
        ),
        (
            "Catch Trials\n\n"
            "Sometimes there will be a catch trial to check whether you noticed the cue.\n\n"
            f"On catch trials, press {CATCH_REWARD_KEY.upper()} for reward and "
            f"{CATCH_PUNISHMENT_KEY.upper()} for punishment.\n\n"
            "Catch trials allow a little more time than the main target."
        ),
        (
            "Feedback\n\n"
            "After each trial, you will see whether the outcome was Hit or Miss.\n\n"
            "Then you will see social feedback such as likes, dislikes, or comments."
        ),
        (
            "Practice\n\n"
            f"You will now complete a short practice block of {PRACTICE_TRIALS} trials.\n\n"
            "Use the practice to learn the sequence:\n"
            "photo -> cue -> delay -> target -> feedback."
        ),
        (
            "Start\n\n"
            "Try to respond quickly and stay focused.\n\n"
            f"Press {MAIN_RESPONSE_KEY.upper()} to begin.\n"
            f"Press {QUIT_KEY.upper()} at any time to quit safely."
        ),
    ]
    for page in instruction_pages:
        show_message_screen(win, page, data_logger, eeg)


def show_end_message(win: visual.Window, message: str, wait_sec: float = 3.0) -> None:
    stim = visual.TextStim(win, text=message, color="white", height=0.04, wrapWidth=0.9)
    timer = core.Clock()
    while timer.getTime() < wait_sec:
        stim.draw()
        win.flip()


def get_session_config() -> TaskConfig:
    now = datetime.now()
    dialog_data = {
        "participant_id": "",
        "debug_mode": DEFAULT_DEBUG_MODE,
    }
    dialog = gui.DlgFromDict(
        dictionary=dialog_data,
        title="SID Task Setup",
        fixed=[],
        order=["participant_id", "debug_mode"],
    )
    if not dialog.OK:
        raise SystemExit("User cancelled setup dialog.")

    participant_id = str(dialog_data["participant_id"]).strip() or "TEST"
    debug_mode = bool(dialog_data["debug_mode"])
    total_trials = DEBUG_TRIALS if debug_mode else TOTAL_TRIALS
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = DATA_DIR / f"sid_{participant_id}_{date_time}.csv"

    return TaskConfig(
        participant_id=participant_id,
        debug_mode=debug_mode,
        date_time=date_time,
        total_trials=total_trials,
        output_csv=output_csv,
    )


def validate_stimuli() -> Dict[str, List[Path]]:
    own_images = collect_image_paths(OWN_PHOTO_DIR)
    other_images = collect_image_paths(OTHER_PHOTO_DIR)
    if not own_images:
        raise RuntimeError(f"No usable own photos found in {OWN_PHOTO_DIR}")
    if not other_images:
        raise RuntimeError(f"No usable other photos found in {OTHER_PHOTO_DIR}")
    return {"own": own_images, "other": other_images}


def build_main_blocks(config: TaskConfig, image_pools: Dict[str, List[Path]]) -> List[List[TrialSpec]]:
    if config.debug_mode:
        return [
            make_balanced_trial_list(
                config.total_trials,
                image_pools,
                enable_catch=True,
                trial_phase="main",
                start_index=PRACTICE_TRIALS + 1,
                run_number=1,
                catch_trials_override=min(MAIN_CATCH_TRIALS_PER_BLOCK, config.total_trials),
            )
        ]

    blocks: List[List[TrialSpec]] = []
    start_index = PRACTICE_TRIALS + 1
    for block_number in range(1, TOTAL_BLOCKS + 1):
        block_trials = make_balanced_trial_list(
            BLOCK_TRIALS,
            image_pools,
            enable_catch=True,
            trial_phase="main",
            start_index=start_index,
            run_number=block_number,
            catch_trials_override=MAIN_CATCH_TRIALS_PER_BLOCK,
        )
        blocks.append(block_trials)
        start_index += BLOCK_TRIALS
    return blocks


def main() -> None:
    global ACTIVE_KEYBOARD
    random.seed()
    config = get_session_config()
    image_pools = validate_stimuli()
    practice_trials = make_balanced_trial_list(
        PRACTICE_TRIALS,
        image_pools,
        enable_catch=True,
        trial_phase="practice",
        start_index=1,
        run_number=0,
        catch_trials_override=PRACTICE_CATCH_TRIALS,
    )
    main_blocks = build_main_blocks(config, image_pools)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_logger = DataLogger(config.output_csv)
    eeg = EEGTrigger()
    win = create_window()
    kb = keyboard.Keyboard()
    ACTIVE_KEYBOARD = kb

    feedback_icons = load_feedback_icons(win)
    static_stimuli = build_static_stimuli(win)

    try:
        show_instructions(win, data_logger, eeg)
        practice_adaptive = AdaptiveThreshold()
        for trial in practice_trials:
            run_trial(
                trial=trial,
                config=config,
                win=win,
                kb=kb,
                static_stimuli=static_stimuli,
                adaptive=practice_adaptive,
                data_logger=data_logger,
                eeg=eeg,
                feedback_icons=feedback_icons,
            )
        show_message_screen(
            win,
            "Practice Complete\n\n"
            f"The main task will start now.\n\n"
            f"You will complete {len(main_blocks)} block(s)"
            + (f" of {BLOCK_TRIALS} trials each.\n\n" if not config.debug_mode else ".\n\n")
            + "Remember:\n"
            f"- Press {MAIN_RESPONSE_KEY.upper()} for the main target\n"
            f"- Press {CATCH_REWARD_KEY.upper()} or {CATCH_PUNISHMENT_KEY.upper()} on catch trials\n"
            f"- Press {QUIT_KEY.upper()} if you need to stop safely",
            data_logger,
            eeg,
            footer_text=f"Press {MAIN_RESPONSE_KEY.upper()} to start Block 1",
        )
        adaptive = AdaptiveThreshold()
        for block_index, block_trials in enumerate(main_blocks, start=1):
            for trial in block_trials:
                run_trial(
                    trial=trial,
                    config=config,
                    win=win,
                    kb=kb,
                    static_stimuli=static_stimuli,
                    adaptive=adaptive,
                    data_logger=data_logger,
                    eeg=eeg,
                    feedback_icons=feedback_icons,
                )
            if block_index < len(main_blocks):
                show_message_screen(
                    win,
                    f"End of Block {block_index}\n\n"
                    "You can take a short break.\n\n"
                    f"Press {MAIN_RESPONSE_KEY.upper()} when you are ready to continue.",
                    data_logger,
                    eeg,
                    footer_text=f"Press {MAIN_RESPONSE_KEY.upper()} to start Block {block_index + 1}",
                )
        data_logger.save()
        show_end_message(win, "Task complete.\nThank you!", wait_sec=3.0)
    except TaskAbort:
        data_logger.save()
        show_end_message(win, "Task stopped early.\nPartial data were saved.", wait_sec=2.5)
    finally:
        eeg.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
