# SocialFeedbackSID_Photo

A code-based PsychoPy prototype for a social feedback / Social Incentive Delay (SID) task using photo stimuli. The task presents own and other photos, reward and punishment cues, a speeded target response, catch trials, and social feedback outcomes.

This script is intended for piloting and demonstration purposes. It is not an exact reproduction of any single published paradigm.

## Contents

- `sid_psychopy_prototype.py` - main PsychoPy task script
- `stimuli/own_photos/` - self/own photo stimuli
- `stimuli/other_photos/` - other-person photo stimuli
- `stimuli/asset/` - feedback icons
- `sid_task_schematic_simple.svg` - task schematic

Output data are saved locally in `data/`. This folder is ignored by Git so participant data are not uploaded by default.

## Requirements

Install PsychoPy before running the task:

```bash
pip install psychopy pandas
```

If you plan to use EEG serial triggers, also install:

```bash
pip install pyserial
```

EEG triggers are disabled by default.

## Run the Task

From this folder, run:

```bash
python sid_psychopy_prototype.py
```

When the setup dialog opens, enter a participant ID and choose whether to run in debug mode.

Full mode runs:

- 8 practice trials
- 4 main blocks
- 30 trials per main block

## Key Responses

- `space` - respond to the target and continue instruction screens
- `r` - reward response on catch trials
- `p` - punishment response on catch trials
- `escape` - safely quit the task

If the task is stopped early, partial data are saved.

## Stimuli

The script expects image files in:

```text
stimuli/own_photos/
stimuli/other_photos/
stimuli/asset/
```

Supported photo formats are `.png`, `.jpg`, and `.jpeg`.

Default feedback icons:

```text
stimuli/asset/RewardHit.png
stimuli/asset/RewardMiss.png
stimuli/asset/PunishmentHit.png
stimuli/asset/PunishmentMiss.png
```

## Data Output

Each run creates a CSV file:

```text
data/sid_<participant_id>_<date_time>.csv
```

The CSV includes trial condition, photo type, cue type, catch-trial response, reaction time, hit/miss outcome, adaptive threshold, and feedback information.

## Editable Settings

Most task settings can be edited near the top of `sid_psychopy_prototype.py`, including timings, number of trials, fullscreen mode, catch-trial frequency, key mappings, feedback text, and EEG trigger settings.

Useful defaults:

```python
FULLSCREEN = False
PRACTICE_TRIALS = 8
BLOCK_TRIALS = 30
TOTAL_BLOCKS = 4
DEFAULT_DEBUG_MODE = False
EEG_ENABLED = False
```

## Privacy Note

The example photo stimuli are AI-generated and do not depict real individuals or original participant photos. If you replace them with identifiable, sensitive, or non-licensed images, keep the repository private or use demo stimuli before sharing publicly.

## References

This task was adapted from social incentive delay and social feedback task paradigms described in:

- Ait Oumeziane, B., Schryer-Praga, J., & Foti, D. (2017). “Why don’t they ‘like’ me more?”: Comparing the time courses of social and monetary reward processing. *Neuropsychologia, 107*, 48–59. https://doi.org/10.1016/j.neuropsychologia.2017.11.001

- Nicolaou, S., Vega, D., & Marco-Pallarés, J. (2025). Opening the Pandora box: Neural processing of self-relevant negative social information. *Biological Psychology, 194*, 108982. https://doi.org/10.1016/j.biopsycho.2024.108982

- Spreckelmeyer, K. N., Krach, S., Kohls, G., Rademacher, L., Irmak, A., Konrad, K., Kircher, T., & Gründer, G. (2009). Anticipation of monetary and social reward differently activates mesolimbic brain structures in men and women. *Social Cognitive and Affective Neuroscience, 4*(2), 158–165. https://doi.org/10.1093/scan/nsn051