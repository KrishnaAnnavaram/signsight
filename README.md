<div align="center">

# signsight — Weather-Robust Traffic-Sign Recognition

**signsight is a traffic-sign classifier with an honest robustness benchmark. It takes sign images through these steps to a class, a robustness report and a latency report:**

`split by track` → `augment with weather` → `train` → `evaluate on corruption suites` → `detect and classify photos`.

![Corruptions](https://img.shields.io/badge/Corruptions-6_x_5_severities-1F3864?style=for-the-badge)
![Ablation](https://img.shields.io/badge/Ablation-3_variants_x_3_seeds-2E5FD9?style=for-the-badge)
![CLI commands](https://img.shields.io/badge/CLI_commands-7-6E86E8?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-25_passing-3DA35B?style=for-the-badge)
![Offline demo](https://img.shields.io/badge/Offline_demo-Yes-F5C542?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-A0399B?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-HOG_%2B_logistic-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN_optional-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-images-3776AB?style=flat-square)
![GTSRB](https://img.shields.io/badge/Data-GTSRB_reader-555555?style=flat-square)
![Docs](https://img.shields.io/badge/Docs-ASD--STE100-5D6D7E?style=flat-square)

**[Summary](#1-summary)** ·
**[Workflow](#4-the-end-to-end-workflow)** ·
**[Run it](#13-how-to-run-signsight)** ·
**[Configuration](#134-environment-variables)** ·
**[Known problems](#16-known-problems)** ·
**[Glossary](#18-glossary)**

</div>

> [!NOTE]
> This README uses ASD-STE100 Simplified Technical English. The writing rules and the project
> vocabulary are in [`docs/ste-style-guide.md`](docs/ste-style-guide.md). Each term in the
> [Glossary](#18-glossary) has only one meaning.

> [!WARNING]
> Do not use signsight in a vehicle or in any safety function. It is a research benchmark, not a certified perception system.

---

signsight measures how fog, rain, glare, snow, blur and night change a traffic-sign classifier, and how much weather augmentation helps.
The split keeps each track of frames in one part, so the score does not come from near-copies of training images.
Each model is evaluated on clean held-out images and on corrupted copies at 5 severities, over several seeds.
The optional condition input comes from the image itself, not from a live weather service.
A colour detector crops the sign from a full photo before the classifier, and the latency covers all stages.

This README is the **one location that explains all of signsight**. It gives these topics:

- the general design
- each component and its procedure, step by step
- the decision rules
- the data map
- the runbook
- the validation results and the known problems

| If you are… | Read |
|---|---|
| A manager or reviewer | [1](#1-summary), [3](#3-design-rules), [4](#4-the-end-to-end-workflow), [15](#15-validation-results), [17](#17-key-points) |
| A developer who joins the project | All sections, in sequence. Keep [13](#13-how-to-run-signsight) and [16](#16-known-problems) open while you work |
| An operator who runs signsight | [13](#13-how-to-run-signsight), then the section for the component that you use |

---

## Table of contents

1. 🧭 [Summary](#1-summary)
2. 🏗️ [How signsight is built](#2-how-signsight-is-built)
   - 2.1 [Components](#21-components)
   - 2.2 [System context](#22-system-context)
   - 2.3 [Repository layout](#23-repository-layout)
3. 🛡️ [Design rules](#3-design-rules)
4. 🔄 [The end-to-end workflow](#4-the-end-to-end-workflow)
   - 4.1 [Full flow](#41-full-flow)
   - 4.2 [The life cycle of one photo](#42-the-life-cycle-of-one-photo)
   - 4.3 [Who does which step](#43-who-does-which-step)
5. 📥 [Data and the track split](#5-data-and-the-track-split)
6. 🌫️ [Corruptions and augmentation](#6-corruptions-and-augmentation)
7. 🔵 [The HOG classifier](#7-the-hog-classifier)
8. 🟢 [The condition model](#8-the-condition-model)
9. 🟣 [The CNN classifier](#9-the-cnn-classifier)
10. ⚖️ [The robustness benchmark](#10-the-robustness-benchmark)
11. 📷 [Detection, prediction and latency](#11-detection-prediction-and-latency)
12. 🗂️ [Data and file map](#12-data-and-file-map)
13. ▶️ [How to run signsight](#13-how-to-run-signsight)
    - 13.1 [Prerequisites](#131-prerequisites) · 13.2 [Installation](#132-installation) · 13.3 [Run signsight](#133-run-signsight) · 13.4 [Environment variables](#134-environment-variables)
14. 🧩 [How to extend signsight](#14-how-to-extend-signsight)
15. ✅ [Validation results](#15-validation-results)
16. ⚠️ [Known problems](#16-known-problems)
17. 📌 [Key points](#17-key-points)
18. 📖 [Glossary](#18-glossary)
19. 📄 [License](#19-license)

---

## 1. Summary

**The problem.** A sign classifier can show a high test accuracy and still fail in fog or at night. These questions are difficult:

- Does the test accuracy come from new signs, or from frames of a sign that the model already saw?
- How much does each weather effect reduce the accuracy, and at which strength?
- Does weather augmentation help, also for effects that the training did not use?
- Can a condition input help, when it comes from the image and not from a weather service?
- How fast is the full pipeline from a photo file to a class?

signsight gives each of these questions its own component. Each component has a test.

| Item | Value |
|---|---|
| Input | GTSRB training and test files, synthetic signs, or a folder of photos |
| Output | `model.joblib`, `metrics.json`, `benchmark.json`, predictions, a latency report |
| Classifiers | `hog` (HOG + colour + logistic regression, default) and `cnn` (PyTorch, extra `torch`) |
| Corruptions | fog, rain, glare, snow, blur, night, each at severities 1 to 5 |
| Variants | `baseline`, `augmented`, `augmented_condition` |
| Offline mode | All commands with synthetic signs and synthetic scenes |
| Safety | No track is in train and validation. The test part never guides training |
| Tests | **25** unit tests (`pytest`), 1 more skips without PyTorch |

```mermaid
flowchart LR
    IN["Sign images"] --> A["Track split"] --> B["Weather augmentation (train only)"] --> C["Classifier"] --> D["Clean and corruption suites"] --> OUT["Robustness report"]
    P["Photo"] --> DET["Detector"] --> C
```

---

## 2. How signsight is built

### 2.1 Components

| Component | Module | Purpose |
|---|---|---|
| Settings | `src/signsight/config.py` | Environment variables and `.env` loader |
| Data | `src/signsight/data.py` | `ImageSet`, GTSRB reader (zip or folder), track split, image split for the leakage check |
| Synthetic data | `src/signsight/synthetic.py` | Signs in tracks, and street-like scenes |
| Corruptions | `src/signsight/corruptions.py` | Six seeded corruptions, the training augmentation |
| Features | `src/signsight/features.py` | HOG, colour histograms, condition cues |
| HOG classifier | `src/signsight/models/hog.py` | `HogClassifier` and `ConditionModel` |
| CNN classifier | `src/signsight/models/cnn.py` | `CnnClassifier` (extra `torch`) |
| Evaluation | `src/signsight/evaluate.py` | Clean metrics, corruption suites, mCE, ablation, leakage check |
| Detector | `src/signsight/detect.py` | Colour mask, largest area, box, IoU |
| Pipeline | `src/signsight/pipeline.py` | Prediction for files, latency for each stage |
| CLI | `src/signsight/cli.py` | The `signsight` command with 7 subcommands |

The component map shows which module calls which module. An arrow points from the caller to the module that it uses.

```mermaid
flowchart TB
    CLI["cli.py<br/>signsight command"]
    CFG["config.py<br/>Settings, load_dotenv"]
    subgraph DATAIN["Data in"]
        DATA["data.py<br/>load_gtsrb_train, load_gtsrb_test,<br/>track_split, image_split"]
        SYN["synthetic.py<br/>make_signs, make_scene"]
    end
    subgraph MODELS["Models"]
        HOG["models/hog.py<br/>HogClassifier, ConditionModel"]
        CNN["models/cnn.py<br/>CnnClassifier, extra torch"]
        FEAT["features.py<br/>hog, colour, condition_cues"]
    end
    COR["corruptions.py<br/>CORRUPTIONS, augment, corrupt_batch"]
    EVA["evaluate.py<br/>clean_metrics, corruption_suite,<br/>ablation, leakage_gap"]
    subgraph PHOTO["Photos"]
        PIPE["pipeline.py<br/>predict_files, latency"]
        DET["detect.py<br/>detect"]
    end

    CLI --> CFG
    CLI --> DATA
    CLI --> SYN
    CLI --> HOG
    CLI -. "SIGNSIGHT_BACKEND=cnn" .-> CNN
    CLI --> EVA
    CLI --> PIPE
    SYN --> DATA
    HOG --> FEAT
    HOG --> COR
    CNN --> COR
    EVA --> COR
    EVA --> DATA
    PIPE --> DET
    PIPE --> DATA
```

### 2.2 System context

```mermaid
flowchart TB
    U["Researcher"] --> CLI["signsight CLI"]
    CLI --> DATA["GTSRB files or synthetic signs"]
    CLI --> TRAIN["Training: HOG or CNN"]
    TRAIN -.-> TORCH["PyTorch (optional)"]
    CLI --> EVAL["Corruption suites and ablation"]
    CLI --> PRED["Detector and classifier on photos"]
    EVAL --> OUT["runs/: model.joblib, metrics.json, benchmark.json"]
```

### 2.3 Repository layout

```
signsight/
├── .github/workflows/ci.yml     # CI: Python 3.11, pip install -e ".[dev]", pytest -q
├── .env.example                 # every environment variable, all values empty
├── pyproject.toml               # package, extras (torch, dev), signsight script
├── data/README.md               # synthetic signs, GTSRB files and terms, own photos
├── docs/ste-style-guide.md      # writing rules and project vocabulary
├── src/signsight/
│   ├── config.py  data.py  synthetic.py  corruptions.py  features.py
│   ├── models/                       # hog.py, cnn.py
│   ├── evaluate.py  detect.py  pipeline.py  cli.py
└── tests/                            # 26 tests (25 run without PyTorch), no network
```

---

## 3. Design rules

### 3.1 No track is in two parts
`track_split` uses `StratifiedGroupKFold` with the track ID as the group. A test checks that the parts share no track. `image_split` exists only to measure the leakage of a random split.

### 3.2 The test part never guides training
The CNN stops early on the validation part. With `--test-images` and `--test-gt`, the official GTSRB test set gives the final numbers, and nothing else uses it.

```mermaid
flowchart LR
    DATA[/"Training images<br/>GTSRB or synthetic"/] --> SPLIT["track_split"]
    SPLIT --> TR["train part"]
    SPLIT --> VA["validation part"]
    TR --> FIT["fit: augment, features,<br/>classifier"]
    VA --> ES["CNN early stopping:<br/>validation loss, patience 3"]
    ES --> FIT
    FIT --> EVAL["clean_metrics,<br/>corruption_suite"]
    OT[/"Official GTSRB test<br/>--test-images, --test-gt"/] -. "if given" .-> EVAL
    VA -. "if no test set" .-> EVAL
    EVAL --> REP[/"Final numbers only:<br/>metrics.json, benchmark.json"/]
```

### 3.3 Robustness is measured, not assumed
Each model is evaluated on corrupted copies of the held-out images, for each corruption and severity. The augmentation uses only fog, rain and glare, so snow, blur and night measure the transfer to unseen effects.

### 3.4 The condition comes from the image
The prototype copied one live weather reading to each image. signsight drops that input. The `ConditionModel` predicts the condition of each image from global image cues, and the classifier gets these probabilities as features.

### 3.5 Rare classes are visible
The classifiers use class-balanced weights. The reports give macro-F1 and the 5 classes with the lowest recall, not only the accuracy.

### 3.6 A photo is cropped before classification
The classifier learns cropped signs. `predict_files` runs the detector on a full photo first, and gives `unknown` below the confidence limit.

### 3.7 All randomness has a seed, and no code asks for input
Data, corruptions, splits, models and the ablation use explicit seeds. Each command takes its paths as options. No code calls `input()`, and signsight needs no API key.

---

## 4. The end-to-end workflow

### 4.1 Full flow

```mermaid
flowchart TD
    IN[/"--data: synthetic,<br/>or a GTSRB zip or folder"/] --> SRC{"Data source"}
    SRC -- "synthetic" --> SYN["make_signs"]
    SRC -- "GTSRB zip or folder" --> GT["load_gtsrb_train:<br/>ROI crop, resize"]
    SYN --> SPLIT["track_split, seed"]
    GT --> SPLIT
    SPLIT -- "train" --> AQ{"--augment or an<br/>augmented variant?"}
    AQ -- "yes" --> AUG["augment: fog, rain,<br/>glare copies"]
    AQ -- "no" --> FIT
    AUG --> COND["ConditionModel<br/>augmented_condition only"]
    AUG --> FIT["HogClassifier or CnnClassifier"]
    COND --> FIT
    SPLIT -- "validation" --> ES["Early stopping, CNN"]
    ES --> FIT
    FIT --> CLEAN["clean_metrics"]
    FIT --> SUITE["corruption_suite:<br/>6 corruptions × severities"]
    OT[/"Official GTSRB test, optional"/] --> CLEAN
    OT --> SUITE
    CLEAN --> REP[("runs/<br/>model.joblib, metrics.json, benchmark.json")]
    SUITE --> REP
    REP --> HUMAN{{"HUMAN<br/>researcher reads the<br/>robustness report"}}
    PH[/"Folder of photos"/] --> PRED["predict, latency:<br/>detect, crop, classify"]
    REP --> PRED
    PRED --> OUT[/"Class, confidence and box,<br/>or unknown"/]

    classDef human fill:#fff3cd,stroke:#b8901f,color:#3d2f00,font-weight:bold
    class HUMAN human
```

### 4.2 The life cycle of one photo

```mermaid
stateDiagram-v2
    state "Image file in the folder" as Listed
    state "RGB image" as Rgb
    state "Sign box with padding" as Boxed
    state "Full image, no box" as NoBox
    state "Crop at SIGNSIGHT_IMAGE_SIZE" as Crop
    state "Class probabilities" as Proba
    state "Known class" as Known
    state "unknown" as Unknown
    [*] --> Listed: list_images, image suffix
    Listed --> Rgb: Image.open, convert to RGB
    Rgb --> Boxed: detect finds a coloured area
    Rgb --> NoBox: no area large enough, or no-detect option
    Boxed --> Crop: crop_resize of the box
    NoBox --> Crop: crop_resize of the full image
    Crop --> Proba: model.predict_proba
    Proba --> Known: top probability at or above min-confidence
    Proba --> Unknown: top probability below min-confidence
    Known --> [*]: print class, confidence, box
    Unknown --> [*]: print unknown, confidence, box
```

1. `list_images` finds the photo in the folder.
2. The pipeline reads the photo and converts it to RGB.
3. `detect` finds the largest area with sign colours and returns a box with 12 % padding.
4. `crop_resize` cuts the box and resizes it to `SIGNSIGHT_IMAGE_SIZE`.
5. The classifier gives the class probabilities.
6. If the top probability is below `--min-confidence`, the answer is `unknown`.
7. The CLI prints the class, the confidence and the box.

### 4.3 Who does which step

```mermaid
sequenceDiagram
    autonumber
    actor R as Researcher
    participant CLI as signsight CLI
    participant DATA as data.py
    participant MOD as HogClassifier
    participant EVA as evaluate.py
    participant FS as runs/ folder
    participant PIPE as pipeline.py

    R->>CLI: signsight train --augment
    CLI->>CLI: load_dotenv, Settings.from_env
    CLI->>DATA: make_signs or load_gtsrb_train
    CLI->>DATA: track_split(data, seed)
    DATA-->>CLI: train and validation parts
    CLI->>CLI: print the counts and the shared tracks
    CLI->>MOD: fit(train images, train labels)
    MOD->>MOD: augment, sign_features, StandardScaler, LogisticRegression
    CLI->>EVA: clean_metrics(model, target)
    CLI->>EVA: corruption_suite, severities 1, 3, 5
    EVA-->>CLI: accuracy, macro-F1, ECE, suite
    CLI->>FS: model.joblib, metrics.json
    CLI-->>R: clean and corruption summary
    R->>CLI: signsight predict --model --images
    CLI->>FS: joblib.load model.joblib
    CLI->>PIPE: predict_files(model, class_names, paths)
    PIPE->>PIPE: detect, crop_resize, predict_proba
    PIPE-->>CLI: list of Prediction
    CLI-->>R: file, class or unknown, confidence, box
```

---

## 5. Data and the track split

**Purpose.** Load sign crops and split them so that no physical sign is in two parts.

```mermaid
flowchart TD
    IN[/"--data value"/] --> Q{"synthetic?"}
    Q -- "yes" --> MS["make_signs: tracks_per_class,<br/>frames, the sign grows"]
    Q -- "no" --> Z{"Suffix .zip?"}
    Z -- "yes" --> ZR["Find GT-*.csv in the zip"]
    Z -- "no" --> FR["Find GT-*.csv in the folder"]
    ZR --> NG{"No GT file?"}
    FR --> NG
    NG -- "yes" --> ERR[/"DataError"/]
    NG -- "no" --> ROW["Each row: ROI crop, resize,<br/>ClassId, track ID from the file name"]
    ROW --> SET["ImageSet: images, labels,<br/>groups, class_names"]
    MS --> SET
    SET --> SPL["track_split: StratifiedGroupKFold,<br/>5 folds, shuffle, seed"]
    SPL --> OUT[/"First fold: train part<br/>and validation part"/]
```

| Source | Reader | Track ID |
|---|---|---|
| GTSRB training zip or folder | `load_gtsrb_train` | `<class>-<track>` from the file name `<track>_<frame>.ppm` |
| GTSRB official test | `load_gtsrb_test` | One ID for each image |
| Synthetic signs | `make_signs(tracks_per_class, frames)` | `<class>-<track>` |

**Procedure**

1. Read the `GT-*.csv` rows of each class folder, in the zip or on disk.
2. Crop each image to its region of interest and resize it.
3. Split with `StratifiedGroupKFold` (5 folds for a validation share of 0.2) and keep the first fold as validation.

---

## 6. Corruptions and augmentation

**Purpose.** Make seeded weather and light effects for training copies and for the evaluation suites.

```mermaid
flowchart LR
    subgraph TRAIN["Training: augment"]
        T1[/"Training images,<br/>share 0.5"/] --> T2["Seeded pick<br/>of images"]
        T2 --> T3["Random corruption from fog,<br/>rain, glare, severity 1 to 5"]
        T3 --> T4[/"Clean images + corrupted copies,<br/>condition labels 0 to 3"/]
    end
    subgraph EVAL["Evaluation: corrupt_batch"]
        E1[/"Held-out images"/] --> E2["For each of the 6 corruptions<br/>and each severity"]
        E2 --> E3["Fixed seed:<br/>seed + 100 × k + severity"]
        E3 --> E4[/"Corrupted copies<br/>for one suite cell"/]
    end
```

| Corruption | Effect | Used in augmentation |
|---|---|---|
| `fog` | Blend with grey haze, 20 % to 68 % | Yes |
| `rain` | Random streaks and a darker image | Yes |
| `glare` | Bright radial spot, 50 % to 100 % strength | Yes |
| `snow` | White flakes on 2 % to 14 % of pixels, lower contrast | No |
| `blur` | Gaussian blur, radius 0.5 to 2.0 px at 32 px | No |
| `night` | Brightness 70 % to 24 %, with sensor noise | No |

**Rules**

- Each corruption is a function of the image, the severity (1 to 5) and a seeded generator.
- `augment` adds corrupted COPIES of a share (default 50 %) of the training images, with a random corruption and severity. The clean images stay.
- The evaluation corrupts copies of the held-out images with a fixed seed for each corruption and severity.

---

## 7. The HOG classifier

**Purpose.** A fast, offline classifier that runs on a CPU.

```mermaid
flowchart LR
    IN[/"Training images<br/>and labels"/] --> A{"augment_share<br/>above 0?"}
    A -- "yes" --> AUG["augment: add<br/>corrupted copies"]
    A -- "no" --> F["sign_features"]
    AUG --> F
    F --> H["hog: 8 × 8 cells, 9 bins,<br/>2 × 2 blocks, L2-Hys"]
    F --> C["colour: 8-bin RGB histograms<br/>+ mean colour"]
    H --> X["Feature row:<br/>324 + 27 values at 32 px"]
    C --> X
    X --> UC{"use_condition?"}
    UC -- "yes" --> CP["Add 4 condition<br/>probabilities, section 8"]
    UC -- "no" --> SC["StandardScaler"]
    CP --> SC
    SC --> LOG["LogisticRegression C 0.5,<br/>class_weight balanced"]
    LOG --> OUT[/"predict_proba, predict"/]
```

| Step | Value |
|---|---|
| Gradient features | HOG, 8 × 8 px cells, 9 direction bins, 2 × 2 blocks, L2-Hys |
| Colour features | 8-bin histogram of each RGB channel, plus the mean colour |
| Scaler | `StandardScaler`, fit on the training part |
| Classifier | `LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)` |

At 32 px, the HOG part has 324 values and the colour part has 27 values.

---

## 8. The condition model

**Purpose.** Give the classifier a condition input that changes with each image.

```mermaid
flowchart TD
    CHK{"augment_share above 0?"} -- "no" --> ERR[/"ValueError: use_condition<br/>needs augmented training data"/]
    CHK -- "yes" --> IN[/"Augmented training images<br/>and condition labels"/]
    IN --> CUES["condition_cues: 7 values<br/>brightness, contrast, saturation,<br/>edge density, bright share, p5, p95"]
    CUES --> FIT["ConditionModel.fit:<br/>StandardScaler, LogisticRegression"]
    CUES --> OOF["cross_val_predict,<br/>cv 3, predict_proba"]
    OOF --> TRX[/"4 out-of-fold probabilities<br/>added to the training features"/]
    NEW[/"New or held-out images"/] --> CUES2["condition_cues"]
    CUES2 --> PP["ConditionModel.predict_proba"]
    FIT --> PP
    PP --> PX[/"4 probabilities added<br/>to the sign features"/]
```

**Procedure**

1. Calculate 7 condition cues of each image: mean brightness, contrast, saturation, edge density, share of bright pixels, and the 5th and 95th brightness percentiles.
2. Fit a logistic regression that predicts the condition (clean, fog, rain, glare) of the augmented training images.
3. For the training rows, get the condition probabilities with 3-fold cross-validation.
4. Add the 4 probabilities to the sign features.
5. At prediction, use the fitted condition model for the probabilities.

---

## 9. The CNN classifier

**Purpose.** A small convolutional network for larger datasets (extra `torch`).

```mermaid
flowchart TD
    IN[/"Training images, labels,<br/>validation part"/] --> W["Class weights from counts,<br/>torch.manual_seed"]
    W --> EP["Epoch: shuffle the order"]
    EP --> B["Batch of 128"]
    B --> AU["augment_share of the batch:<br/>random fog, rain or glare"]
    AU --> ST["Forward, weighted cross-entropy,<br/>backward, Adam step"]
    ST --> MORE{"More batches?"}
    MORE -- "yes" --> B
    MORE -- "no" --> VL{"Validation part given?"}
    VL -- "yes" --> L["Validation loss"]
    VL -- "no" --> NXT{"More epochs?"}
    L --> BEST{"Lower than the<br/>best loss − 0.0001?"}
    BEST -- "yes" --> SAVE["Keep a copy of the state,<br/>reset the patience count"]
    BEST -- "no" --> BAD{"3 epochs<br/>with no gain?"}
    SAVE --> NXT
    BAD -- "no" --> NXT
    BAD -- "yes" --> STOP["Load the best state"]
    NXT -- "yes" --> EP
    NXT -- "no" --> STOP
    STOP --> OUT[/"CnnClassifier:<br/>softmax predict_proba"/]
```

| Item | Value |
|---|---|
| Layers | 2 × (conv 32) → pool → 2 × (conv 64) → pool → conv 128 → global average pool → dropout 0.3 → linear |
| Loss | Cross-entropy with class weights |
| Optimizer | Adam, learning rate 0.002, batch 128 |
| Augmentation | With `--augment` or the `augmented` variant: on each training batch, 50 % of images get a random fog, rain or glare corruption. The `baseline` variant uses 0 % |
| Early stopping | Validation loss, patience 3, the best state is kept |
| Epochs | `SIGNSIGHT_EPOCHS` (default 15) |

---

## 10. The robustness benchmark

**Purpose.** Compare the variants over several seeds on clean and corrupted images.

```mermaid
flowchart TD
    IN[/"ImageSet, variants,<br/>SIGNSIGHT_SEEDS seeds from SIGNSIGHT_SEED"/] --> S["For each seed:<br/>track_split"]
    S --> T{"Official test set given?"}
    T -- "yes" --> TG["Target: official test"]
    T -- "no" --> TV["Target: held-out tracks"]
    TG --> V["For each variant: fit on train,<br/>validation for early stopping"]
    TV --> V
    V --> CM["clean_metrics:<br/>accuracy, macro-F1"]
    V --> CS["corruption_suite:<br/>6 corruptions, severities 1, 3, 5"]
    CS --> CA["Corrupt accuracy:<br/>mean of the suite"]
    CS --> MCE["mCE: errors ÷ errors of baseline,<br/>mean over corruptions"]
    CM --> SUM["Mean and std over seeds,<br/>accuracy by corruption"]
    CA --> SUM
    MCE --> SUM
    SUM --> OUT[/"benchmark.json: variants"/]
```

| Variant | Training data | Condition input |
|---|---|---|
| `baseline` | Clean training images | No |
| `augmented` | Clean images + fog, rain and glare copies | No |
| `augmented_condition` | Same as `augmented` | Yes (HOG only) |

| Metric | Meaning |
|---|---|
| Clean accuracy, macro-F1 | On the held-out tracks, or the official test set |
| Corrupt accuracy | Mean accuracy over 6 corruptions and severities 1, 3 and 5 |
| mCE | For each corruption, sum of errors over severities ÷ the same sum of `baseline`. Then the mean. Lower is better |
| ECE | Expected calibration error, 10 bins |
| Leakage gap | Accuracy with a random image split − accuracy with the track split |

Each value is the mean and the standard deviation over `SIGNSIGHT_SEEDS` seeds (default 3).

The leakage check trains the `baseline` variant two times on the same data:

```mermaid
flowchart LR
    IN[/"ImageSet and the<br/>baseline variant"/] --> IS["image_split:<br/>random, stratified"]
    IN --> TS["track_split:<br/>StratifiedGroupKFold"]
    IS --> F1["fit, then accuracy<br/>on its validation part"]
    TS --> F2["fit, then accuracy<br/>on its validation part"]
    F1 --> GAP["gap = image_split<br/>− track_split"]
    F2 --> GAP
    GAP --> OUT[/"leakage: printed report,<br/>and benchmark.json for benchmark"/]
```

---

## 11. Detection, prediction and latency

**Purpose.** Classify full photos, and measure the time of the full pipeline.

```mermaid
flowchart TD
    IN[/"RGB image"/] --> STEP["Use every n-th pixel:<br/>n = long side ÷ 128"]
    STEP --> MASK["sign_colour_mask:<br/>red, blue or yellow pixels"]
    MASK --> COMP["_largest_component:<br/>4-connected flood fill"]
    COMP --> Q{"Area at least 0.2 %<br/>of the small image?"}
    Q -- "no" --> NONE[/"None: no crop"/]
    Q -- "yes" --> BOX["Box of the area,<br/>scaled back to full size"]
    BOX --> PAD["Add 12 % padding,<br/>clip to the image"]
    PAD --> OUT[/"Box x1, y1, x2, y2"/]
```

| Detector rule | Value |
|---|---|
| Red pixel | R > 110, R > 1.6 G, R > 1.6 B |
| Blue pixel | B > 100, B > 1.4 R, B > 1.15 G |
| Yellow pixel | R > 150, G > 110, B < 0.6 G |
| Work size | About 128 px on the long side |
| Minimum area | 0.2 % of the image |
| Padding | 12 % of the box on each side |

`latency` reads each file, detects, crops and classifies one image at a time after 3 warm-up images. It reports the median (p50) and the 95th percentile (p95) in milliseconds for each stage and for the total.

```mermaid
flowchart LR
    P[/"Image paths"/] --> W["3 warm-up runs,<br/>not counted"]
    W --> RD["read:<br/>open, convert to RGB"]
    RD --> DT["detect"]
    DT --> CR["crop:<br/>crop_resize"]
    CR --> CL["classify:<br/>predict_proba"]
    CL --> REP[/"p50 and p95 ms for each stage<br/>and the total, fps_p50"/]
```

---

## 12. Data and file map

| Path | Committed? | Contents |
|---|---|---|
| `data/README.md` | Yes | Synthetic signs, GTSRB files and terms, own photos |
| `data/*` | No (git ignores it) | GTSRB zips, scenes, photos |
| `runs/model.joblib` | No (git ignores it) | The model, the class names and the image size |
| `runs/metrics.json` | No (git ignores it) | Clean metrics and the corruption suite of `train` |
| `runs/benchmark.json` | No (git ignores it) | The ablation and the leakage check |
| `.env` | No (git ignores it) | Local settings |
| `.env.example` | Yes | All variable names, no values |

---

## 13. How to run signsight

### 13.1 Prerequisites

| Need | For |
|---|---|
| Python 3.11+ | All components |
| `numpy`, `pillow`, `scikit-learn>=1.4` | Core (installed with the package) |
| `torch` (extra `torch`) | `SIGNSIGHT_BACKEND=cnn` |
| GTSRB files | Real results (optional) |

### 13.2 Installation

```bash
git clone https://github.com/KrishnaAnnavaram/signsight.git
cd signsight
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ,torch for the CNN
```

### 13.3 Run signsight

Offline (no key, no network):

```bash
signsight demo                                        # ablation, leakage check, scene photos, latency
signsight train --augment --out runs                  # synthetic signs
signsight make-scenes --out data/scenes --n 20
signsight predict --model runs/model.joblib --images data/scenes
signsight latency --model runs/model.joblib --images data/scenes
signsight benchmark --out runs
```

With GTSRB:

```bash
signsight benchmark --data data/GTSRB_Final_Training_Images.zip \
    --test-images data/GTSRB/Final_Test/Images --test-gt data/GT-final_test.csv
SIGNSIGHT_BACKEND=cnn SIGNSIGHT_IMAGE_SIZE=48 signsight train --data data/GTSRB_Final_Training_Images.zip --augment
signsight evaluate --model runs/model.joblib --data data/GTSRB_Final_Training_Images.zip
```

| Command | What it does |
|---|---|
| `demo` | Runs the ablation, the leakage check, scene prediction with and without the detector, and latency |
| `train` | Trains one variant on the track split and saves the model and metrics |
| `evaluate` | Prints clean metrics and the full corruption suite (severities 1 to 5) of a saved model |
| `benchmark` | Runs the ablation over seeds and the leakage check, and writes `benchmark.json` |
| `predict` | Detects and classifies each image in a folder |
| `latency` | Measures the time of each pipeline stage |
| `make-scenes` | Writes synthetic street-like images with one sign each |

The diagram shows the order of the commands and the files that connect them.

```mermaid
flowchart LR
    INS["pip install -e .[dev]"] --> TRN["signsight train"]
    INS --> BEN["signsight benchmark"]
    INS --> MS["signsight make-scenes"]
    INS --> DEMO["signsight demo<br/>writes no files"]
    GT[("GTSRB zip or folder,<br/>optional")] -- "--data" --> TRN
    GT -- "--data" --> BEN
    TRN --> MOD[("runs/model.joblib<br/>runs/metrics.json")]
    BEN --> BJ[("runs/benchmark.json")]
    MS --> SC[("data/scenes/*.png")]
    MOD --> EV["signsight evaluate"]
    MOD --> PR["signsight predict"]
    MOD --> LA["signsight latency"]
    SC --> PR
    SC --> LA
```

### 13.4 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `SIGNSIGHT_DATA_DIR` | Settings | Data folder. Default `data`. No command reads this value. Give the paths with `--data` and `--images` |
| `SIGNSIGHT_OUT` | `train`, `benchmark` | Output folder. Default `runs` |
| `SIGNSIGHT_SEED` | All components | First seed. Default 42 |
| `SIGNSIGHT_IMAGE_SIZE` | Data, models | Crop size in pixels, a multiple of 8 and at least 16. Default 32 |
| `SIGNSIGHT_BACKEND` | `train`, `benchmark` | `hog` (default) or `cnn` |
| `SIGNSIGHT_EPOCHS` | CNN | Maximum epochs. Default 15 |
| `SIGNSIGHT_SEEDS` | `benchmark`, `demo` | Number of seeds. Default 3 |

The CLI reads a local `.env` file. A variable that is already in the environment wins. A value that is not valid stops the command with `error:`. signsight needs no credentials.

```mermaid
flowchart LR
    ENV[/".env file"/] --> LD["load_dotenv:<br/>sets only absent variables"]
    PENV[/"Process environment"/] --> FE["Settings.from_env"]
    LD --> FE
    FE --> CHK{"Values valid?<br/>integers, backend hog or cnn,<br/>image size, epochs, seeds"}
    CHK -- "yes" --> SET[/"Settings: data_dir, out_dir, seed,<br/>image_size, backend, epochs, seeds"/]
    CHK -- "no" --> ERR[/"ConfigError: the CLI prints<br/>error: and returns 1"/]
```

---

## 14. How to extend signsight

| You want to… | Do this | Code change? |
|---|---|---|
| Use larger crops | Set `SIGNSIGHT_IMAGE_SIZE=48` | No |
| Use the CNN | Install the extra `torch` and set `SIGNSIGHT_BACKEND=cnn` | No |
| Add a corruption | Write a function `(image, severity, rng)` and add it to `CORRUPTIONS` | Small |
| Use a pretrained backbone | Write a class with `fit`, `predict_proba`, `predict` and `classes_` | Yes |
| Use a trained detector | Replace `detect` with a function that returns a box | Yes |
| Export a model | Add an ONNX export for the CNN | Yes |

Planned milestones (not built):

- **M6:** GTSRB results with the official test set for the HOG and CNN variants, in this README.
- **M7:** a real adverse-weather sign set, to check the synthetic corruptions.
- **M8:** a trained detector and an ONNX export with latency on the target hardware.

---

## 15. Validation results

All numbers come from the **synthetic** signs (10 classes, 12 tracks of 10 frames for each class, 32 px, seeds 42 to 44). They are not GTSRB results.

| Validation | Result | Command |
|---|---|---|
| Unit tests (CI installs only `.[dev]`) | **25 passed**, 1 skipped (PyTorch absent) | `pytest -q` |
| Unit tests with PyTorch | **26 passed** | `pip install -e ".[dev,torch]"`, `pytest -q` |
| `baseline` | clean accuracy 0.947 ± 0.034, corrupt accuracy 0.413 ± 0.011, mCE 1.000 | `signsight demo` |
| `augmented` | clean accuracy 0.928 ± 0.032, corrupt accuracy 0.600 ± 0.024, mCE 0.685 ± 0.027 | `signsight demo` |
| `augmented_condition` | clean accuracy 0.926 ± 0.022, corrupt accuracy 0.597 ± 0.021, mCE 0.688 ± 0.024 | `signsight demo` |
| Severity 3, `baseline` → `augmented` | fog 0.33 → 0.76, rain 0.36 → 0.68, glare 0.40 → 0.57, snow 0.18 → 0.40, blur 0.59 → 0.73, night 0.10 → 0.49 | `signsight demo` |
| Leakage check | image split 1.000, track split 0.963, gap +0.037 | `signsight demo` |
| 20 scene photos | accuracy 0.80 with the detector, 0.10 without it | `signsight demo` |
| Latency, HOG, one image | p50 17.0 ms, p95 20.3 ms (read + detect + crop + classify, CPU) | `signsight demo` |

Augmentation with fog, rain and glare improves the accuracy on all six corruptions, also on snow, blur and night, which it did not use.
It costs about 2 points of clean accuracy.
The condition input gives no measurable gain here: the difference is smaller than one standard deviation.
The detector is necessary for full photos. Without it, the accuracy falls to chance level.
The prototype reported a high accuracy on a random image split (prototype result, not reproduced here). The leakage check shows why such a split gives too high a score.

---

## 16. Known problems

Read these problems before you use signsight in production.

| # | Area | Problem | Impact and action |
|---|---|---|---|
| 1 | Data | CI and the demo use only synthetic signs. GTSRB results are not reproduced in CI | Run `signsight benchmark` with the GTSRB files and the official test set (M6) |
| 2 | Corruptions | The corruptions are simple image effects, not real weather | Check the results on a real adverse-weather set (M7) |
| 3 | Condition input | The condition model knows only clean, fog, rain and glare | A gain on real weather is not shown. Keep the `augmented` variant as the default |
| 4 | Detector | The colour detector finds the largest coloured area. Red cars, blue sky parts or two signs confuse it | Use a trained detector for real photos (M8) |
| 5 | Domain | GTSRB shows German signs from one camera type. Other countries and cameras are a different domain | Check the `unknown` rate and the confidence on your photos |
| 6 | HOG model | HOG at 32 px loses fine details, for example the digits of speed limits | Use the CNN at 48 px for GTSRB |
| 7 | Latency | The latency is for one CPU, one image at a time, HOG model | Measure again on the target hardware with the final model |
| 8 | Safety | No result here supports use in a vehicle | Use signsight only for research |

---

## 17. Key points

1. **No track is in two parts.** The track split removes the leakage of near-copies.
2. **Robustness is measured on 6 corruptions at 5 severities.** Over seeds, with mCE against the baseline.
3. **Augmentation helps, also on unseen effects.** Corrupt accuracy rises from 0.413 to 0.600 on the synthetic benchmark.
4. **The condition comes from the image.** No constant weather reading, and no API key.
5. **Photos are cropped before classification.** The detector and the latency cover the full pipeline.
6. **The full benchmark runs offline.** All 25 core tests run with no network.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **Augmentation** | Corrupted copies of training images that are added to the training data |
| **Condition** | The predicted imaging condition of an image: clean, fog, rain or glare |
| **Corruption** | A synthetic weather or light effect on an image |
| **Corruption suite** | Corrupted copies of the held-out images for each corruption and severity |
| **Crop** | The image part inside the region of interest or the detector box |
| **ECE** | Expected calibration error |
| **HOG** | Histogram of oriented gradients |
| **Latency** | Milliseconds for one image, from file read to class |
| **mCE** | Errors over severities divided by the errors of the baseline, mean over corruptions |
| **Severity** | The strength of a corruption, 1 to 5 |
| **Track** | All frames of one physical sign |
| **Variant** | One model setting of the ablation |

---

## 19. License

[MIT](LICENSE) © 2026 Krishna Annavaram
