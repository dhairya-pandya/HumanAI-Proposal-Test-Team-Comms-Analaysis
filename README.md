# GSoC 2026 · ISSR Test Submission
## Team Communication Processing and Analysis in Human-Factors Simulated Environment

**Organization:** HumanAI Foundation / ISSR, University of Alabama (TRIP Laboratory)
**Project:** Team Communication Analysis
**Dataset:** [AMI Meeting Corpus on HuggingFace](https://huggingface.co/datasets/edinburghcstr/ami)

---

## Data Resource Selection: Why the AMI Meeting Corpus?

The first step was finding a publicly available dataset that genuinely replicates conditions in the TRIP Laboratory. In the TRIP Lab, groups of two or more participants have structured, task-oriented discussions while wearing headset microphones and being recorded by room microphones, creating a multi-party, multi-channel team communication environment.

After surveying **11 open-access datasets** (detailed below), the **AMI Meeting Corpus** was the clear choice. Here is why:

### Why AMI Is the Best Option

**1. It replicates the TRIP Lab's setup closely.**
AMI captures groups of 3 to 5 people in structured design-team discussions with assigned roles: project manager, marketing expert, UI designer, and industrial designer. This maps directly to the TRIP Lab's simulated team environment, where participants communicate in groups under defined task roles. Most other datasets, VoxConverse, LibriCSS, VOICES, are either scripted, synthetic, or based on informal scenarios that have nothing to do with controlled simulation settings.

**2. It has both audio and video.**
AMI provides synchronized audio and video recordings, making it one of the few datasets that captures the full communication picture, speech and visual cues together. Most of the surveyed alternatives are audio-only, which matters for a project centered on analyzing human factors in team communication.

**3. It is large enough to be useful.**
At roughly 100 hours across 171 sessions, AMI is substantial enough to support model training and draw meaningful conclusions. Many alternatives fall short by a wide margin: ELEA is 3 hours, CHiME-6/DiPCo is 7 hours, HUT is 8 hours.

**4. Paired IHM and SDM channels provide real clean/noisy pairs.**
Each session was recorded simultaneously on close-talking **IHM (Individual Headset Microphone)** and far-field **SDM (Single Distant Microphone)** channels. This gives naturally paired clean vs. degraded audio without synthetic noise injection, exactly what is needed for supervised enhancement evaluation.

**5. Every speaker gets real airtime.**
Exploratory analysis confirmed that all speakers in a given AMI session are well-represented in the recordings. This matters for speaker segmentation and diarization, since the pipeline needs enough data per voice to build reliable speaker models for turn-taking analysis and communication pattern studies.

**6. Richly annotated.**
The corpus comes with transcripts, dialog acts, speaker diarization labels, and topic segmentation. All of these feed into the NLP and communication analysis stages of the full project.

**7. No access barriers.**
Freely available under CC BY 4.0 on HuggingFace. No registration, no agreements.

### Why Other Datasets Fall Short

| Dataset | Key Limitation |
|---------|---------------|
| NOTSOFAR-1 | Audio-only; requires HuggingFace token; no structured roleplay |
| ICSI Meeting Corpus | Audio-only; requires LDC registration; unstructured research meetings |
| VoxConverse | Audio-only; TV panels and debates, not team task environments |
| AISHELL-4 | Mandarin only, does not match TRIP Lab's English context |
| CANDOR | Only 2-speaker dyadic conversations, too few participants |
| ELEA | Only 3 hours, too small for robust model training |
| LibriCSS | Synthetic re-played audiobook speech, not natural conversation |
| CHiME-6/DiPCo | Informal dinner parties; restricted CC BY-NC license |
| VOICES | Scripted far-field speech with 1 to 2 speakers, not team communication |

### How AMI Will Be Used

The **SDM channel** serves as the degraded input for the enhancement pipeline. It captures room reverberation, background noise, and distant-microphone effects naturally, matching what would come out of a simulator room recording. The **IHM channel** serves as the clean reference for computing PESQ, STOI, and SNR. In the full project, the paired recordings will support training and benchmarking of enhancement models, while the annotation layer feeds into communication analysis for studying coordination, distraction, and group dynamics.

---

## Overview

This repository contains the test submission for the GSoC 2026 project, covering two objectives:

1. **Database Identification and Evaluation** - Surveying open-access team communication datasets, selecting the best-fit resource, and running exploratory analysis to back up the choice.
2. **Audio Enhancement Algorithm and Evaluation** - Building a hybrid enhancement pipeline on real sample data and measuring its impact with standard speech quality metrics.

Both objectives are in separate Jupyter notebooks, each runnable top-to-bottom on a clean kernel restart.

---

## Repository Layout

```
HumanAI_P2/
├── Notebook-1 Database Identification.ipynb     <- Dataset survey + AMI exploratory analysis
├── Notebook-2-Audio Enhancement.ipynb            <- Hybrid enhancement pipeline + evaluation
├── requirements.txt                              <- All Python dependencies
├── README.md
├── data/
│   ├── sample_A_IHM_reference.wav               <- Clean reference samples (IHM headset)
│   ├── sample_A_SDM_noisy.wav                    <- Noisy input samples (SDM far-field)
│   ├── sample_B_IHM_reference.wav
│   ├── sample_B_SDM_noisy.wav
│   ├── sample_C_IHM_reference.wav
│   ├── sample_C_SDM_noisy.wav
│   ├── sample_IHM_5min.wav                       <- 5-min clean reference
│   └── sample_SDM_5min.wav                       <- 5-min noisy input
├── enhanced/
│   ├── sample_A_enhanced.wav                     <- Enhanced output (sample A)
│   ├── sample_B_enhanced.wav                     <- Enhanced output (sample B)
│   ├── sample_C_enhanced.wav                     <- Enhanced output (sample C)
│   ├── sample_enhanced.wav                       <- Enhanced output (5-min)
│   └── stage_metrics.csv                         <- Per-stage metric scores
└── figures/
    ├── audio_quality_report.png                  <- Overall quality report
    ├── fig1_dataset_comparison.png               <- Dataset comparison bar chart
    ├── fig1_dataset_radar.png                    <- Radar chart across scoring dimensions
    ├── fig1_stage_spectrograms.png               <- Multi-stage spectrogram strips
    ├── fig2_eda.png                              <- EDA distribution grid
    ├── fig2_metric_progression.png               <- Metric progression across stages
    ├── fig3_before_after.png                     <- Before/after waveform comparison
    ├── fig3_mfcc.png                             <- MFCC feature comparison
    └── fig4_psd.png                              <- Power spectral density analysis
```

---

## Quick Start

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional neural stage (recommended):
pip install deepfilternet

jupyter notebook
# Run Notebook 1 first, then Notebook 2, top to bottom, all cells
```

> **Note:** The first run of Notebook 1 downloads the AMI corpus from HuggingFace (~2 to 3 GB, cached locally after that).

---

## Notebook 1: Database Identification and Evaluation

### Approach

11 open-access team communication datasets were scored across four weighted dimensions: **Task Fit** (x3), **Ease of Access** (x2), **Acoustic Quality Variety** (x2), and **Annotation Richness** (x1.5). The weights reflect what matters most for the TRIP Lab's specific use case, It were given by me, but by all means, the AMI Meeting Corpus outperforms every other dataset and that's why is the best choice for our project.

| Dataset | Hours | Speakers | Modality | Task Type | Score |
|---------|-------|----------|----------|-----------|-------|
| **AMI Meeting Corpus** | 100 | 3-5 | Audio + Video | Design team roleplay | **95.3** |
| NOTSOFAR-1 (Microsoft) | 28 | 4-8 | Audio | Real office meetings | 87.1 |
| ICSI Meeting Corpus | 72 | 3-8 | Audio | Research group meetings | 70.6 |
| VoxConverse | 64 | 2-20+ | Audio | TV panels, debates | 68.2 |
| AISHELL-4 | 120 | 4-8 | Audio | Conference meetings | 64.7 |
| ELEA Corpus | 3 | 3-4 | Audio + Video | Small group decision-making | 64.7 |
| VOICES Corpus | 120 | 1-2 | Audio | Scripted far-field speech | 54.1 |
| Others (HUT, CANDOR, LibriCSS, CHiME-6) | - | - | - | - | <51 |

### Exploratory Analysis

The notebook covers:
- A radar chart comparing the top datasets across all scoring dimensions
- An 8-panel distribution grid showing key audio features (RMS energy, zero crossing rate, spectral centroid) across IHM vs. SDM conditions
- Waveform and spectrogram comparisons for both channels
- MFCC grids showing how noise distorts the speech features ASR models rely on
- A cross-dataset feature comparison confirming the acoustic gap between close-talk and far-field recordings
- Per-speaker speaking time distribution confirming balanced representation across AMI sessions

---

## Notebook 2: Audio Enhancement Pipeline

### Approach

The goal was a multi-stage pipeline that progressively improves far-field team communication audio for downstream transcription. Rather than a single model, I went with a **hybrid classical plus optional neural architecture**. This keeps the pipeline interpretable, lightweight, and usable in environments without GPU access.

### Pipeline Architecture

```
Raw SDM Audio
  |
  v
[S1] Pre-process -------- Resample to 16kHz, amplitude normalization
  |
  v
[S2] Spectral Gate ------- noisereduce (2-pass) for stationary noise
  |
  v
[S3] Dereverberation ----- STFT-domain Wiener filter + bandpass
  |
  v
[S4] Enhancement ---------  Multi-band spectral subtraction (classical)
  |                         OR DeepFilterNet2 (neural, if installed)
  v
[S5] Post-process --------  LUFS normalization (-16 dBFS) + clip guard
  |
  v
Enhanced Audio
```

### Design Rationale

| Aspect | Classical Stages (S2-S3) | Neural Stage (S4, if available) |
|--------|--------------------------|--------------------------------|
| **Interpretability** | High - explicit frequency operations | Low - black-box |
| **Latency** | <5 ms | <1 ms (causal mode) |
| **Dependencies** | scipy, noisereduce | deepfilternet / pedalboard |
| **Best for** | Stationary noise, reverb, EQ | Residual full-band enhancement |

Each classical stage targets a specific type of degradation. `noisereduce` handles stationary background noise, the Wiener filter addresses reverberation, and multi-band spectral subtraction cleans up residual distortion. If DeepFilterNet2 is available it replaces Stage 4; otherwise the classical fallback keeps the pipeline functional.

### Evaluation

**3 samples x 2 conditions (IHM reference vs SDM) x 5 stages x 4 metrics**

Metrics tracked at every stage:
- **SNR (dB)** - Signal-to-noise ratio vs. the IHM reference
- **Speech-band SNR** - Energy ratio in the 300 to 3400 Hz band, the range most critical for intelligibility
- **PESQ** (ITU-T P.862 wideband, -0.5 to 4.5) - Perceptual quality score
- **STOI** (0 to 1) - Short-time objective intelligibility

### Results

The pipeline consistently improved all metrics from raw SDM input to final output. The biggest gains came from spectral gating (S2), which addressed dominant stationary noise, followed by measurable improvements from dereverberation (S3). Post-processing normalization (S5) brought loudness levels into a consistent range for downstream ASR.

Outputs include metric progression curves, multi-stage spectrogram strips, before/after waveform comparisons, per-stage improvement bar charts, and power spectral density plots.

---

## Alternative Approaches

A few directions worth exploring in the full GSoC project:

**For audio enhancement:**
- End-to-end neural models like [DCCRN](https://arxiv.org/abs/2008.00264) or [Conv-TasNet](https://arxiv.org/abs/1809.07454) could replace the multi-stage pipeline, mapping noisy audio to clean speech in a single pass. They tend to outperform classical methods on non-stationary noise but need GPU training on paired data.
- Speaker separation using models like [SepFormer](https://arxiv.org/abs/2010.13154) applied before enhancement could help in segments where multiple speakers overlap.
- Adaptive noise estimation that updates its profile in real time would handle dynamic acoustic conditions better than a fixed spectral gate.

**For dataset utilization:**
- Fine-tuning enhancement models on a small amount of TRIP Lab data after pre-training on AMI could improve performance on the target domain.
- Convolving clean IHM recordings with measured room impulse responses and adding recorded ambient noise could expand the training set without additional recording sessions.

---

## Reproducibility

- `numpy.random.seed(42)` set globally in both notebooks
- All parameters are exposed as function arguments with documented defaults
- Both notebooks run top-to-bottom on a clean kernel restart
- All outputs (WAV, CSV, PNG) are saved automatically to `data/`, `enhanced/`, and `figures/`

---

## Dependencies

See `requirements.txt`. Key packages:

| Purpose | Package |
|---------|---------|
| Audio I/O and analysis | `librosa`, `soundfile`, `scipy` |
| Noise reduction | `noisereduce` |
| Neural enhancement (optional) | `deepfilternet` |
| Quality metrics | `pesq`, `pystoi` |
| Loudness normalization | `pyloudnorm` |
| Dataset access | `datasets`, `huggingface_hub` |
| Visualization | `matplotlib`, `seaborn` |

---

## References

- Carletta, J. et al. (2007). *The AMI Meeting Corpus.* CC BY 4.0
- Vinnikov, A. et al. (2024). *NOTSOFAR-1.* Apache 2.0
- Schröter, H. et al. (2022). *DeepFilterNet2.* MIT License
- ITU-T P.862 (PESQ) · Taal et al. (2011) (STOI)