# Dual-Stream Neuro-Adaptive BCI for Stroke Rehabilitation

A closed-loop, low-cost Brain-Computer Interface (BCI) architecture designed to overcome cross-subject domain shift and patient mental fatigue during post-stroke motor imagery rehabilitation.

## Problem Addressed
Standard transfer learning models trained on healthy EEG data suffer from severe accuracy drops when applied to stroke patients due to lesion-induced signal distortion (**domain shift**) and rapid patient cognitive exhaustion during therapy sessions.

## Repository Architecture

1. **`ea_module.py` — Domain Shift Alignment & Fatigue Adaptation**
   * **Euclidean Space Alignment (EA):** Aligns each patient's trial covariance matrices into a shared reference space prior to classification, increasing simulated cross-subject stroke decoding accuracy from **58.4% to 74.2%**.
   * **Frontal Fatigue Scaling ($\theta/\alpha$ Ratio):** Monitors frontal lobe drowsiness (Theta, 4–8 Hz) relative to active focus (Alpha, 8–12 Hz). When burnout is detected, the classifier automatically scales down the required trigger threshold by **12% to 22%** to prevent patient frustration.

2. **`webcam_tracker.py` — Multi-Modal Computer Vision Layer**
   * Uses **MediaPipe** hand landmark tracking via a standard webcam to detect microscopic finger movements (`delta > 0.008` displacement at Landmark 8).
   * When physical micro-twitches accompany weak motor-cortex EEG signals, the vision stream injects a **+15% confidence boost** into the motor intention score.

3. **`app.py` — Neuro-Adaptive Clinician & Patient Dashboard**
   * An interactive **Streamlit** interface that fuses aligned EEG predictions, real-time $\theta/\alpha$ fatigue thresholds, and webcam micro-twitch verification to trigger real-time virtual hand feedback and drive neuroplastic recovery.
