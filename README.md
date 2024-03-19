# SI699-HMS

## Introduction

Electroencephalography (EEG) is a non-invasive method of monitoring and recording electrical activity in the brain, which plays a crucial role in diagnosing and treating various brain-related disorders, particularly in critically ill patients. However, the manual interpretation of EEG data remains a major bottleneck in neurocritical care, as it is time-consuming, expensive, and prone to fatigue-related errors and inter-rater reliability issues. To address these challenges, there is a pressing need to develop automated methods for EEG analysis. The objective of this research is to develop a robust model trained on EEG signals to automatically detect and classify seizures and other types of harmful brain activity, aiming to assist doctors and brain researchers in providing faster and more accurate diagnoses and treatments. This work has significant implications for improving patient care in neurocritical settings, advancing epilepsy research, and supporting the development of new therapeutic interventions.

## Data

 train.csv
| **Column Name**                 | **Description**                                                                    |
|---------------------------------|------------------------------------------------------------------------------------|
| eeg_id                          | A unique identifier for the entire EEG recording.                                  |
| eeg_sub_id                      | An ID for the specific 50-second long subsample this row's labels apply to.        |
| eeg_label_offset_seconds        | The time between the beginning of the consolidated EEG and this subsample.         |
| spectrogram_id                  | A unique identifier for the entire EEG recording.                                  |
| spectrogram_sub_id              | An ID for the specific 10-minute subsample this row's labels apply to.             |
| spectogram_label_offset_seconds | The time between the beginning of the consolidated spectrogram and this subsample. |
| label_id                        | An ID for this set of labels.                                                      |
| patient_id                      | An ID for the patient who donated the data.                                        |
| expert_consensus                | The consensus annotator label. Provided for convenience only.                      |
