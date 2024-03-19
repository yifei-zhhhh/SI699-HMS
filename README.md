# SI699-HMS

## Introduction

Electroencephalography (EEG) is a non-invasive method of monitoring and recording electrical activity in the brain, which plays a crucial role in diagnosing and treating various brain-related disorders, particularly in critically ill patients. However, the manual interpretation of EEG data remains a major bottleneck in neurocritical care, as it is time-consuming, expensive, and prone to fatigue-related errors and inter-rater reliability issues. To address these challenges, there is a pressing need to develop automated methods for EEG analysis. The objective of this research is to develop a robust model trained on EEG signals to automatically detect and classify seizures and other types of harmful brain activity, aiming to assist doctors and brain researchers in providing faster and more accurate diagnoses and treatments. This work has significant implications for improving patient care in neurocritical settings, advancing epilepsy research, and supporting the development of new therapeutic interventions.

## Data
The dataset consists of electroencephalography (EEG) recordings labeled with different types of harmful brain activity patterns, including seizures, periodic discharges (generalized and lateralized), and rhythmic delta activity (generalized and lateralized). Even expert annotators often disagree on the correct classifications, highlighting the complexity of this task.

The dataset comprises the following files:
- train.csv and test.csv: Metadata files containing information about the EEG and spectrogram samples, patient identifiers, and for the training set, the expert consensus labels and vote counts from annotators.
- sample_submission.csv: A file showing the required format for submissions, which should contain the predicted probabilities for each brain activity class.
- train_eegs/ and test_eegs/: Folders containing the raw EEG data recordings, sampled at a frequency of 200 Hz.
- train_spectrograms/ and test_spectrograms/: Folders containing spectrogram image representations of the EEG data.
- example_figures/: A folder with larger versions of the example images used in the overview section.

For train.csv, the detailed information is given:
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

## Model Prototype
<img width="1324" alt="image" src="https://github.com/yifei-zhhhh/SI699-HMS/assets/119919755/cc0b4c6c-b077-4f2e-bc53-68de6fd1481e">

## How to compile
The data size is quite large (26.4GB) and also accelerator is required to run this project, so we highly recommend you to run the code in Kaggle, and you can directly load data inside from [the hms page](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview).
