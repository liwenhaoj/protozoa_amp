# protozoa_amp
bert and cnn model for protozoa amp predict
# Antimicrobial Peptide Prediction Model

This project implements two deep learning models (BERT and CNN) for antimicrobial peptide (AMP) prediction. It provides a complete pipeline for data processing, model training, evaluation, and prediction.

## Project Structure

```
.
├── 1_bert_make_model.py    # BERT model training script
├── 2_bert_predict.py       # BERT model prediction script
├── 3_cnn_make_model.py     # CNN model training script
├── 4_cnn_predict.py        # CNN model prediction script
├── AMPS_1.fasta            # Sequence file to be predicted
├── requirment.txt          # Project dependencies
├── TrainingAMP_3.csv       # CNN and Bert model training input file
└── README.md               # Project documentation (Chinese)

```

## Requirements

- Python 3.12
- CUDA support (recommended for GPU acceleration)
- Other dependencies listed in `requirment.txt`

## Installation

1. Clone the repository to your local machine
2. Install dependencies:
   ```bash
   pip install -r requirments.txt
   ```

## Model Description

### BERT Model
- Fine-tuned from pre-trained BERT model
- Custom tokenizer for amino acid sequences
- Maximum sequence length: 50 amino acids
- Binary classification output (AMP or non-AMP)

### CNN Model
- Improved convolutional neural network architecture
- Includes embedding, convolutional, and fully connected layers
- Maximum sequence length: 100 amino acids
- Includes data augmentation functionality
- Binary classification output (AMP or non-AMP)

## Usage

### Model Training

1. BERT Model Training:
   ```bash
   python 1_bert_make_model.py
   ```
   - Input file: `TrainingAMP_3.csv`
   - Output files: `bert_peptide_classification_model.pth_*`

2. CNN Model Training:
   ```bash
   python 3_cnn_make_model.py
   ```
   - Input file: `TrainingAMP_3.csv`
   - Output files:
     - `best_amino_acid_cnn.pth` (best model)
     - `amino_acid_cnn_final.pth` (final model)
     - Various evaluation plots (PDF format)

### Sequence Prediction

1. BERT Model Prediction:
   ```bash
   python 2_bert_predict.py
   ```
   - Input file: `AMPS_1.fasta`
   - Output file: `bert_predict_*.txt`

2. CNN Model Prediction:
   ```bash
   python 4_cnn_predict.py
   ```
   - Input file: `AMPS_1.fasta`
   - Output file: `predictions.txt`

## Output Format

### Prediction Results Format
- BERT Model Output:
  ```
  Sequence_ID    Sequence    Probability
  ```

- CNN Model Output:
  ```
  Sequence_ID    Sequence    Probability    Prediction
  ```

### Evaluation Metrics
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- F1 Score
- Accuracy

## Important Notes

1. Ensure input sequence files are in correct FASTA format
2. GPU is recommended for model training
3. Model training may take significant time
4. Use correct model files for prediction

## Model Performance

Both models provide comprehensive evaluation metrics including:
- Accuracy
- F1 Score
- Area Under ROC Curve (AUC)
- Precision-Recall Curve

Detailed performance metrics can be found in the plots generated during training.

## License

Please comply with the license requirements when using this project.

## Contact

For questions or suggestions, please submit an Issue or Pull Request.
