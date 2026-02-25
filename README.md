# BERT-NER-CoNLL2003-Classifier
A high-performance Named Entity Recognition (NER) system built with PyTorch and BERT. This project identifies entities such as People, Organizations, Locations, and Miscellaneous names in text.

📊 Performance
The model was trained on the CoNLL-2003 dataset (Reuters news) and achieved impressive results on the validation set:
<img width="505" height="294" alt="image" src="https://github.com/user-attachments/assets/a51a4639-f7d4-4838-948a-9ee29d818f81" />

🚀 Getting Started
1. Prerequisites
Ensure you have Python installed and install the required libraries:

pip install torch transformers kagglehub seqeval tqdm

2. Project Structure
main.py: The training script.

test.py: Evaluates the model and generates the performance report.

predict.py: Interactive CLI script to test the model with your own sentences.

model.py: Defines the NERModel architecture using bert-base-cased.

dataset_transformation.py: Handles data downloading, parsing, and token alignment.

3. Usage
Training
To train the model from scratch (it will save as ner_model.pth):

python main.py

Evaluation
To see the detailed statistical report (Precision/Recall/F1) on the validation dataset:

python test.py

Inference (Live Testing)
To test the model interactively with your own sentences:

python predict.py

🧠 How it Works
Data: The system automatically downloads the CoNLL-2003 dataset using kagglehub.

Architecture: It uses a pre-trained bert-base-cased Transformer model with a linear classification head on top.

Token Alignment: Since BERT uses sub-word tokenization (WordPiece), the system carefully aligns original labels with the new tokens, ensuring accurate entity detection even for complex words.

IOB Tagging: Supports standard IOB (Inside-Outside-Beginning) format for 9 labels (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).

⚠️ Known Limitations & Observations
1. Temporal Data Gap (2003 vs. Present)
The model was trained on the CoNLL-2003 dataset. As a result, it may struggle with modern entities that did not exist or weren't prominent in 2003 news cycles.

Example: The model might fail to recognize "Tesla" as an Organization or "Elon Musk" as a Person because these specific patterns were not present in the training data. It relies on context, but "unseen" words often default to the O (Outside) tag.

2. Language Sensitivity
English: The model performs at its peak (~94% F1-score) because it uses the bert-base-cased English weights and English training labels.

Polish: Performance is significantly lower. The model does not understand Polish grammar, especially word inflections (e.g., it may recognize "Poznań" but fail on "Poznania").

3. Comparison Examples
Below are the real-world test cases used to verify the model's behavior:

Case A (Domain Match): "Peter Cook from the British Government met with United Nations officials in London."

Result: Perfect detection (matches the 2003 news style).

<img width="242" height="134" alt="image" src="https://github.com/user-attachments/assets/861fcfe2-89b8-4d14-b7ef-f5f249ce61cf" />


Case B (Modern/Out-of-Domain): "Elon Musk announced that Tesla will build a new factory in Berlin."

Result: Potential misclassification due to the temporal gap and casing sensitivity.

<img width="226" height="155" alt="image" src="https://github.com/user-attachments/assets/e1227186-08ed-4810-8b8f-f5e15e92e6f9" />
