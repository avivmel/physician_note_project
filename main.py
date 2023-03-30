import argparse

from src.eval import eval_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', type=str, required=True)  # path to test data
    parser.add_argument('--symptom_cui', type=str, default="C0003862") # the id of the symptom being searched for in the UMLS medical taxonomy
    parser.add_argument('--metamap_url', type=str, default="https://ii.nlm.nih.gov/metamaplite/rest/annotate")  # metamap api used, can be self hosted
    parser.add_argument('--pos_label', type=str, default="Positive")
    parser.add_argument('--not_mentioned_label', type=str, default="")
    parser.add_argument('--patient_record_column', type=str, default="transcription")
    parser.add_argument('--label_column', type=str, default="Positive/ Negative")
    args = parser.parse_args()

    eval_model(args)

# default="/Users/aviv/Desktop/predicta/joint-pain-labeled-csv_patient_only_extracted_symptoms.csv"