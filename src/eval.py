import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from src.classifier import Classifier



def run_metrics(y_true, y_pred):
    metric_dict = {}
    metric_dict["accuracy"] = accuracy_score(y_true, y_pred)
    metric_dict["recall"] = recall_score(y_true, y_pred, zero_division=1)
    metric_dict["f1"] = f1_score(y_true, y_pred, zero_division=1)
    metric_dict["cm"] = confusion_matrix(y_true, y_pred)

    return metric_dict


def print_metric_dict(metric_dict, rounding_decimals=3):
    for metric in metric_dict.keys():
        if metric != "cm":
            print(f'{metric}: {round(metric_dict[metric], rounding_decimals)}')


def eval_model(args):

    classifier = Classifier(args.metamap_url)

    # load test data
    test_data_df = pd.read_csv(args.csv_file_path).iloc[:]
    test_data_df = test_data_df.fillna("")

    # run classifier
    print("Running model")
    tqdm.pandas()

    test_data_df['model_prediction'] = test_data_df.progress_apply(
        lambda row: classifier.classify(row[args.patient_record_column], args.symptom_cui), axis=1)

    # set Positive and Negative to true and blank rows to false, so that the task of classifying if
    # a symptom is mentioned can be treated as a binary classification problem
    test_data_df['symptom_mention'] = test_data_df[args.label_column] != args.not_mentioned_label
    test_data_df['symptom_mention_prediction'] = test_data_df['model_prediction'] != args.not_mentioned_label

    # filter only records where a symptom was detected
    symptom_test_data_df = test_data_df[test_data_df['symptom_mention_prediction'] == True].copy()

    # set Positive to true and Negative to false
    symptom_test_data_df[args.label_column] = symptom_test_data_df[args.label_column] == args.pos_label
    symptom_test_data_df['model_prediction'] = symptom_test_data_df['model_prediction'] == args.pos_label

    print("Metrics for classification of whether a symptom is mentioned:")
    print_metric_dict(run_metrics(test_data_df['symptom_mention'], test_data_df['symptom_mention_prediction']))
    print("-" * 50)
    print("Metrics for classification of whether symptom mention is positive/negative:")
    print_metric_dict(run_metrics(symptom_test_data_df[args.label_column], symptom_test_data_df['model_prediction']))

