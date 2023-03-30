# Overview

I integrated MetaMap (a tool that tags symtoms with the UMLS medical taxonomy) and a finetuned version of BERT to classify patient records.
Metamap tags whether a symptom (specified by a UMLS id) is present in the record. The finetuned BERT model determines whether 
the mention is positive or negative. 

The patient/other and current/history fields are not currently supported, but can be implemented by finetuning BERT given a hand labeled dataset, as shown in this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7772072/ (their model is publically available, but I have yet to implement it in my program). 

This program uses an external api for MetaMap to make it simple to set up, but MetaMap can be ran locally as well. 
# Use

Install dependencies and run `python -m spacy download en_core_web_sm`.

To run the model on a test set, run `python main.py --csv_file_path "data/evaluation_set.csv" --symptom_cui "C0003862"`. This should yield the following output:

```
Metrics for classification of whether a symptom is mentioned:
accuracy: 0.979
recall: 0.957
f1: 0.978
--------------------------------------------------
Metrics for classification of whether symptom mention is positive/negative:
accuracy: 0.977
recall: 0.957
f1: 0.978
```


"C0003862" is the id for joint pain.
See this table for symptom_cui for different symptoms:

| Symptom        | cui      |
|----------------|----------|
| joint pain     | C0003862 |
| hypertension   | C0020538 |
| anemia         | C0002871 |
| abdominal pain | C0000737 |
| vomiting       | C0042963 |
| rash           | C0015230 |

# Data

`evaluation_set.csv` - a subset of the joint pain dataset, consisting of all the labeled joint pain patient files and 50 randomly sampled files that did not mention joint pain. Used for testing.

`finetune_set.csv` - the hand labeled data set I made for my experiment with finetuning a language model. See the notebook folder for the finetuning code.




