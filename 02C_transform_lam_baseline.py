"""Convert the original LAM RoBERTa baseline predictions from the token-level BIO format
into the in-text span-annotation format used by our evaluation, and write the baseline
CSV consumed by 03_evaluate_predictions.py.

The input is external to this pipeline and not shipped: the raw RoBERTa (Legal-RoBERTa-15k)
BIO predictions (TSV with the columns `Tokens`, `Labels`, `Predictions`, each a stringified
list), from the original LAM:ECHR release (https://github.com/trusthlt/mining-legal-arguments),
specifically `predictions/roberta-large-fp-15000_checkpoint-143334_roberta-argType.csv`.

The transformed output has the columns `text`/`target`.
"""
import os
from argparse import ArgumentParser
from ast import literal_eval

import pandas as pd
from tqdm import tqdm

from utils import LABEL_MAPPING

# Label ordering of the original LAM RoBERTa baseline: 15 B- classes, 15 I- classes and
# `O` at index 30. The in-text tokens below match those configured in config.ini.
ID2LABEL = [
    'B-Distinguishing',
    'B-Einschätzungsspielraum',
    'B-Entscheidung des EGMR',
    'B-Konsens der prozessualen Parteien',
    'B-Overruling',
    'B-Rechtsvergleichung',
    'B-Sinn & Zweck Auslegung',
    'B-Subsumtion',
    'B-Systematische Auslegung',
    'B-Verhältnismäßigkeitsprüfung – Angemessenheit',
    'B-Verhältnismäßigkeitsprüfung – Geeignetheit',
    'B-Verhältnismäßigkeitsprüfung – Legitimer Zweck',
    'B-Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
    'B-Vorherige Rechtsprechung des EGMR',
    'B-Wortlaut Auslegung',
    'I-Distinguishing',
    'I-Einschätzungsspielraum',
    'I-Entscheidung des EGMR',
    'I-Konsens der prozessualen Parteien',
    'I-Overruling',
    'I-Rechtsvergleichung',
    'I-Sinn & Zweck Auslegung',
    'I-Subsumtion',
    'I-Systematische Auslegung',
    'I-Verhältnismäßigkeitsprüfung – Angemessenheit',
    'I-Verhältnismäßigkeitsprüfung – Geeignetheit',
    'I-Verhältnismäßigkeitsprüfung – Legitimer Zweck',
    'I-Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
    'I-Vorherige Rechtsprechung des EGMR',
    'I-Wortlaut Auslegung',
    'O',
]


def transform_sequence(tokens, labels, labelmap):
    """Render a BIO-tagged token sequence as in-text span annotations.

    Consecutive tokens sharing a label are wrapped as
    `[begin_of_annotation] ... [end_of_annotation] <label> [end_of_label]`; a new `B-`
    tag opens a fresh span. The surface text is reconstructed with simple detokenization
    spacing (no leading space before punctuation).
    """
    O_token = 30
    text = ''
    last_lab = 'O'
    full_last_lab = 'O'
    started_bracket = False

    # the baseline occasionally emits fewer labels than tokens; pad to the token length
    if len(tokens) != len(labels):
        if list(set(labels)) == [30]:
            labels = [30] * len(tokens)
        else:
            labels = tuple(list(labels) + [labels[-1]] * (len(tokens) - len(labels)))

    for pos, word in enumerate(tokens):
        current_lab = LABEL_MAPPING[labelmap[labels[pos]][2:]] if labels[pos] != O_token else 'O'
        full_lab = labelmap[labels[pos]] if labels[pos] != O_token else 'O'
        if last_lab != current_lab:
            if last_lab != 'O':
                if current_lab == 'O':
                    text += f' [end_of_annotation] {last_lab} [end_of_label]'
                    started_bracket = False
                else:
                    text += f' [end_of_annotation] {last_lab} [end_of_label] [begin_of_annotation]'
                    started_bracket = True
            else:
                text += f"{' ' if pos != 0 else ''}[begin_of_annotation]"
                started_bracket = True
        elif full_last_lab != full_lab and full_lab[:2] == 'B-':
            # a new B- span starts right after another span of the same base label
            text += f' [end_of_annotation] {last_lab} [end_of_label] [begin_of_annotation]'
            started_bracket = True

        last_lab = current_lab
        full_last_lab = full_lab

        if (word not in [',', '.', '“', ':', ')', '(', '[', ']']) & (pos != 0):
            text += ' '
        text += word

    if started_bracket:
        text += f' [end_of_annotation] {last_lab} [end_of_label]'
    return text


def return_marked_arguments(tokens, gt, pred, labelmap):
    """Render both the gold and the predicted BIO sequences as in-text annotations."""
    gold_text = transform_sequence(tokens, gt, labelmap)
    pred_text = transform_sequence(tokens, pred, labelmap)
    return gold_text, pred_text


def read_and_transform_original(predictions_file, output_file):
    """Transform the raw BIO baseline predictions into the evaluation format.

    Re-renders the gold and predicted BIO sequences as in-text annotations and writes a
    CSV with the columns `text`/`target`.
    """
    df = pd.read_csv(predictions_file, sep='\t', encoding='utf-8')
    df['Labels'] = df['Labels'].map(lambda x: tuple(literal_eval(x)))
    df['Predictions'] = df['Predictions'].map(lambda x: tuple(literal_eval(x)))
    df['Tokens'] = df['Tokens'].map(lambda x: tuple(literal_eval(x)))
    print(f"loaded {df.shape[0]} original predictions")

    transformed = []
    for _, row in tqdm(df.iterrows()):
        marked_gold, marked_pred = return_marked_arguments(row['Tokens'], row['Labels'], row['Predictions'], ID2LABEL)
        transformed.append({'text': marked_pred, 'target': marked_gold})

    df_transformed = pd.DataFrame(transformed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_transformed.to_csv(output_file, encoding='utf-8')
    print(f"saved {len(df_transformed)} transformed baseline predictions to {output_file}")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--predictions', default='lam_predictions/original_predictions_roberta_fp_15k.csv', type=str,
                        help='raw RoBERTa BIO predictions (TSV: Tokens/Labels/Predictions)')
    parser.add_argument('--output', default='outputs/paper/predictions/lam_baseline_predictions.csv', type=str,
                        help='destination CSV read by 03_evaluate_predictions.py')
    args = parser.parse_args()

    read_and_transform_original(args.predictions, args.output)
