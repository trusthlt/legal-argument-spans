"""Score the prediction CSVs (listed in config.ini [PREDICTIONS]) and report, per
configuration, the argument component identification (ACI), the argument
component classification (ACC, macro/weighted F1) and the output-format
validity, using the relaxed span-matching metric from the paper."""
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from argparse import ArgumentParser
import pandas as pd
import os
import configparser

from utils import find_positions, get_argument_component_identification

def evaluate(name, path, threshold=0.2):
    """Score one prediction CSV against the gold in-text annotations.

    Each row holds the gold annotation (`target`) and the model output (`text`). ACI is
    scored per token (argument / non-argument), ACC per gold argument: a predicted span
    matches a gold span if both boundaries are closer than `threshold` (relative to the
    gold span length). Outputs that do not reproduce the input text are counted as invalid.

    Args:
        name (str): Name of the configuration as listed in config.ini [PREDICTIONS].
        path (str): Path to the prediction CSV with the columns `text`/`target`.
        threshold (float, optional): Relaxed span-acceptance threshold. Defaults to 0.2.

    Returns:
        tuple[str, str, tuple[int, int, float]]: The ACC classification report, the ACI
        classification report, and (invalid segments, all segments, output-format validity).
    """
    already_predicted = pd.read_csv(path, index_col=0, na_filter=False)
    all_predicted, all_gold = [], []
    all_predicted_identification, all_gold_identification = [], []
    all_predicted_segments, all_gold_segments = [], []
    bracket_thres = threshold # ie 10% on each side
    for _, e in tqdm(already_predicted.iterrows()):
        target = e['target']
        predicted = e['text']

        p_text = predicted

        # token spans and labels of the gold and the predicted annotations
        t_labels, t_positions = [l[1] for l in find_positions(target, target)], [l[0] for l in find_positions(target, target)]
        p_labels, p_positions = [l[1] for l in find_positions(predicted, target)], [l[0] for l in find_positions(predicted, target)]

        # token sequences without annotation tokens, to check that the text was reproduced
        tokenized, _ = get_argument_component_identification(target)
        tokenized_p, _ = get_argument_component_identification(p_text)

        # ACI: label every token as argument / non-argument
        t_aci = ['non-argument' for t in tokenized]
        for pos in t_positions:
            for p in range(pos[0], pos[1]):
                t_aci[p] = 'argument'
        p_aci = ['non-argument' for t in tokenized]
        for pos in p_positions:
            for p in range(pos[0], pos[1]):
                # a predicted span beyond the paragraph makes the output invalid
                if (pos[0] > len(p_aci)) or (pos[1] > len(p_aci)):
                    p_aci = ['invalid' for t in tokenized]
                    break
                p_aci[p] = 'argument'
        if tokenized != tokenized_p:
            # without labels there was a difference in input and output
            p_aci = ['invalid' for t in tokenized]

        assert len(p_aci) == len(t_aci), print(f"p: {len(p_aci)}, t: {len(t_aci)}\n{t_positions}\n{p_positions}\n{target}\n{predicted}")
        assert len(p_aci) == len(tokenized)

        # flatten for classification report per token
        all_predicted_identification += p_aci
        all_gold_identification += t_aci
        # append for segment wise stats
        all_predicted_segments.append(p_aci)
        all_gold_segments.append(t_aci)

        # ACC: match every gold span to a predicted span, exact position first
        match_positions = {k: -1 for k, _ in enumerate(t_positions)}
        for l, tl in enumerate(t_labels):
            pos = t_positions[l]

            if pos in p_positions:
                match_idx = p_positions.index(pos)
                match_positions[l] = match_idx
            else:
                # threshold position match
                for q, ppos in enumerate(p_positions):
                    ratio = abs(pos[1] - pos[0])
                    if (abs(pos[0] - ppos[0])/ratio < bracket_thres) & (abs(pos[1] - ppos[1])/ratio < bracket_thres):
                        match_idx = q
                        match_positions[l] = match_idx
                        break
        
            
        assert len(p_positions) == len(p_labels), f"{p_text}\n{p_labels}\n{p_positions}"
        # predicted label per gold span, 'umatched' if no predicted span matched
        p = [p_labels[match_positions[t]] if match_positions[t] != -1 else 'umatched' for t, _ in enumerate(t_positions)]
        g = t_labels
        assert len(p) == len(g), "sequences of unequal length"
        all_predicted += p
        all_gold += g

    # ACC report (per gold span)
    cr1 = classification_report(all_gold, all_predicted, zero_division=0)
    f1 = f1_score(all_gold, all_predicted, average='macro', zero_division=0)

    print(cr1)
    print(f"macro-average f1-score: {f1}")

    # ACI report (per token)
    cr2 = classification_report(all_gold_identification, all_predicted_identification, zero_division=0)
    f1 = f1_score(all_gold_identification, all_predicted_identification, average='macro', zero_division=0)

    print(cr2)
    # output-format validity: share of paragraphs without invalid output
    invalid_segments = ['invalid' for c in all_predicted_segments if 'invalid' in c]
    print(f"invalid segments: {len(invalid_segments)}/{len(all_predicted_segments)} - {len(invalid_segments)/len(all_gold_segments)}")

    return cr1, cr2, (len(invalid_segments), len(all_predicted_segments), 1-(len(invalid_segments)/len(all_predicted_segments)))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='relaxed span-acceptance threshold as a fraction of the '
                             'argument length (0.0/0.1/0.2 == 0%%/10%%/20%% in the paper; '
                             'the paper reports 10%% as the main setting)')
    parser.add_argument('--report_dir', default='outputs/reports', type=str)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    # prediction CSVs to score, as listed in config.ini [PREDICTIONS]
    predictions_dict = [{'name': name, 'path': path} for name, path in config['PREDICTIONS'].items()]

    write = True
    # relaxed span-acceptance threshold (see --threshold / the paper)
    thres = args.threshold
    if write:
        os.makedirs(args.report_dir, exist_ok=True)
        # one report per threshold, e.g. all_scores_report_thres_10.txt
        report_file = os.path.join(args.report_dir, f'all_scores_report_thres_{int(thres*100)}.txt')
        with open(report_file, 'w') as f:
            for i, p in tqdm(enumerate(predictions_dict)):
                try:
                    cr_label, cr_binary, segs = evaluate(p['name'], p['path'], threshold=thres)
                except FileNotFoundError:
                    print(f"file not found, skipping: {p['name']} ({p['path']})")
                    continue
                f.write('\n')
                f.write('-'*20)
                f.write(p['name'])
                f.write('\n')
                f.write(cr_label)
                f.write(cr_binary)
                f.write(f"\ninvalid segments: {segs[0]}/{segs[1]} -> {segs[2]}\n")
    else:
        for i, p in tqdm(enumerate(predictions_dict)):
            try:
                cr_label, cr_binary, segs = evaluate(p['name'], p['path'], threshold=thres)
            except FileNotFoundError:
                print(f"file not found, skipping: {p['name']} ({p['path']})")

    