import json
import os
import sys

import numpy as np

from evaluation.eval_detection import ANETdetection


def AnetEvaluate(root, subset, epoch):
    ground_truth_filename = os.path.join(root, 'val.json')
    prediction_filename = os.path.join(root, 'predictions', f'prediction_{epoch}.json')
    with open(os.path.join(root, 'anet.json')) as f:
        anns = json.load(f)

    valid_gts = os.listdir(os.path.join(root, 'merge'))
    valid_gts = [name.split('.')[0] for name in valid_gts]

    val_anns = {'database': {}}
    for k, v in anns['database'].items():
        if v['subset'] == subset and k in valid_gts:
            val_anns['database'][k] = v
    
    with open(ground_truth_filename, 'w') as f:
        json.dump(val_anns, f)

    anet_eval = ANETdetection(
        ground_truth_filename, prediction_filename, 
        check_status=False, subset=subset,
        tiou_thresholds=np.linspace(0.5, 0.95, 10), verbose=True
    )
    anet_eval.evaluate()


if __name__ == "__main__":
    AnetEvaluate('../data/ActivityNet/', 'validation', sys.argv[1])