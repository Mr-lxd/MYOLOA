import warnings

warnings.filterwarnings('ignore')
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from tidecv import TIDE, datasets


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str,
                        default=r'D:\navigationData\annotations\1_cleaned.json')  # Default path seems incomplete
    parser.add_argument('--pred_json', type=str, default=r'D:\navigationData\annotations\2_corrected.json',
                        help='')  # Help message seems incomplete

    return parser.parse_known_args()[0]  # Reads known args, ignoring others


if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    # tide = TIDE()
    # tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    # tide.summarize()
    # tide.plot(out_dir='result')