from .io.box_filtering import filter_boxes
from .metrics.coco_eval import evaluate_detection


def evaluate_list(result_boxes_list,
                  gt_boxes_list,
                  height: int,
                  width: int,
                  camera: str = 'gen1',
                  apply_bbox_filters: bool = True,
                  downsampled_by_2: bool = False,
                  return_aps: bool = True):
    assert camera in {'gen1', 'gen4'}

    if camera == 'gen1':
        classes = ("car", "pedestrian")
    elif camera == 'gen4':
        classes = ("pedestrian", "two-wheeler", "car")
    else:
        raise NotImplementedError
    apply_bbox_filters=False
    if apply_bbox_filters:
        # Default values taken from: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/0393adea2bf22d833893c8cb1d986fcbe4e6f82d/src/psee_evaluator.py#L23-L24
        min_box_diag = 60 if camera == 'gen4' else 30
        # In the supplementary mat, they say that min_box_side is 20 for gen4.
        min_box_side = 20 if camera == 'gen4' else 10
        if downsampled_by_2:
            assert min_box_diag % 2 == 0
            min_box_diag //= 2
            assert min_box_side % 2 == 0
            min_box_side //= 2

        half_sec_us = int(1e3)
        filter_boxes_fn = lambda x: filter_boxes(x, half_sec_us, min_box_diag, min_box_side)

        gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        #gt_boxes_list = list(gt_boxes_list)
        #for gt_box in gt_boxes_list:
            #gt_box = filter_boxes(gt_box,half_sec_us, min_box_diag, min_box_side)
        # NOTE: We also filter the prediction to follow the prophesee protocol of evaluation.
        result_boxes_list = map(filter_boxes_fn, result_boxes_list)

    return evaluate_detection(gt_boxes_list, result_boxes_list,
                              height=height, width=width,
                              classes=classes, return_aps=return_aps)
#def filter_boxes_fn(x, half_sec_us, min_box_diag, min_box_side):
def filter_boxes(boxes, skip_ts=int(5e5), min_box_diag=60, min_box_side=20):
    """Filters boxes according to the paper rule. 

    To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
    To note: we assume the initial time of the video is always 0

    Args:
        boxes (np.ndarray): structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence'] 
        (example BBOX_DTYPE is provided in src/box_loading.py)

    Returns:
        boxes: filtered boxes
    """
    ts = boxes['t']
    width = boxes['w']
    height = boxes['h']
    diag_square = width ** 2 + height ** 2
    mask = (ts > skip_ts) * (diag_square >= min_box_diag ** 2) * (width >= min_box_side) * (height >= min_box_side)
    return boxes[mask]
