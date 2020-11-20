"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def DIoU_NMS(candidates, threshold):
    """
    Distance Intersection over Union(DIoU)
    Non-Maximum Suppression(NMS)

    @param `candidates`:
            [[center_x, center_y, w, h, class_id, propability], ...]
    @param `threshold`: If DIoU is above the threshold, bboxes are considered
            the same. All but the bbox with the highest probability are removed.
    """
    bboxes = []
    for class_id in set(candidates[:, 4]):
        class_bboxes = candidates[candidates[:, 4] == class_id]
        if class_bboxes.shape[0] == 1:
            # One candidate
            bboxes.append(class_bboxes)
            continue

        while True:
            half_wh = class_bboxes[:, 2:4] * 0.5
            max_index = np.argmax(class_bboxes[:, 5])
            max_bbox = class_bboxes[max_index, :]
            max_half_wh = half_wh[max_index, :]
            # Max probability
            bboxes.append(max_bbox[np.newaxis, :])

            enclose_left = np.minimum(
                class_bboxes[:, 0] - half_wh[:, 0],
                max_bbox[0] - max_half_wh[0],
            )
            enclose_right = np.maximum(
                class_bboxes[:, 0] + half_wh[:, 0],
                max_bbox[0] + max_half_wh[0],
            )
            enclose_top = np.minimum(
                class_bboxes[:, 1] - half_wh[:, 1],
                max_bbox[1] - max_half_wh[1],
            )
            enclose_bottom = np.maximum(
                class_bboxes[:, 1] + half_wh[:, 1],
                max_bbox[1] + max_half_wh[1],
            )

            enclose_width = enclose_right - enclose_left
            enclose_height = enclose_bottom - enclose_top

            width_mask = enclose_width >= class_bboxes[:, 2] + max_bbox[2]
            height_mask = enclose_height >= class_bboxes[:, 3] + max_bbox[3]
            # bboxes with no overlap with max_bbox
            no_overlap_mask = np.logical_or(width_mask, height_mask)
            no_overlap_bboxes = class_bboxes[no_overlap_mask]

            overlap_mask = np.logical_not(no_overlap_mask)
            class_bboxes = class_bboxes[overlap_mask]
            if class_bboxes.shape[0] == 1:
                if no_overlap_bboxes.shape[0] == 1:
                    bboxes.append(no_overlap_bboxes)
                    break

                class_bboxes = no_overlap_bboxes
                continue

            half_wh = half_wh[overlap_mask]
            enclose_left = enclose_left[overlap_mask]
            enclose_right = enclose_right[overlap_mask]
            enclose_top = enclose_top[overlap_mask]
            enclose_bottom = enclose_bottom[overlap_mask]

            inter_left = np.maximum(
                class_bboxes[:, 0] - half_wh[:, 0],
                max_bbox[0] - max_half_wh[0],
            )
            inter_right = np.minimum(
                class_bboxes[:, 0] + half_wh[:, 0],
                max_bbox[0] + max_half_wh[0],
            )
            inter_top = np.maximum(
                class_bboxes[:, 1] - half_wh[:, 1],
                max_bbox[1] - max_half_wh[1],
            )
            inter_bottom = np.minimum(
                class_bboxes[:, 1] + half_wh[:, 1],
                max_bbox[1] + max_half_wh[1],
            )

            class_bboxes_area = class_bboxes[:, 2] * class_bboxes[:, 3]
            max_bbox_area = max_bbox[2] * max_bbox[3]
            inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
            iou = inter_area / (class_bboxes_area + max_bbox_area - inter_area)

            c_squared = (enclose_right - enclose_left) * (
                enclose_right - enclose_left
            ) + (enclose_bottom - enclose_top) * (enclose_bottom - enclose_top)
            d_squared = (class_bboxes[:, 0] - max_bbox[0]) * (
                class_bboxes[:, 0] - max_bbox[0]
            ) + (class_bboxes[:, 1] - max_bbox[1]) * (
                class_bboxes[:, 1] - max_bbox[1]
            )

            # DIoU = IoU - d^2 / c^2
            little_overlap_mask = iou - d_squared / c_squared < threshold
            little_overlap_bboxes = class_bboxes[little_overlap_mask]
            if (
                no_overlap_bboxes.shape[0] > 0
                and little_overlap_bboxes.shape[0] > 0
            ):
                class_bboxes = np.concatenate(
                    [no_overlap_bboxes, little_overlap_bboxes], axis=0
                )
                continue

            if no_overlap_bboxes.shape[0] > 0:
                if no_overlap_bboxes.shape[0] == 1:
                    bboxes.append(no_overlap_bboxes)
                    break

                class_bboxes = no_overlap_bboxes
                continue

            if little_overlap_bboxes.shape[0] > 0:
                if little_overlap_bboxes.shape[0] == 1:
                    bboxes.append(little_overlap_bboxes)
                    break

                class_bboxes = little_overlap_bboxes
                continue

            break

    if len(bboxes) == 0:
        return np.zeros(shape=(1, 6))

    return np.concatenate(bboxes, axis=0)


def candidates_to_pred_bboxes(
    candidates,
    input_size,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.25,
):
    """
    @param candidates: Dim(-1, (x, y, w, h, obj_score, probabilities))

    @return Dim(-1, (x, y, w, h, class_id, class_probability))
    """
    # Remove low socre candidates
    # This step should be the first !!
    class_ids = np.argmax(candidates[:, 5:], axis=-1)
    # class_prob = obj_score * max_probability
    class_prob = (
        candidates[:, 4] * candidates[np.arange(len(candidates)), class_ids + 5]
    )
    candidates = candidates[class_prob > score_threshold, :]

    # Remove out of range candidates
    half_wh = candidates[:, 2:4] * 0.5
    mask = candidates[:, 0] - half_wh[:, 0] >= 0
    candidates = candidates[mask, :]
    half_wh = half_wh[mask, :]
    mask = candidates[:, 0] + half_wh[:, 0] <= 1
    candidates = candidates[mask, :]
    half_wh = half_wh[mask, :]
    mask = candidates[:, 1] - half_wh[:, 1] >= 0
    candidates = candidates[mask, :]
    half_wh = half_wh[mask, :]
    mask = candidates[:, 1] + half_wh[:, 1] <= 1
    candidates = candidates[mask, :]

    # Remove small candidates
    candidates = candidates[
        np.logical_and(
            candidates[:, 2] > (4 / input_size[0]),  # width
            candidates[:, 3] > (4 / input_size[1]),  # height
        ),
        :,
    ]

    class_ids = np.argmax(candidates[:, 5:], axis=-1)
    class_prob = (
        candidates[:, 4] * candidates[np.arange(len(candidates)), class_ids + 5]
    )

    # x, y, w, h, class_id, class_probability
    candidates = np.concatenate(
        [
            candidates[:, :4],
            class_ids[:, np.newaxis],
            class_prob[:, np.newaxis],
        ],
        axis=-1,
    )

    return DIoU_NMS(candidates, iou_threshold)


def fit_pred_bboxes_to_original(bboxes, input_size, original_shape):
    """
    @param `bboxes`: Dim(-1, (x, y, w, h, class_id, probability))
    @param `input_size`: (width, height)
    @param `original_shape`: (height, width, channels)
    """

    height, width, _ = original_shape
    bboxes = np.copy(bboxes)

    w_h = width / height
    iw_ih = input_size[0] / input_size[1]

    if w_h > iw_ih:
        scale = w_h / iw_ih
        bboxes[:, 1] = scale * (bboxes[:, 1] - 0.5) + 0.5
        bboxes[:, 3] = scale * bboxes[:, 3]
    elif w_h < iw_ih:
        scale = iw_ih / w_h
        bboxes[:, 0] = scale * (bboxes[:, 0] - 0.5) + 0.5
        bboxes[:, 2] = scale * bboxes[:, 2]

    return bboxes
