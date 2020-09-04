import jax
import jax.numpy as jnp


@jax.jit
def _giou(boxes1: jnp.ndarray, boxes2: jnp.ndarray) -> jnp.ndarray:
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = jnp.hsplit(boxes1, 4)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = jnp.hsplit(boxes2, 4)

    b1_width = jnp.maximum(0, b1_xmax - b1_xmin)
    b1_height = jnp.maximum(0, b1_ymax - b1_ymin)
    b2_width = jnp.maximum(0, b2_xmax - b2_xmin)
    b2_height = jnp.maximum(0, b2_ymax - b2_ymin)

    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = jnp.maximum(b1_ymin, b2_ymin)
    intersect_xmin = jnp.maximum(b1_xmin, b2_xmin)
    intersect_ymax = jnp.minimum(b1_ymax, b2_ymax)
    intersect_xmax = jnp.minimum(b1_xmax, b2_xmax)

    intersect_width = jnp.maximum(0, intersect_xmax - intersect_xmin)
    intersect_height = jnp.maximum(0, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = jnp.nan_to_num(intersect_area / union_area)

    enclose_ymin = jnp.minimum(b1_ymin, b2_ymin)
    enclose_xmin = jnp.minimum(b1_xmin, b2_xmin)
    enclose_ymax = jnp.maximum(b1_ymax, b2_ymax)
    enclose_xmax = jnp.maximum(b1_xmax, b2_xmax)
    enclose_width = jnp.maximum(0, enclose_xmax - enclose_xmin)
    enclose_height = jnp.maximum(0, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - jnp.nan_to_num((enclose_area - union_area) / enclose_area)
    return giou.squeeze()


@jax.jit
def giou_loss(boxes1: jnp.ndarray, boxes2: jnp.ndarray) -> jnp.ndarray:
    """Based on: https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/giou_loss.py#L65

    boxes are encoded as [y_min, x_min, y_max, x_max], e.g. jnp.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    """
    if len(boxes1) != len(boxes2):
        raise ValueError(f"len(boxes1) != len(boxes2): {len(boxes1)} != {len(boxes2)}")

    giou = _giou(boxes1, boxes2)
    mean_loss_all_samples = jnp.average(1 - giou)
    return mean_loss_all_samples
