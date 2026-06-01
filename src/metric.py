from torch import Tensor
from copy import deepcopy


UTILITY_LABELS = {
    "mse": "MSE",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
}

EFFECTIVENESS_LABELS = {
    "pert": "x̃",
    "swap": "ASRₒ",
    "pert_swap": "ASRₚ",
    "cloak": "TSR",
}


def _format_rate(value: tuple) -> str:
    if value[1] == 0:
        return "nan/0"
    return f"{value[0] / value[1] * 100:.3f}/{value[1]:.0f}"


def _format_labeled_rate(name: str, value: tuple) -> str:
    label = EFFECTIVENESS_LABELS.get(name, name)
    return f"{label} {_format_rate(value)}"


def get_metric_data_template(effectiveness) -> dict:
    data = {
        "utility": (0, 0, 0, 0),
        "pert_source_utility": (0, 0, 0, 0),
        "pert_target_utility": (0, 0, 0, 0),
        "pert_source_effectiveness": {},
        "pert_target_effectiveness": {},
    }

    for function in effectiveness.candi_funcs.keys():
        data["pert_source_effectiveness"][function] = {
            "pert": (0, 0),
            "swap": (0, 0),
            "pert_swap": (0, 0),
            "cloak": (0, 0),
        }
        data["pert_target_effectiveness"][function] = {
            "swap": (0, 0),
            "pert_swap": (0, 0),
        }

    return data


def get_robustness_metric_data_template(config, effectiveness) -> dict:
    data = {
        "pert_source_effectiveness": {},
        "pert_target_effectiveness": {},
    }

    for effec in effectiveness.candi_funcs.keys():
        data["pert_source_effectiveness"][effec] = {}
        data["pert_target_effectiveness"][effec] = {}

        if config.evaluate.effectiveness.ASRo:
            data["pert_source_effectiveness"][effec]["swap"] = (0, 0)
            data["pert_target_effectiveness"][effec]["swap"] = (0, 0)

        if config.evaluate.effectiveness.ASRp:
            data["pert_source_effectiveness"][effec]["pert_swap"] = (0, 0)
            data["pert_target_effectiveness"][effec]["pert_swap"] = (0, 0)

        if config.evaluate.effectiveness.TSR:
            data["pert_source_effectiveness"][effec]["cloak"] = (0, 0)

    data = {
        "utility": {"mse": 0, "psnr": 0, "ssim": 0, "lpips": 0},
        "noise": deepcopy(data),
        "compress": deepcopy(data),
        "crop": deepcopy(data),
        "logo": deepcopy(data),
        "brighten": deepcopy(data),
        "darken": deepcopy(data),
    }

    return data


def get_robustness_forensics_metric_data_template(effectiveness) -> dict:
    item_data = {}
    for effec in effectiveness.candi_funcs.keys():
        item_data[effec] = {"cloak": (0, 0)}

    data = {
        "clean": deepcopy(item_data),
        "noise": deepcopy(item_data),
        "compress": deepcopy(item_data),
        "crop": deepcopy(item_data),
        "logo": deepcopy(item_data),
        "brighten": deepcopy(item_data),
        "darken": deepcopy(item_data),
    }

    return data


def get_image_robustness_data_template(effectiveness) -> dict:
    data_item = get_metric_data_template(effectiveness)
    data = {
        "noise": deepcopy(data_item),
        "compress": deepcopy(data_item),
        "crop": deepcopy(data_item),
        "logo": deepcopy(data_item),
        "brighten": deepcopy(data_item),
        "darken": deepcopy(data_item),
    }

    return data


def get_defense_metric(
    utility_evaluator,
    effectiveness_evaluator,
    imgs_A: Tensor,
    imgs_B: Tensor,
    pert_imgs: Tensor,
    cloak_imgs: Tensor | None,
    source_swap: Tensor | None,
    pert_source_swap: Tensor,
    target_swap: Tensor | None,
    pert_target_swap: Tensor | None,
) -> tuple[dict, dict, dict, dict, dict]:
    utility = utility_evaluator.calculate_utility(imgs_A, pert_imgs)
    source_utility = utility_evaluator.calculate_utility(source_swap, pert_source_swap)
    target_utility = utility_evaluator.calculate_utility(target_swap, pert_target_swap)
    source_effectiveness = effectiveness_evaluator.calculate_effectiveness(
        imgs_A,
        pert_imgs,
        source_swap,
        pert_source_swap,
        cloak_imgs,
    )
    target_effectiveness = effectiveness_evaluator.calculate_effectiveness(
        imgs_B, None, target_swap, pert_target_swap, None
    )

    return (
        utility,
        source_utility,
        target_utility,
        source_effectiveness,
        target_effectiveness,
    )


def get_robustness_forensics_metric(
    effectiveness, cloak_imgs: Tensor, swap_results: dict
) -> dict:
    effectivenesses = {}
    for k, v in swap_results.items():
        result = effectiveness.calculate_effectiveness(None, None, None, v, cloak_imgs)
        effectivenesses[k] = result

    return effectivenesses


def merge_single_dict(sum: dict, item: dict):
    # sum and item must have identical structure
    for key in item:
        if isinstance(sum[key], dict) and isinstance(item[key], dict):
            merge_single_dict(sum[key], item[key])
        elif isinstance(sum[key], tuple) and isinstance(item[key], tuple):
            sum[key] = tuple(a + b for a, b in zip(sum[key], item[key]))
        else:
            sum[key] = sum[key] + item[key]


def merge_single_robustness_metric(
    data: dict,
    source_effectiveness: dict,
    target_effectiveness: dict,
    experiment: str,
) -> None:
    merge_single_dict(
        data[experiment]["pert_source_effectiveness"], source_effectiveness
    )
    merge_single_dict(
        data[experiment]["pert_target_effectiveness"], target_effectiveness
    )


def merge_metric(
    effectiveness,
    metrics: dict,
    utility: dict | None,
    source_utility: dict,
    target_utility: dict | None,
    source_effectiveness: dict,
    target_effectiveness: dict | None,
) -> None:
    if utility is not None:
        metrics["utility"] = tuple(
            x + y
            for x, y in zip(
                metrics["utility"],
                (
                    utility["mse"],
                    utility["psnr"],
                    utility["ssim"],
                    utility["lpips"],
                ),
            )
        )
    if source_utility is not None:
        metrics["pert_source_utility"] = tuple(
            x + y
            for x, y in zip(
                metrics["pert_source_utility"],
                (
                    source_utility["mse"],
                    source_utility["psnr"],
                    source_utility["ssim"],
                    source_utility["lpips"],
                ),
            )
        )
    if target_utility is not None:
        metrics["pert_target_utility"] = tuple(
            x + y
            for x, y in zip(
                metrics["pert_target_utility"],
                (
                    target_utility["mse"],
                    target_utility["psnr"],
                    target_utility["ssim"],
                    target_utility["lpips"],
                ),
            )
        )

    for effec in effectiveness.candi_funcs.keys():
        metrics["pert_source_effectiveness"][effec] = {
            key2: (value1[0] + value2[0], value1[1] + value2[1])
            for (key1, value1), (key2, value2) in zip(
                metrics["pert_source_effectiveness"][effec].items(),
                source_effectiveness[effec].items(),
            )
        }
        if target_effectiveness is not None:
            metrics["pert_target_effectiveness"][effec] = {
                key2: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    metrics["pert_target_effectiveness"][effec].items(),
                    target_effectiveness[effec].items(),
                )
            }


def generate_iter_utility_log(utilities: dict) -> str:
    if utilities is None:
        return "()"

    vals = (
        f"{UTILITY_LABELS.get(k, k)} {v:.3f}"
        for k, v in utilities.items()
    )
    return f"({', '.join(vals)})"


def generate_iter_effectiveness_label(effectiveness: dict) -> str:
    for values in effectiveness.values():
        labels = (EFFECTIVENESS_LABELS.get(k, k) for k in values.keys())
        return f"({', '.join(labels)})"
    return "()"


def generate_iter_effectiveness_log(
    effectiveness: dict, include_labels: bool = True
) -> str:
    parts = []

    for effec, values in effectiveness.items():
        if include_labels:
            vals = (_format_labeled_rate(k, v) for k, v in values.items())
        else:
            vals = (_format_rate(v) for v in values.values())
        parts.append(f"{effec} ({', '.join(vals)})")

    return " ".join(parts)


def generate_iter_score_log(scores: dict) -> str:
    vals = (f"{scores[effec]['iter']:.3f}" for effec in scores)
    return f"({', '.join(vals)})"


def generate_summary_utility_log(data: dict, item: str, batch: int) -> str:
    if not data.get(item):
        return "()"
    vals = (
        f"{label} {value / batch:.3f}"
        for label, value in zip(UTILITY_LABELS.values(), data[item])
    )
    return f"({', '.join(vals)})"


def generate_summary_effectiveness_label(data: dict, item: str) -> str:
    for values in data[item].values():
        labels = (EFFECTIVENESS_LABELS.get(k, k) for k in values.keys())
        return f"({', '.join(labels)})"
    return "()"


def generate_summary_effectiveness_log(
    data: dict, item: str, include_labels: bool = True
) -> str:
    parts = []

    for effec, values in data[item].items():
        if include_labels:
            vals = (_format_labeled_rate(k, v) for k, v in values.items())
        else:
            vals = (_format_rate(v) for v in values.values())
        parts.append(f"{effec} ({', '.join(vals)})")

    return " ".join(parts)


def generate_summary_score_log(scores: dict) -> str:
    vals = (f"{scores[effec]['total']:.3f}" for effec in scores)
    return f"({', '.join(vals)})"


def generate_iter_robustness_log(source: dict, target: dict) -> str:
    parts = []

    for effec, values in source.items():
        vals = (_format_labeled_rate(k, v) for k, v in values.items())
        parts.append(f"{effec} ({', '.join(vals)})")

    for effec, values in target.items():
        vals = (_format_labeled_rate(k, v) for k, v in values.items())
        parts.append(f"{effec} ({', '.join(vals)})")

    return " ".join(parts)


def generate_summary_robustness_utility_log(data: dict, batch: int) -> str:
    vals = (
        f"{UTILITY_LABELS.get(k, k)} {v / batch:.3f}"
        for k, v in data.items()
    )
    return f"({', '.join(vals)})"


def generate_summary_robustness_log(data: dict) -> str:
    parts = []

    source = data["pert_source_effectiveness"]
    target = data["pert_target_effectiveness"]

    for effec, values in source.items():
        vals = (_format_labeled_rate(k, v) for k, v in values.items())
        parts.append(f"{effec} ({', '.join(vals)})")

    for effec, values in target.items():
        vals = (_format_labeled_rate(k, v) for k, v in values.items())
        parts.append(f"{effec} ({', '.join(vals)})")

    return " ".join(parts)


def generate_forensics_robustness_log(data: dict) -> str:
    vals = (_format_labeled_rate("cloak", v["cloak"]) for v in data.values())
    return f"({', '.join(vals)})"
