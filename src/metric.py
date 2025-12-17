from torch import Tensor
from copy import deepcopy


def get_metric_data_template(effectiveness) -> dict:
    data = {
        "pert_utility": (0, 0, 0, 0),
        "src_pert_swap_utility": (0, 0, 0, 0),
        "tgt_pert_swap_utility": (0, 0, 0, 0),
        "src_pert_swap_effectiveness": {},
        "tgt_pert_swap_effectiveness": {},
    }

    for function in effectiveness.candi_funcs.keys():
        data["src_pert_swap_effectiveness"][function] = {
            "pert": (0, 0),
            "swap": (0, 0),
            "pert_swap": (0, 0),
            "cloak": (0, 0),
        }
        data["tgt_pert_swap_effectiveness"][function] = {
            "swap": (0, 0),
            "pert_swap": (0, 0),
        }

    return data


def get_robustness_metric_data_template(config, effectiveness) -> dict:
    data = {
        "pert_as_src_effectiveness": {},
        "pert_as_tgt_effectiveness": {},
    }

    for effec in effectiveness.candi_funcs.keys():
        data["pert_as_src_effectiveness"][effec] = {}
        data["pert_as_tgt_effectiveness"][effec] = {}

        if config.evaluate.effectiveness.ASRo:
            data["pert_as_src_effectiveness"][effec]["swap"] = (0, 0)
            data["pert_as_tgt_effectiveness"][effec]["swap"] = (0, 0)

        if config.evaluate.effectiveness.ASRp:
            data["pert_as_src_effectiveness"][effec]["pert_swap"] = (0, 0)
            data["pert_as_tgt_effectiveness"][effec]["pert_swap"] = (0, 0)

        if config.evaluate.effectiveness.TSR:
            data["pert_as_src_effectiveness"][effec]["cloak"] = (0, 0)

    data = {
        "utility": {"mse": 0, "psnr": 0, "ssim": 0, "lpips": 0},
        "noise": deepcopy(data),
        "compress": deepcopy(data),
        "crop": deepcopy(data),
        "logo": deepcopy(data),
        "inc_bright": deepcopy(data),
        "dec_bright": deepcopy(data),
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
        "inc_bright": deepcopy(item_data),
        "dec_bright": deepcopy(item_data),
    }

    return data


def get_image_robustness_data_template(effectiveness) -> dict:
    data_item = get_metric_data_template(effectiveness)
    data = {
        "noise": deepcopy(data_item),
        "compress": deepcopy(data_item),
        "crop": deepcopy(data_item),
        "logo": deepcopy(data_item),
        "inc_bright": deepcopy(data_item),
        "dec_bright": deepcopy(data_item),
    }

    return data


def get_defense_metric(
    utility,
    effectiveness,
    imgs_A: Tensor,
    imgs_B: Tensor,
    x_imgs: Tensor,
    cloak_imgs: Tensor | None,
    imgs_A_src_swap: Tensor,
    pert_imgs_A_src_swap: Tensor,
    imgs_A_tgt_swap: Tensor,
    pert_imgs_A_tgt_swap: Tensor,
) -> tuple[dict, dict, dict, dict, dict]:
    pert_utilities = utility.calculate_utility(imgs_A, x_imgs)
    pert_as_src_swap_utilities = utility.calculate_utility(
        imgs_A_src_swap, pert_imgs_A_src_swap
    )
    pert_as_tgt_swap_utilities = utility.calculate_utility(
        imgs_A_tgt_swap, pert_imgs_A_tgt_swap
    )
    source_effectivenesses = effectiveness.calculate_effectiveness(
        imgs_A,
        x_imgs,
        imgs_A_src_swap,
        pert_imgs_A_src_swap,
        cloak_imgs,
    )
    target_effectivenesses = effectiveness.calculate_effectiveness(
        imgs_B, None, imgs_A_tgt_swap, pert_imgs_A_tgt_swap, None
    )

    return (
        pert_utilities,
        pert_as_src_swap_utilities,
        pert_as_tgt_swap_utilities,
        source_effectivenesses,
        target_effectivenesses,
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
    source_effectivenesses: dict,
    target_effectivenesses: dict,
    experiment: str,
) -> None:
    merge_single_dict(
        data[experiment]["pert_as_src_effectiveness"], source_effectivenesses
    )
    merge_single_dict(
        data[experiment]["pert_as_tgt_effectiveness"], target_effectivenesses
    )


def merge_metric(
    effectiveness,
    metrics: dict,
    pert_utilities: dict,
    pert_as_src_swap_utilities: dict,
    pert_as_tgt_swap_utilities: dict,
    source_effectivenesses: dict,
    target_effectivenesses: dict,
) -> None:
    metrics["pert_utility"] = tuple(
        x + y
        for x, y in zip(
            metrics["pert_utility"],
            (
                pert_utilities["mse"],
                pert_utilities["psnr"],
                pert_utilities["ssim"],
                pert_utilities["lpips"],
            ),
        )
    )
    metrics["src_pert_swap_utility"] = tuple(
        x + y
        for x, y in zip(
            metrics["src_pert_swap_utility"],
            (
                pert_as_src_swap_utilities["mse"],
                pert_as_src_swap_utilities["psnr"],
                pert_as_src_swap_utilities["ssim"],
                pert_as_src_swap_utilities["lpips"],
            ),
        )
    )
    metrics["tgt_pert_swap_utility"] = tuple(
        x + y
        for x, y in zip(
            metrics["tgt_pert_swap_utility"],
            (
                pert_as_tgt_swap_utilities["mse"],
                pert_as_tgt_swap_utilities["psnr"],
                pert_as_tgt_swap_utilities["ssim"],
                pert_as_tgt_swap_utilities["lpips"],
            ),
        )
    )

    for effec in effectiveness.candi_funcs.keys():
        metrics["src_pert_swap_effectiveness"][effec] = {
            key2: (value1[0] + value2[0], value1[1] + value2[1])
            for (key1, value1), (key2, value2) in zip(
                metrics["src_pert_swap_effectiveness"][effec].items(),
                source_effectivenesses[effec].items(),
            )
        }
        metrics["tgt_pert_swap_effectiveness"][effec] = {
            key2: (value1[0] + value2[0], value1[1] + value2[1])
            for (key1, value1), (key2, value2) in zip(
                metrics["tgt_pert_swap_effectiveness"][effec].items(),
                target_effectivenesses[effec].items(),
            )
        }


def generate_iter_utility_log(utilities: dict) -> str:
    return f"""
    {tuple(f'{v:.3f}' for _,v in utilities.items())}
    """.strip()


def generate_iter_effectiveness_log(effectiveness: dict) -> str:
    content = ""
    for effec in effectiveness:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in effectiveness[effec].items())} "

    return content


def generate_iter_score_log(scores: dict) -> str:
    iter_scores = []
    for effec in scores:
        iter_scores.append(f"{scores[effec]['iter']:.3f}")

    return str(tuple(iter_scores))


def generate_summary_utility_log(data: dict, item: str, batch: int) -> str:
    return f"""
        {tuple(f'{x / (batch):.5f}' for x in data[item])}
        """.strip()


def generate_summary_effectiveness_log(data: dict, item: str) -> str:
    content = ""
    for effec in data[item]:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in data[item][effec].items())} "

    return content


def generate_summary_score_log(scores: dict) -> str:
    total_scores = []
    for effec in scores:
        total_scores.append(f"{scores[effec]['total']:.3f}")

    return str(tuple(total_scores))


def generate_iter_robustness_log(source: dict, target: dict) -> str:
    content = ""
    for effec in source:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in source[effec].items())} "
    for effec in target:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in target[effec].items())} "
    return content


def generate_summary_robustness_utility_log(data: dict, batch: int) -> str:
    return f"""
        {tuple(f'{v/batch:.3f}' for _,v in data.items())}
        """.strip()


def generate_summary_robustness_log(data: dict) -> str:
    content = ""
    source = data["pert_as_src_effectiveness"]
    target = data["pert_as_tgt_effectiveness"]
    for effec in source:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in source[effec].items())} "
    for effec in target:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in target[effec].items())} "
    return content


def generate_forensics_robustness_log(data: dict) -> str:
    content = "("
    for _, v in data.items():
        content += f"{v['anchor'][0]/v['anchor'][1]*100:.3f}/{v['anchor'][1]:.0f}, "
    content = content[:-2]
    content += ")"

    return content
