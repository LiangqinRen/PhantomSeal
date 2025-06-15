from torch import tensor


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
            "anchor": (0, 0),
        }
        data["tgt_pert_swap_effectiveness"][function] = {
            "swap": (0, 0),
            "pert_swap": (0, 0),
        }

    return data


def get_defense_metric(
    utility,
    effectiveness,
    imgs_A: tensor,
    imgs_B: tensor,
    x_imgs: tensor,
    cloak_imgs: tensor,
    imgs_A_src_swap: tensor,
    pert_imgs_A_src_swap: tensor,
    imgs_A_tgt_swap: tensor,
    pert_imgs_A_tgt_swap: tensor,
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


def merge_metric(
    effectiveness,
    data: dict,
    pert_utilities: dict,
    pert_as_src_swap_utilities: dict,
    pert_as_tgt_swap_utilities: dict,
    source_effectivenesses: dict,
    target_effectivenesses: dict,
) -> None:
    data["pert_utility"] = tuple(
        x + y
        for x, y in zip(
            data["pert_utility"],
            (
                pert_utilities["mse"],
                pert_utilities["psnr"],
                pert_utilities["ssim"],
                pert_utilities["lpips"],
            ),
        )
    )
    data["src_pert_swap_utility"] = tuple(
        x + y
        for x, y in zip(
            data["src_pert_swap_utility"],
            (
                pert_as_src_swap_utilities["mse"],
                pert_as_src_swap_utilities["psnr"],
                pert_as_src_swap_utilities["ssim"],
                pert_as_src_swap_utilities["lpips"],
            ),
        )
    )
    data["tgt_pert_swap_utility"] = tuple(
        x + y
        for x, y in zip(
            data["tgt_pert_swap_utility"],
            (
                pert_as_tgt_swap_utilities["mse"],
                pert_as_tgt_swap_utilities["psnr"],
                pert_as_tgt_swap_utilities["ssim"],
                pert_as_tgt_swap_utilities["lpips"],
            ),
        )
    )

    for effec in effectiveness.candi_funcs.keys():
        data["src_pert_swap_effectiveness"][effec] = {
            key1: (value1[0] + value2[0], value1[1] + value2[1])
            for (key1, value1), (key2, value2) in zip(
                data["src_pert_swap_effectiveness"][effec].items(),
                source_effectivenesses[effec].items(),
            )
        }
        data["tgt_pert_swap_effectiveness"][effec] = {
            key1: (value1[0] + value2[0], value1[1] + value2[1])
            for (key1, value1), (key2, value2) in zip(
                data["tgt_pert_swap_effectiveness"][effec].items(),
                target_effectivenesses[effec].items(),
            )
        }


def generate_iter_utility_log(utilities: dict) -> str:
    return f"""
    {tuple(f'{v:.5f}' for _,v in utilities.items())}
    """.strip()


def generate_iter_effectiveness_log(effectiveness: dict) -> str:
    content = ""
    for effec in effectiveness:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in effectiveness[effec].items())} "

    return content


def generate_summary_utility_log(data: dict, item: str, batch: int) -> str:
    return f"""
        {tuple(f'{x / (batch):.5f}' for x in data[item])}
        """.strip()


def generate_summary_effectiveness_log(data: dict, item: str) -> str:
    content = ""
    for effec in data[item]:
        content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in data[item][effec].items())} "

    return content
