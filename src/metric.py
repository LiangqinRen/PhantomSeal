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
