import seaborn as sns

from guaranteed_fair_ensemble.constants import ALL_BASELINES

Colour = tuple[float, float, float]


def _get_baseline_colours(baselines: list[str]) -> dict[str, Colour]:
    palette = sns.color_palette("Greys", n_colors=len(baselines))
    return dict(zip(baselines, palette, strict=True))


def _get_fairensemble_colours() -> dict[str, Colour]:
    palette = sns.color_palette("Dark2", n_colors=3)
    return {
        "FairEnsemble": palette[0],  # Dark
    }


def get_method_colours() -> dict[str, Colour]:
    return {
        **_get_baseline_colours(baselines=ALL_BASELINES),
        **_get_fairensemble_colours(),
    }


def get_method_type_colours() -> dict[str, Colour]:
    all_methods = get_method_colours()
    baseline_colour = ALL_BASELINES[1]
    return {
        "Baseline": all_methods[baseline_colour],
        "FairEnsemble (ours)": all_methods["FairEnsemble"],
    }


if __name__ == "__main__":
    print(get_method_colours())
