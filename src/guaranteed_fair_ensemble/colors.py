import seaborn as sns

from guaranteed_fair_ensemble.constants import PRETTY_METHOD_NAMES

Colour = tuple[float, float, float]

PRETTY_NAMES = list(PRETTY_METHOD_NAMES.values())


def _get_baseline_colours(baselines: list[str]) -> dict[str, Colour]:
    palette = sns.color_palette("Greys", n_colors=len(baselines))
    return dict(zip(baselines, palette, strict=True))


def _get_fairensemble_colours() -> dict[str, Colour]:
    palette = sns.color_palette("Dark2", n_colors=3)
    return {
        "OxEnsemble": palette[0],  # Dark
    }


def get_method_colours() -> dict[str, Colour]:
    return {
        **_get_baseline_colours(baselines=PRETTY_NAMES),
        **_get_fairensemble_colours(),
    }


def get_method_type_colours() -> dict[str, Colour]:
    all_methods = get_method_colours()
    baseline_colour = PRETTY_NAMES[1]
    return {
        "Baseline": all_methods[baseline_colour],
        "OxEnsemble  (ours)": all_methods["OxEnsemble"],
    }


if __name__ == "__main__":
    print(get_method_colours())
