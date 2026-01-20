import torch
import torch.nn as nn


def _strip_dropout(m: nn.Module) -> nn.Module:
    if isinstance(m, nn.Sequential):
        return nn.Sequential(*[x for x in m if not isinstance(x, nn.Dropout)])
    return m


def extract_subhead(fair_ensemble, member_idx: int) -> tuple[nn.Module, nn.Linear]:
    """
    Returns:
      subhead_module: full classifier logic (Sequential or Linear) with last Linear sliced
      last_linear: the sliced last Linear *inside* subhead_module (for merge_heads_pytorch)
    """
    clf = _strip_dropout(fair_ensemble.classifiers)

    layers = list(clf) if isinstance(clf, nn.Sequential) else [clf]
    linear_idxs = [i for i, m in enumerate(layers) if isinstance(m, nn.Linear)]
    if not linear_idxs:
        raise ValueError("No nn.Linear found in classifier to slice.")
    last_i = linear_idxs[-1]
    final: nn.Linear = layers[last_i]

    out_per = final.out_features // fair_ensemble.num_classifiers
    start, stop = member_idx * out_per, (member_idx + 1) * out_per

    sliced = nn.Linear(final.in_features, out_per, bias=(final.bias is not None))
    with torch.no_grad():
        sliced.weight.copy_(final.weight[start:stop])
        if final.bias is not None:
            sliced.bias.copy_(final.bias[start:stop])

    layers[last_i] = sliced

    subhead = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
    return subhead, sliced


def replace_last_linear(subhead: nn.Module, new_last: nn.Linear) -> nn.Module:
    """Replace the last nn.Linear in `subhead` with `new_last` and return subhead."""
    if isinstance(subhead, nn.Linear):
        return new_last

    if not isinstance(subhead, nn.Sequential):
        raise ValueError("Expected subhead to be nn.Linear or nn.Sequential.")

    layers = list(subhead)
    last_i = max(i for i, m in enumerate(layers) if isinstance(m, nn.Linear))
    layers[last_i] = new_last
    return nn.Sequential(*layers)


def _last_linear(m: nn.Module) -> nn.Linear:
    if isinstance(m, nn.Linear):
        return m
    if isinstance(m, nn.Sequential):
        for layer in reversed(list(m)):
            if isinstance(layer, nn.Linear):
                return layer
    raise ValueError(
        f"Expected nn.Linear or nn.Sequential containing nn.Linear, got {type(m)}"
    )


def _split_prefix_and_last_linear(m: nn.Module) -> tuple[list[nn.Module], nn.Linear]:
    """
    Returns (prefix_layers, last_linear) where prefix_layers are all layers
    before the last nn.Linear. Works for Linear or Sequential.
    """
    if isinstance(m, nn.Linear):
        return [], m

    if not isinstance(m, nn.Sequential):
        raise ValueError(f"Expected nn.Linear or nn.Sequential, got {type(m)}")

    layers = list(m)
    # find last linear
    last_i = None
    for i in range(len(layers) - 1, -1, -1):
        if isinstance(layers[i], nn.Linear):
            last_i = i
            break
    if last_i is None:
        raise ValueError("Sequential contains no nn.Linear")

    prefix = layers[:last_i]
    last_linear = layers[last_i]
    assert isinstance(last_linear, nn.Linear)
    return prefix, last_linear


def _same_prefix(a: list[nn.Module], b: list[nn.Module]) -> bool:
    """Conservative prefix equality check by type (and key hyperparams for common layers)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if type(x) is not type(y):
            return False
        # ensure key params match for Dropout if any slipped in, activations, etc.
        if isinstance(x, nn.Dropout) and x.p != y.p:
            return False
        # for other layers, type match is usually sufficient given they're shared logic
    return True


def concat_subheads(subheads: list[nn.Module]) -> nn.Module:
    """
    Combine multiple subhead modules into one module by concatenating their *last Linear*.

    Assumptions:
      - Each subhead outputs a single logit: last_linear.out_features == 1
      - All subheads share the same prefix (for Sequential case)
      - All last linears share the same in_features (and bias presence)

    Returns:
      - nn.Linear if subheads were Linear-only
      - nn.Sequential(prefix..., combined_last_linear) if subheads were Sequential
    """
    if not subheads:
        raise ValueError("subheads must be non-empty")

    prefixes: list[list[nn.Module]] = []
    lasts: list[nn.Linear] = []
    for m in subheads:
        prefix, last = _split_prefix_and_last_linear(m)
        prefixes.append(prefix)
        lasts.append(last)

    # Validate last linears
    in_features = lasts[0].in_features
    use_bias = lasts[0].bias is not None

    if any(layer.in_features != in_features for layer in lasts):
        raise ValueError(
            "All subheads must have the same last-linear in_features to concatenate."
        )
    if any((layer.bias is None) != (not use_bias) for layer in lasts):
        raise ValueError("All subheads must either all have bias or all have no bias.")
    if any(layer.out_features != 1 for layer in lasts):
        raise ValueError(
            "Each subhead's last linear must have out_features=1 to concatenate."
        )

    # Validate prefixes are identical (Sequential case)
    base_prefix = prefixes[0]
    if any(not _same_prefix(base_prefix, p) for p in prefixes[1:]):
        raise ValueError(
            "Subheads do not share the same prefix; cannot combine into one module safely."
        )

    # Build combined last linear
    combined_last = nn.Linear(in_features, len(lasts), bias=use_bias)
    with torch.no_grad():
        combined_last.weight.copy_(torch.cat([l.weight for l in lasts], dim=0))  # noqa: E741
        if use_bias:
            combined_last.bias.copy_(torch.cat([l.bias for l in lasts], dim=0))  # noqa: E741

    # If there is no prefix, return a Linear; otherwise return Sequential(prefix + combined_last)
    if len(base_prefix) == 0:
        return combined_last
    combined = nn.Sequential(*base_prefix, combined_last)

    combined.out_features = combined_last.out_features  # ty:ignore[unresolved-attribute]

    return combined
