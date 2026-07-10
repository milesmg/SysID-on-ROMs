#!/usr/bin/env python3

import argparse
import itertools
import re
import shlex
import sys


### ADJUSTED: Use BATCH_SIZE for grouped trajectory-window batches.
GROUPED_KEYS = {"ETAS", "ITERS", "BETA", "WINDOW_T", "WINDOW_N_OBS", "WINDOW_START_POLICY", "BATCH_SIZE"}
METADATA_KEYS = {"TARGET", "SWEEP_TARGET", "MODEL", "CODE", "SWEEP_NAME", "RUN_NAME", "BASE_RUN_NAME"}
CASE_SENTINEL = "__ROW_CASE__"


def canonical_key(key):
    normalized = key.strip().upper().replace("-", "_")
    aliases = {
        "NOBS": "N_OBS",
        "N_OBSERVATIONS": "N_OBS",
        "EPSILON2": "EPS2",
        "REFERENCE_DT": "REFERENCE_DT_FACTOR",
        "REFERENCE_DTFACTOR": "REFERENCE_DT_FACTOR",
        "SAVE_FREQ": "SAVE_FREQUENCY",
        "PRINT_FREQ": "PRINT_FREQUENCY",
        "JULIA_THREADS": "JULIA_NUM_THREADS",
        "BLAS_THREADS": "JULIA_BLAS_THREADS",
        ### ADJUSTED: Add aliases for variable-window FOM sweep parameters.
        "WINDOW_LENGTH": "WINDOW_T",
        "WINDOW_LENGTHS": "WINDOW_T",
        "WINDOW_NOBS": "WINDOW_N_OBS",
        "WINDOW_OBS": "WINDOW_N_OBS",
        "WINDOW_N_OBSERVATIONS": "WINDOW_N_OBS",
        "WINDOW_POLICY": "WINDOW_START_POLICY",
        "WINDOW_START": "WINDOW_START_POLICY",
        "BATCH_SIZES": "BATCH_SIZE",
        ### ADJUSTED: Preserve common names for dimension, boundary, and polynomial learner sweeps.
        "BOUNDARY": "BOUNDARY_CONDITION",
        "BC": "BOUNDARY_CONDITION",
        "SPATIAL_DIMENSION": "DIMENSION",
        "POLY_DEGREE": "POLYNOMIAL_DEGREE",
        ### ADJUSTED: Accept short aliases for named sweep initial conditions.
        "INIT_COND": "INITIAL_CONDITION",
        "INITIAL_CONDITION_NAME": "INITIAL_CONDITION",
        "U0_NAME": "INITIAL_CONDITION",
    }
    return aliases.get(normalized, normalized)


def strip_comment(line):
    depth = 0
    quote = None
    escaped = False
    for i, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            continue
        if char == "#" and depth == 0:
            return line[:i]
    return line


def top_level_split(text):
    parts = []
    start = 0
    depth = 0
    quote = None
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            continue
        if char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def is_wrapped_list(text):
    text = text.strip()
    if not (len(text) >= 2 and text[0] == "[" and text[-1] == "]"):
        return False
    depth = 0
    quote = None
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return False
    return True


def unquote(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def normalize_token(token):
    token = unquote(token.strip())
    if is_wrapped_list(token):
        inner = token[1:-1]
        return ",".join(normalize_token(part) for part in top_level_split(inner))
    return token


def parse_values(key, raw_value):
    value = strip_comment(raw_value).strip()
    if not value:
        return []
    if is_wrapped_list(value):
        inner_parts = top_level_split(value[1:-1])
        has_nested_lists = any(is_wrapped_list(part) for part in inner_parts)
        if key in GROUPED_KEYS and not has_nested_lists:
            return [normalize_token(value)]
        return [normalize_token(part) for part in inner_parts]
    parts = top_level_split(value)
    if key in GROUPED_KEYS and len(parts) > 1:
        return [",".join(normalize_token(part) for part in parts)]
    return [normalize_token(part) for part in parts]


def read_sweep_file(path):
    metadata = {}
    variables = []
    ### ADJUSTED: Support explicit row-wise [[case]] sweep files in the moved sweep parser.
    shared_assignments = []
    cases = []
    current_case = None

    def one_value(line_number, key, values):
        if len(values) != 1:
            raise ValueError(f"{path}:{line_number}: row-wise {key} must have exactly one value")
        return values[0]

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = strip_comment(raw_line).strip()
            if not line:
                continue
            if line == "[[case]]":
                if current_case is not None:
                    cases.append(current_case)
                current_case = []
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
            raw_key, raw_value = line.split("=", 1)
            key = canonical_key(raw_key)
            values = parse_values(key, raw_value)
            if not values:
                raise ValueError(f"{path}:{line_number}: {key} has no values")
            if current_case is not None:
                current_case.append((key, one_value(line_number, key, values)))
            elif key in METADATA_KEYS:
                metadata[key] = values[0]
            else:
                if cases:
                    raise ValueError(f"{path}:{line_number}: shared values must appear before the first [[case]]")
                if len(values) == 1:
                    shared_assignments.append((key, values[0]))
                else:
                    variables.append((key, values))
    if current_case is not None:
        cases.append(current_case)
    if cases:
        row_cases = []
        for case in cases:
            row_cases.append(shared_assignments + case)
        return metadata, [(CASE_SENTINEL, row_cases)]
    variables = [(key, [value]) for key, value in shared_assignments] + variables
    return metadata, variables


def combo_count(variables):
    if len(variables) == 1 and variables[0][0] == CASE_SENTINEL:
        return len(variables[0][1])
    count = 1
    for _, values in variables:
        count *= len(values)
    return count


def combo_at(variables, index):
    total = combo_count(variables)
    if index < 0 or index >= total:
        raise IndexError(f"combination index {index} outside 0:{total - 1}")
    if len(variables) == 1 and variables[0][0] == CASE_SENTINEL:
        return variables[0][1][index]
    products = itertools.product(*(values for _, values in variables))
    values = next(itertools.islice(products, index, None))
    return list(zip((key for key, _ in variables), values))


def combo_label(combo, swept_keys=None):
    ### ADJUSTED: Prefer explicit row-wise CASE_NAME values for readable run directories.
    for key, value in combo:
        if key == "CASE_NAME":
            return re.sub(r"[^A-Za-z0-9._+-]+", "_", value).strip("_")[:180] or "case"
    pieces = []
    for key, value in combo:
        if swept_keys is not None and key not in swept_keys:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", value).strip("_")
        pieces.append(f"{key.lower()}-{cleaned[:48]}")
    label = "__".join(pieces)
    return label[:180].strip("_") or "combo"


def target_from_metadata(metadata):
    for key in ("SWEEP_TARGET", "TARGET", "MODEL", "CODE"):
        if key in metadata:
            return metadata[key].strip().lower()
    return ""


def command_count(args):
    _, variables = read_sweep_file(args.sweep_file)
    print(combo_count(variables))


def command_target(args):
    metadata, _ = read_sweep_file(args.sweep_file)
    print(target_from_metadata(metadata))


def command_list(args):
    metadata, variables = read_sweep_file(args.sweep_file)
    total = combo_count(variables)
    print(f"target={target_from_metadata(metadata) or '<unset>'}")
    print(f"count={total}")
    for index in range(total):
        combo = combo_at(variables, index)
        assignments = " ".join(f"{key}={value}" for key, value in combo)
        print(f"{index}: {assignments}")


def command_env(args):
    metadata, variables = read_sweep_file(args.sweep_file)
    total = combo_count(variables)
    combo = combo_at(variables, args.index)
    swept_keys = {key for key, values in variables if len(values) > 1}
    target = target_from_metadata(metadata)
    if target:
        print(f"export SWEEP_TARGET={shlex.quote(target)}")
    for key, value in metadata.items():
        if key == "BASE_RUN_NAME":
            key = "RUN_NAME"
        if key in {"TARGET", "MODEL", "CODE", "SWEEP_TARGET"}:
            continue
        print(f"export {key}={shlex.quote(value)}")
    for key, value in combo:
        print(f"export {key}={shlex.quote(value)}")
    print(f"export SWEEP_COMBO_INDEX={args.index}")
    print(f"export SWEEP_COMBO_COUNT={total}")
    print(f"export SWEEP_COMBO_LABEL={shlex.quote(combo_label(combo, swept_keys))}")


def main(argv):
    parser = argparse.ArgumentParser(description="Expand FOM/ROM HPC sweep files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser("count")
    count_parser.add_argument("sweep_file")
    count_parser.set_defaults(func=command_count)

    target_parser = subparsers.add_parser("target")
    target_parser.add_argument("sweep_file")
    target_parser.set_defaults(func=command_target)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("sweep_file")
    list_parser.set_defaults(func=command_list)

    env_parser = subparsers.add_parser("env")
    env_parser.add_argument("sweep_file")
    env_parser.add_argument("index", type=int)
    env_parser.set_defaults(func=command_env)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
