"""
perturbation_parser.py
----------------------
Parses human-readable perturbation specifications (from API payloads,
config files, or CLI arguments) into structured dicts consumed by
do_operator and intervention_utils.

Supported formats
-----------------
String shorthand:
  "TP53:0"          → {gene_name: "TP53", value: 0.0,  mode: "zero"}
  "MYC:2.5"         → {gene_name: "MYC",  value: 2.5,  mode: "fixed"}
  "BRCA1:+1.0"      → {gene_name: "BRCA1",value: 1.0,  mode: "additive"}
  "KRAS:*0.5"       → {gene_name: "KRAS", value: 0.5,  mode: "scale"}

Dict / JSON format (from API):
  {"gene": "TP53", "value": 0.0}
  {"gene": "MYC",  "value": 2.5, "mode": "fixed"}
  {"genes": ["TP53", "KRAS"], "values": [0.0, 0.0]}
"""

from __future__ import annotations

import re
from typing import Dict, List, Union


# ---------------------------------------------------------------------------
# Parsed perturbation dataclass (plain dict for JSON-compatibility)
# ---------------------------------------------------------------------------

ParsedPerturbation = Dict  # keys: gene_name, value, mode


# ---------------------------------------------------------------------------
# String parser
# ---------------------------------------------------------------------------

_STRING_PATTERN = re.compile(
    r"^(?P<gene>[A-Za-z0-9_\-]+)"   # gene name
    r":(?P<prefix>[+\-*]?)"          # optional mode prefix
    r"(?P<value>-?\d+(\.\d+)?)$"     # numeric value
)

_MODE_MAP = {
    "":  "zero_or_fixed",   # resolved below
    "+": "additive",
    "-": "additive",        # negative additive
    "*": "scale",
}


def parse_string_perturbation(spec: str) -> ParsedPerturbation:
    """
    Parse a single string specification like "TP53:0" or "MYC:+1.5".

    Returns:
        {'gene_name': str, 'value': float, 'mode': str}
    """
    spec = spec.strip()
    m = _STRING_PATTERN.match(spec)
    if not m:
        raise ValueError(
            f"Invalid perturbation string: '{spec}'. "
            "Expected format: 'GENE_NAME:[+|-|*]NUMERIC_VALUE'."
        )

    gene_name = m.group("gene")
    prefix    = m.group("prefix")
    raw_val   = m.group("value")

    value = float(raw_val)
    if prefix == "-":
        value = -abs(value)

    mode_raw = _MODE_MAP.get(prefix, "zero_or_fixed")
    if mode_raw == "zero_or_fixed":
        mode = "zero" if value == 0.0 else "fixed"
    else:
        mode = mode_raw

    return {"gene_name": gene_name, "value": value, "mode": mode}


# ---------------------------------------------------------------------------
# Dict / JSON parser
# ---------------------------------------------------------------------------

def parse_dict_perturbation(spec: dict) -> List[ParsedPerturbation]:
    """
    Parse a dict-format perturbation (from API request body).

    Accepts:
        {"gene": "TP53", "value": 0.0}               → single gene
        {"gene": "MYC",  "value": 2.5, "mode": "fixed"}
        {"genes": ["TP53", "KRAS"], "values": [0, 0]} → multi-gene

    Returns:
        List of ParsedPerturbation dicts.
    """
    if "genes" in spec:
        genes  = spec["genes"]
        values = spec.get("values", [0.0] * len(genes))
        mode   = spec.get("mode", "zero_or_fixed")
        if len(genes) != len(values):
            raise ValueError("'genes' and 'values' must have the same length.")
        results = []
        for g, v in zip(genes, values):
            m = mode
            if m == "zero_or_fixed":
                m = "zero" if v == 0.0 else "fixed"
            results.append({"gene_name": g, "value": float(v), "mode": m})
        return results

    if "gene" in spec:
        gene  = spec["gene"]
        value = float(spec.get("value", 0.0))
        mode  = spec.get("mode", "zero" if value == 0.0 else "fixed")
        return [{"gene_name": gene, "value": value, "mode": mode}]

    raise ValueError(f"Cannot parse perturbation dict: {spec}")


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def parse_perturbation(
    spec: Union[str, dict, list],
) -> List[ParsedPerturbation]:
    """
    Parse any supported perturbation format.

    Args:
        spec: str, dict, or list of str/dict

    Returns:
        Flat list of ParsedPerturbation dicts.
    """
    if isinstance(spec, str):
        return [parse_string_perturbation(spec)]

    if isinstance(spec, dict):
        return parse_dict_perturbation(spec)

    if isinstance(spec, list):
        results = []
        for item in spec:
            results.extend(parse_perturbation(item))
        return results

    raise TypeError(f"Unsupported perturbation spec type: {type(spec)}")


# ---------------------------------------------------------------------------
# Resolve gene names → indices
# ---------------------------------------------------------------------------

def resolve_gene_names(
    parsed: List[ParsedPerturbation],
    gene_vocab: Dict[str, int],
) -> List[ParsedPerturbation]:
    """
    Replace 'gene_name' keys with 'gene_idx' using the provided vocabulary.

    Args:
        parsed    : output of parse_perturbation
        gene_vocab: {gene_name: gene_index} mapping

    Returns:
        Same list with 'gene_idx' added (and 'gene_name' preserved for display).
    """
    resolved = []
    for p in parsed:
        name = p["gene_name"]
        if name not in gene_vocab:
            raise KeyError(
                f"Gene '{name}' not found in vocabulary "
                f"(vocab size: {len(gene_vocab)})."
            )
        p = dict(p)
        p["gene_idx"] = gene_vocab[name]
        resolved.append(p)
    return resolved
