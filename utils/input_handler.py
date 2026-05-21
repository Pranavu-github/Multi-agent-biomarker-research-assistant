"""
utils/input_handler.py — Standardise pipeline input.

Accepts multiple input formats and returns a normalised DataFrame
with columns: Gene, Cell_Type, Disease (optional).

Formats supported:
  1. CSV file path (.csv)
  2. String: "Gene + Cell_Type"  or  "Gene + Cell_Type + Disease"
  3. Dict: {"Gene": ..., "Cell_Type": ..., "Disease": ...}
  4. Dict with lists for batch processing
"""

from __future__ import annotations

import os
import pandas as pd


def load_input_data(input_data) -> pd.DataFrame:
    """
    Normalise input into a DataFrame with columns:
        Gene (str), Cell_Type (str), Disease (str — may be empty)

    Args:
        input_data: One of the supported input formats (see module docstring).

    Returns:
        pd.DataFrame with at least columns: Gene, Cell_Type, Disease.

    Raises:
        FileNotFoundError: If a CSV path is given but doesn't exist.
        ValueError: If the format is unrecognised or column names are wrong.
    """

    # ── CASE 1: CSV path ─────────────────────────────────────────────────────
    if isinstance(input_data, str) and input_data.endswith(".csv"):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"CSV file not found: {input_data}")
        df = pd.read_csv(input_data)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        if "Gene" not in df.columns or "Cell_Type" not in df.columns:
            raise ValueError("CSV must contain columns: Gene, Cell_Type (Disease optional)")
        if "Disease" not in df.columns:
            df["Disease"] = ""
        return df[["Gene", "Cell_Type", "Disease"]].fillna("")

    # ── CASE 2: String "Gene + Cell_Type [+ Disease]" ────────────────────────
    if isinstance(input_data, str) and "+" in input_data:
        parts = [p.strip() for p in input_data.split("+")]
        if len(parts) == 2:
            gene, cell, disease = parts[0], parts[1], ""
        elif len(parts) == 3:
            gene, cell, disease = parts[0], parts[1], parts[2]
        else:
            raise ValueError(
                'String format must be "Gene + Cell_Type" or "Gene + Cell_Type + Disease"'
            )
        return pd.DataFrame({"Gene": [gene], "Cell_Type": [cell], "Disease": [disease]})

    # ── CASE 3 & 4: Dict input ───────────────────────────────────────────────
    if isinstance(input_data, dict):
        gene    = input_data.get("Gene")
        cell    = input_data.get("Cell_Type")
        disease = input_data.get("Disease", "")

        # Single values
        if isinstance(gene, str) and isinstance(cell, str):
            d = "" if disease is None else str(disease)
            return pd.DataFrame({"Gene": [gene], "Cell_Type": [cell], "Disease": [d]})

        # Lists
        if isinstance(gene, list) and isinstance(cell, list):
            if len(gene) != len(cell):
                raise ValueError("Gene and Cell_Type lists must be the same length")
            if isinstance(disease, list):
                if len(disease) != len(gene):
                    raise ValueError("Disease list must match Gene / Cell_Type length")
            else:
                disease = [str(disease) if disease else ""] * len(gene)
            return pd.DataFrame({"Gene": gene, "Cell_Type": cell, "Disease": disease})

    raise ValueError(
        "Invalid input format. Provide:\n"
        "  - CSV path (.csv)\n"
        '  - String: "Gene + Cell_Type" or "Gene + Cell_Type + Disease"\n'
        '  - Dict: {"Gene": ..., "Cell_Type": ..., "Disease": ...}'
    )
