import os
import numpy as np
import pandas as pd


# -------------------------------------------------------
# BASE PATH — change main-2 if your folder is different
# -------------------------------------------------------
BASE_PATH = (
    "/Users/Ruaminie/Downloads"
    "/eit-measurement-study-ss2026-main-2"
    "/data/raw"
)

OUTPUT_EXCEL = (
    "/Users/Ruaminie/Downloads"
    "/EIT_SNR_Results.xlsx"
)

PATTERNS = ["Adjacent", "Opposite", "Skip3"]
SHAPES   = ["circle", "rectangle", "triangle"]
TYPES    = ["conductive", "non-conductive"]

# -------------------------------------------------------
# ALL INCLUSION FILES — exact names per pattern/shape/type
# -------------------------------------------------------
INCLUSION_FILES = {
    "Adjacent": {
        "circle": {
            "conductive":     ["centre.txt", "between_centre_edge.txt", "edge_toward_electrode0.txt"],
            "non-conductive": ["centre.txt", "between_centre_edge.txt", "edge_toward_electrode0.txt"],
        },
        "rectangle": {
            "conductive":     ["centre.txt", "edge.txt", "electrode1_2.txt", "electrode3_4.txt", "electrode5_6.txt"],
            "non-conductive": ["centre.txt", "edge.txt", "electrode1_2.txt", "electrode3_4.txt", "electrode5_6.txt"],
        },
        "triangle": {
            "conductive":     ["centre.txt", "base_electrode2_3.txt", "edge_electrode5_6.txt", "edge_electrode9_10_11_12.txt", "electrode7_8.txt"],
            "non-conductive": ["centre.txt", "base_electrode2_3.txt", "electrode5_6.txt",       "edge_electrode9_10_11_12.txt", "electrode7_8.txt"],
        },
    },
    "Opposite": {
        "circle": {
            "conductive":     ["centre.txt", "between_centre_edge.txt", "edge_toward_electrode0.txt"],
            "non-conductive": ["centre.txt", "between_centre_edge.txt", "edge_toward_electrode0.txt"],
        },
        "rectangle": {
            "conductive":     ["centre.txt", "Edge.txt", "electrode1_2.txt", "electrode3_4.txt", "electrode5_6.txt"],
            "non-conductive": ["centre.txt", "edge.txt", "electrode1_2.txt", "electrode3_4.txt", "electrode5_6.txt"],
        },
        "triangle": {
            "conductive":     ["centre.txt", "base_electrode2_3.txt", "electrode5_6.txt", "edge_electrode9_10_11_12.txt", "electrode7_8.txt"],
            "non-conductive": ["centre.txt", "base_electrode_2_3.txt", "edge_electrodes_5_6.txt", "electrode7_8.txt", "edge_electrode9_10_11_12.txt"],
        },
    },
    "Skip3": {
        "circle": {
            "conductive":     ["centre.txt", "between_centre_edge.txt", "edge_toward_electrode0.txt"],
            "non-conductive": ["centre.txt", "between_centre_edge.txt", "edge_toward_electrode0.txt"],
        },
        "rectangle": {
            "conductive":     ["center.txt", "edge.txt", "electrode1_2.txt", "electrode3_4.txt", "electrode5_6.txt"],
            "non-conductive": ["centre.txt", "edge.txt", "electrode1_2.txt", "electrode3_4.txt", "electrode5_6.txt"],
        },
        "triangle": {
            "conductive":     ["centre.txt", "base_electrode2_3.txt", "edge_electrode9_10_11_12.txt", "electrode5_6.txt", "electrode7_8.txt"],
            "non-conductive": ["centre.txt", "base_electrode_2_3.txt", "edge_electrodes_5_6.txt",     "edge_electrodes_7_8.txt", "edge_electrodes_9_10_11_12.txt"],
        },
    },
}


def read_txt_file(file_path, high_value_limit=9):
    """
    Reads a .txt file — supports 'I got: ...' format.
    Accepts rows with either 208 or 192 values.
    Removes rows with outliers (values > high_value_limit).
    Returns array and the detected row size.
    """
    values_all = []
    measurement_number = 1
    detected_size = None

    with open(file_path, "r", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove 'I got:' prefix if present
        if "got:" in line:
            idx = line.index("got:") + 4
            line = line[idx:].strip()

        str_data = line.split()
        float_data = []

        for value in str_data:
            try:
                float_data.append(float(value))
            except ValueError:
                pass

        count = len(float_data)

        # Accept 208 or 192 values
        if count in (208, 192):
            if detected_size is None:
                detected_size = count
            if max(float_data) > high_value_limit:
                pass  # outlier row removed silently
            else:
                values_all.append(float_data)

        measurement_number += 1

    return np.array(values_all), detected_size


def calculate_metrics(raw_water, raw_inclusion):
    V_water      = np.mean(raw_water,     axis=0)
    V_inclusion  = np.mean(raw_inclusion, axis=0)
    noise        = V_inclusion - V_water
    signal_power = np.mean(V_water ** 2)
    noise_power  = np.mean(noise ** 2)
    SNR          = 10 * np.log10(signal_power / noise_power)
    dist         = np.linalg.norm(noise) / np.linalg.norm(V_water)
    return round(SNR, 4), round(dist, 6)


def run_all():
    results = []
    errors  = []

    print("=" * 70)
    print("EIT SNR & DISTINGUISHABILITY — ALL CONFIGURATIONS")
    print("=" * 70)

    for pattern in PATTERNS:
        for shape in SHAPES:

            # One water file per shape — shared for conductive & non-conductive
            water_path = os.path.join(
                BASE_PATH, pattern, shape, "onlywater.txt"
            )

            if not os.path.exists(water_path):
                msg = f"MISSING water: {pattern}/{shape}/onlywater.txt"
                print(msg)
                errors.append(msg)
                continue

            raw_water, water_size = read_txt_file(water_path)

            if raw_water.size == 0:
                msg = f"EMPTY water: {pattern}/{shape}/onlywater.txt"
                print(msg)
                errors.append(msg)
                continue

            print(f"\n--- {pattern} / {shape} | water rows: {len(raw_water)} | values per row: {water_size} ---")

            for inc_type in TYPES:

                file_list = (
                    INCLUSION_FILES
                    .get(pattern, {})
                    .get(shape, {})
                    .get(inc_type, [])
                )

                for pos_file in file_list:

                    inc_path = os.path.join(
                        BASE_PATH, pattern, shape, inc_type, pos_file
                    )

                    if not os.path.exists(inc_path):
                        msg = f"MISSING: {pattern}/{shape}/{inc_type}/{pos_file}"
                        print(msg)
                        errors.append(msg)
                        continue

                    raw_inc, inc_size = read_txt_file(inc_path)

                    if raw_inc.size == 0:
                        msg = f"EMPTY: {pattern}/{shape}/{inc_type}/{pos_file}"
                        print(msg)
                        errors.append(msg)
                        continue

                    # Water and inclusion must have same number of values
                    if raw_water.shape[1] != raw_inc.shape[1]:
                        msg = (
                            f"SIZE MISMATCH: water={raw_water.shape[1]} "
                            f"vs inclusion={raw_inc.shape[1]} — "
                            f"{pattern}/{shape}/{inc_type}/{pos_file}"
                        )
                        print(msg)
                        errors.append(msg)
                        continue

                    snr, dist = calculate_metrics(raw_water, raw_inc)
                    position  = pos_file.replace(".txt", "")

                    results.append({
                        "Pattern":            pattern,
                        "Shape":              shape.capitalize(),
                        "Inclusion Type":     inc_type.capitalize(),
                        "Position":           position,
                        "Values per Row":     water_size,
                        "Water Rows Used":    len(raw_water),
                        "Inclusion Rows":     len(raw_inc),
                        "SNR (dB)":           snr,
                        "Distinguishability": dist,
                    })

                    print(
                        f"  OK {inc_type:15} | {position:35} | "
                        f"SNR={snr} dB | D={dist}"
                    )

    if results:
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_EXCEL, index=False)

        print("\n" + "=" * 70)
        print("DONE!")
        print(f"Configurations calculated : {len(results)}")
        print(f"Missing / skipped         : {len(errors)}")
        print(f"\nExcel saved to:")
        print(f"  {OUTPUT_EXCEL}")
        print("=" * 70)
    else:
        print("No results. Check your folder paths.")

    if errors:
        print("\nMISSING FILES:")
        for e in errors:
            print(" -", e)


run_all()
