import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reads .txt files and extracts voltage values.
def load_txt_data(file_path):
    values_all = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 1

        for line in lines:
            if line.startswith("I"):
                str_data = line.split(' ')
                float_data = []

                for v in str_data:
                    if v != 'I' and v != 'got:':
                        try:
                            float_data.append(float(v))
                        except:
                            print(v, ' in measurement ', i)

                count = len(float_data)

                if count == 208:
                    values_all.append(float_data)
                elif count > 208:
                    print('back to line needed at measurement:', i)
                else:
                    print('Measurement:', i, 'float data:', count, 'str data:', len(str_data))

                i += 1

    return np.array(values_all)  #(N, 208)


def save_boxplots_in_blocks(data, file_name, output_folder, block_size=20):
    if data.size == 0:
        print("No valid data found for:", file_name)
        return

    plot_df = pd.DataFrame(data)
    plot_df.columns = [f"V_{i + 1}" for i in range(plot_df.shape[1])]

    save_folder = os.path.join(output_folder, "boxplots")
    os.makedirs(save_folder, exist_ok=True)

    total_columns = plot_df.shape[1]     # No of cols.

    for start in range(0, total_columns, block_size):
        end = min(start + block_size, total_columns)

        plt.figure(figsize=(14, 6))
        plot_df.iloc[:, start:end].boxplot(showfliers=True)

        plt.title(file_name + f" - Raw Box Plot - Voltage Indices {start + 1} to {end}")
        plt.xlabel("Voltage measurement index")
        plt.ylabel("Voltage [V]")
        plt.xticks(rotation=90)
        plt.tight_layout()

        output_file = os.path.join(
            save_folder,
            file_name + f"_boxplot_{start + 1}_to_{end}.png"
        )

        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", output_file)


folder = r'C:\Users\manoj\OneDrive\Desktop\measurements_29042026\Adjacent\rectangle\non-conductive'

files = os.listdir(folder)

for file in files:
    if file.endswith('.txt'):
        file_path = os.path.join(folder, file)
        print("Loading:", file_path)

        data = load_txt_data(file_path)
        print(file, data.shape)

        save_boxplots_in_blocks(data, file[:-4], folder, block_size=20)