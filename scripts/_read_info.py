import json

with open('scripts/_dixit_info.json') as f:
    data = json.load(f)

with open('scripts/_dixit_readable.txt', 'w') as out:
    out.write(f"SHAPE: {data['shape']}\n\n")
    out.write("OBS_COLS:\n")
    for c in data['obs_columns']:
        out.write(f"  - {c}\n")
    out.write("\nVAR_COLS:\n")
    for c in data['var_columns']:
        out.write(f"  - {c}\n")
    out.write(f"\nVAR_NAMES_SAMPLE: {data['var_names_sample']}\n")
    if 'var_gene_name_sample' in data:
        out.write(f"VAR_GENE_NAME_SAMPLE: {data['var_gene_name_sample']}\n")
    if 'perturbation_name_unique_count' in data:
        out.write(f"\nPERT_NAME_COUNT: {data['perturbation_name_unique_count']}\n")
        out.write("PERT_NAME_SAMPLES:\n")
        for s in data['perturbation_name_samples']:
            out.write(f"  - {s}\n")
    if 'guide_identity_unique_count' in data:
        out.write(f"\nGUIDE_ID_COUNT: {data['guide_identity_unique_count']}\n")
        out.write("GUIDE_ID_SAMPLES:\n")
        for s in data['guide_identity_samples']:
            out.write(f"  - {s}\n")

print("Written to scripts/_dixit_readable.txt")
