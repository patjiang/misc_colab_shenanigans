import os
import sys
import re
import hashlib
import argparse
import shutil
import subprocess
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    from Bio import BiopythonDeprecationWarning
    warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
except ImportError:
    pass

def check_dependencies():
    missing_packages = False
    try:
        import colabfold
    except ImportError:
        print("ColabFold not found. Installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"])
        missing_packages = True

    try:
        import openmm
        import pdbfixer
    except ImportError:
        print("OpenMM/pdbfixer not found (required for Amber relax).")
        print("WARNING: OpenMM is best installed via Conda. Attempting a pip install, but if relaxation fails, run: conda install -c conda-forge openmm pdbfixer")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openmm", "pdbfixer"])
        missing_packages = True

    if missing_packages:
        print("Dependencies installed. Please restart the script if you encounter import issues.")

check_dependencies()

from colabfold.download import download_alphafold_params
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type

def add_hash(x, y):
    return x + "_" + hashlib.sha1(y.encode()).hexdigest()[:5]

def main():
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Run AlphaFold2 locally with a custom template.")
    parser.add_argument("sequence", type=str, help="Raw protein sequence (use ':' for chain breaks in complexes)")
    parser.add_argument("template_path", type=str, help="Path to the custom template structure (PDB or mmCIF)")
    args = parser.parse_args()

    # --- Hardcoded Variables ---
    query_sequence = "".join(args.sequence.split())
    jobname = 'alphafold_job'
    num_relax = 1  # amber_relax = 1 as requested
    template_mode = "custom"
    use_amber = num_relax > 0
    msa_mode = "mmseqs2_uniref_env"
    pair_mode = "unpaired_paired"
    model_type = "auto"
    num_recycles = 3
    recycle_early_stop_tolerance = "auto"
    relax_max_iterations = 200
    pairing_strategy = "greedy"
    calc_extra_ptm = False
    max_msa = None
    num_seeds = 1
    use_dropout = False
    save_all = False
    save_recycles = False
    dpi = 200

    basejobname = "".join(jobname.split())
    basejobname = re.sub(r'\W+', '', basejobname)
    jobname = add_hash(basejobname, query_sequence)

    def check_dir(folder):
        return not os.path.exists(folder)

    if not check_dir(jobname):
        n = 0
        while not check_dir(f"{jobname}_{n}"): n += 1
        jobname = f"{jobname}_{n}"

    os.makedirs(jobname, exist_ok=True)
    print(f"Jobname: {jobname}")
    print(f"Sequence length: {len(query_sequence.replace(':', ''))}")

    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    custom_template_path = os.path.join(jobname, "template")
    os.makedirs(custom_template_path, exist_ok=True)
    
    if not os.path.isfile(args.template_path):
        print(f"Error: Template file '{args.template_path}' not found.")
        sys.exit(1)
        
    filename = os.path.basename(args.template_path)
    shutil.copy(args.template_path, os.path.join(custom_template_path, filename))
    use_templates = True

    log_filename = os.path.join(jobname, "log.txt")
    setup_logging(Path(log_filename))

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)
    use_cluster_profile = not ("multimer" in model_type and max_msa is not None)

    recycle_tol = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)

    download_alphafold_params(model_type, Path("."))

    def empty_input_callback(input_features):
        pass

    def empty_prediction_callback(protein_obj, length, prediction_result, input_features, mode):
        pass

    print("Starting prediction...")
    results = run(
        queries=queries,
        result_dir=jobname,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        num_relax=num_relax,
        msa_mode=msa_mode,
        model_type=model_type,
        num_models=5,
        num_recycles=num_recycles,
        relax_max_iterations=relax_max_iterations,
        recycle_early_stop_tolerance=recycle_tol,
        num_seeds=num_seeds,
        use_dropout=use_dropout,
        model_order=[1, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode,
        pairing_strategy=pairing_strategy,
        stop_at_score=float(100),
        prediction_callback=empty_prediction_callback,
        dpi=dpi,
        zip_results=False,
        save_all=save_all,
        max_msa=max_msa,
        use_cluster_profile=use_cluster_profile,
        input_features_callback=empty_input_callback,
        save_recycles=save_recycles,
        user_agent="colabfold/local-cli",
        calc_extra_ptm=calc_extra_ptm,
    )

    print(f"Done! Results and pLDDT plots are saved in the '{jobname}' directory.")

if __name__ == "__main__":
    main()
