# DARLIN-01D0 Contained Dry-Run Commands

Timestamp: 2026-05-10T03:51:14Z

These commands are prepared for DARLIN-01D. They were not executed in DARLIN-01D0.

## TA / Tigre

```bash
cd /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/TA_Tigre && PYTHONPATH=/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN TMPDIR=/home/zhutao/tmp TMP=/home/zhutao/tmp TEMP=/home/zhutao/tmp conda run -n snakemake_darlin snakemake -s /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py --configfile config.yaml --cores 1 --config sbatch=0 -n -p
```

## RA / Rosa

```bash
cd /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/RA_Rosa && PYTHONPATH=/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN TMPDIR=/home/zhutao/tmp TMP=/home/zhutao/tmp TEMP=/home/zhutao/tmp conda run -n snakemake_darlin snakemake -s /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py --configfile config.yaml --cores 1 --config sbatch=0 -n -p
```

## Required Invariants

- Keep `PYTHONPATH=/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN` so `darlin.settings` resolves `CARLIN_dir` under the scratch code copy.
- Keep `-n -p`; do not remove dry-run mode.
- Run one locus at a time.
- Do not run MATLAB, PEAR, FastQC, MultiQC, or preprocessing.
