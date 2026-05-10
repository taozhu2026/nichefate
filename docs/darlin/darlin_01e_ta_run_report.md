# DARLIN-01E TA/Tigre Run Report

Timestamp: 2026-05-10T11:26:47Z

## Command

```bash
cd /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01c/TA_Tigre && PYTHONPATH=/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN TMPDIR=/home/zhutao/tmp/darlin_01e TMP=/home/zhutao/tmp/darlin_01e TEMP=/home/zhutao/tmp/darlin_01e conda run -n darlin-repro snakemake -s /home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01d0/code_copy/snakemake_DARLIN/snakefiles/snakefile_matlab_DARLIN_Part1.py --configfile config.yaml --cores 4 --config sbatch=0 --printshellcmds
```

## Result

- Status: `failed_interrupted_after_first_tool_error`
- Exit code: `120`
- Log: `/home/zhutao/scratch/nichefate/darlin_meiji_e1_ra_ta_01e/logs/TA_Tigre_run.log`
- Error category: `ta_failed_environment`

## Error Evidence

- PEAR failed with `pear: unrecognized option '--min-assembly-length'`.
- `run_fastqc.sh` attempted `module load fastqc`, but `module` is not available in this shell.
- FastQC began analyzing R1 before the run was interrupted to honor the stop-on-first-error policy.
- MATLAB/CARLIN was not reached.

## Outputs Observed

- Created scratch-contained `DARLIN/` directories for cutoffs 3 and 10.
- Created scratch-contained `pear_output/` and `fastqc_before_pear/` directories.
- Created Snakemake metadata/log files under `.snakemake/`.
- No `DARLIN_analysis.done`, `Summary.mat`, or PEAR assembled FASTQ was produced.

## Next Action

Patch scratch run_pear.sh to use short PEAR options or compatible PEAR, remove/module-guard run_fastqc.sh module load, clean incomplete TA scratch outputs, then rerun TA only.
