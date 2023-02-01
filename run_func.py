from funcx import FuncXClient, FuncXExecutor
from pprint import pprint

import numpy as np 
import pandas as pd
import glob
from subprocess import Popen, PIPE
import argparse
import itertools
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import compare_j

fxc = FuncXClient()
fx = FuncXExecutor(fxc)
endpoint_id = '3d0f563e-f797-40ba-a960-e4614eb258a3'  # gl-login1.arc-ts.umich.edu
#endpoint_id = '21777219-17f0-4de0-ab2d-8ac6df700d18'  # gl-login1.local

endpoint_status = fxc.get_endpoint_status(endpoint_id)
print("Status: %s" % endpoint_status['status'])
pprint(endpoint_status)
#print("Workers: %s" % endpoint_status['logs'][0]['total_workers'])
#print("Tasks: %s" % endpoint_status['logs'][0]['outstanding_tasks'])

def run_score(trait, iteration, root):

	def write_slurm_script(trait, iteration, tissue, root, slurm_path):
		script = ['#!/bin/bash',
				f'#SBATCH --job-name={trait}.{tissue}.ITERATION_{iteration}.run_score',
				'#SBATCH --cpus-per-task=1',
				'#SBATCH --nodes=1',
				'#SBATCH --ntasks-per-node=1',
				'#SBATCH --mem-per-cpu=65g',
				'#SBATCH --time=5:00:00',
				'#SBATCH --account=apboyle99',
				'#SBATCH --partition=largmeme',
				'#SBATCH --output=logs/%x.log',
				'#SBATCH --array=1-50',
				'',
				'ANCESTRY="EUR"',
				f'TRAIT="{trait}"',
				f'TISSUE="{tissue}"',
				f'ITERATION="{iteration}"',
				f'ROOT_PATH="{root}"',
				'',
				'EUR_1KG_BFILE="${ROOT_PATH}/data/1KG/1000G_EUR_Phase3_plink/1000G.EUR.QC"',
				'SUMSTATS="${ROOT_PATH}/data/GWAS/${TRAIT}/${TRAIT}.PLINK.TITR"',
				'CLUMP="${ROOT_PATH}/data/GWAS/${TRAIT}/${TRAIT}.PLINK.dedup"',
				'TISSUE_PATH="${ROOT_PATH}/data/RegDB/${TISSUE}/1000G_phase3_master_scores_1KG-pruned_quantile_normalized_chrALL.${TISSUE}.${SLURM_ARRAY_TASK_ID}.SNPs"',
				'OUTPATH="${ROOT_PATH}/results/${TISSUE}"',
				'',
				'PLINK_PATH="/nfs/turbo/boylelab/plink/plink"',
				'PLINK2_PATH="/nfs/turbo/boylelab/plink2/plink2"',
				'',
				'BED_FILE="${ROOT_PATH}/data/UKB/${ANCESTRY}/ukb_imp_chrALL_v3.bed"',
				'BIM_FILE="${ROOT_PATH}/data/UKB/${ANCESTRY}/ukb_imp_chrALL_v3.bim"',
				'FAM_FILE="${ROOT_PATH}/data/UKB/phenos/${ANCESTRY}/${ANCESTRY}.${TRAIT}.fam"',
				'',
				'RANGE_LIST="${ROOT_PATH}/data/UKB/phenos/range_list.expanded"',
				'SAMPLE_PATH="${ROOT_PATH}/data/UKB/phenos/${ANCESTRY}/${ANCESTRY}.${TRAIT}.sample.IDs"',
				'MASTER_SNPS="${ROOT_PATH}/results/traits/${TRAIT}/${TRAIT}.TURF.ITERATION_$((ITERATION-1)).snplist"',
				'EXCLUDE_SNPS="${ROOT_PATH}/results/traits/${TRAIT}/${TRAIT}.TURF.ITERATION_$((ITERATION-1)).exclude"',
				'',
				'SCORING_SNPS="${ROOT_PATH}/data/RegDB/${TISSUE}/1000G_phase3_master_scores_1KG-pruned_quantile_normalized_chrALL.${TISSUE}.${SLURM_ARRAY_TASK_ID}.ITERATION_$((ITERATION)).SNPs"',
				'PREV_SNPS="${ROOT_PATH}/data/RegDB/${TISSUE}/1000G_phase3_master_scores_1KG-pruned_quantile_normalized_chrALL.${TISSUE}.${SLURM_ARRAY_TASK_ID}.ITERATION_$((ITERATION-1)).SNPs"',
				'',
				'rm ${PREV_SNPS}',
				'rm ${OUTPATH}/${TRAIT}.${TISSUE}.${SLURM_ARRAY_TASK_ID}.ITERATION_$((ITERATION-1)).*',
				'',
				'cat ${TISSUE_PATH} ${MASTER_SNPS} | sort | uniq > ${SCORING_SNPS}',
				'',
				'${PLINK_PATH} --bfile ${EUR_1KG_BFILE} --clump-p1 1 --clump-r2 0.2 --clump-kb 250 --clump ${CLUMP} --clump-snp-field SNP --clump-field P \ ',
					'--extract ${SCORING_SNPS} --exclude ${EXCLUDE_SNPS} --out ${OUTPATH}/${TRAIT}.${TISSUE}.${SLURM_ARRAY_TASK_ID}.ITERATION_${ITERATION}',
					'',
				'${PLINK2_PATH} --bed ${BED_FILE} --bim ${BIM_FILE} --fam ${FAM_FILE} --score ${SUMSTATS} 1 2 3 header no-mean-imputation --q-score-range ${RANGE_LIST} ${SUMSTATS}.SNP.pvalue \ ',
				'--extract ${OUTPATH}/${TRAIT}.${TISSUE}.${SLURM_ARRAY_TASK_ID}.ITERATION_${ITERATION}.clumped --keep ${SAMPLE_PATH} --out ${OUTPATH}/${TRAIT}.${TISSUE}.${SLURM_ARRAY_TASK_ID}.ITERATION_${ITERATION} \ ',
				'--write-snplist',
				'',
				'exit']

		slurm_file_path = f'{slurm_path}/{trait}.{tissue}.run_score.sh'

		with open(f'{slurm_file_path}','w') as f:
			f.write('\n'.join(script))

		return slurm_file_path

	def submit_slurm_script(slurm_file_path):
		cmd = f'sbatch {slurm_file_path}'
		p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
		stdout = p.communicate()[0]
		return stdout

	def write_stdout(stdout, trait, iteration, tissue, slurm_path):
		with open(f'{slurm_path}/{trait}.{tissue}.ITERATION_{iteration}.jobs','a') as f:
			f.write(stdout + '\n')


	slurm_path = f'{root}/repo/RegDB-tissue-heritability/PRS/slurm'


	tissues = ["ADIPOSE_TISSUE","ADRENAL_GLAND","ARTERIAL_BLOOD_VESSEL","BLOOD_ORGAN",
				 "BLOOD_VESSEL","BONE_ELEMENT","BONE_MARROW","BRAIN",
				 "BREAST","COLON","CONNECTIVE_TISSUE","EAR",
				 "EMBRYO","ENDOCRINE_GLAND","EPITHELIUM","ESOPHAGUS",
				 "EXOCRINE_GLAND","EXTRAEMBRYONIC_COMPONENT","EYE","GONAD","GENERIC",
				 "HEART","IMMUNE_ORGAN","INTESTINE","KIDNEY",
				 "LARGE_INTESTINE","LIMB","LIVER","LUNG",
				 "LYMPH_NODE","LYMPHOID_TISSUE","MAMMARY_GLAND","MOUTH",
				 "MUSCULATURE_OF_BODY","NERVE","OVARY","PANCREAS",
				 "PENIS","PLACENTA","PROSTATE_GLAND","SKIN_OF_BODY",
				 "SKIN_OF_PREPUCE_OF_PENIS","SMALL_INTESTINE","SPINAL_CORD","SPLEEN",
				 "STOMACH","TESTIS","THYMUS","THYROID_GLAND",
				 "UTERUS","VAGINA","VASCULATURE"]

	for tissue in tissues:
		slurm_file_path = write_slurm_script(trait, iteration, tissue, root, slurm_path)
		stdout = submit_slurm_script(slurm_file_path)
		write_stdout(stdout, trait, iteration, tissue, slurm_path)

future = fx.submit(run_score, "BMI", 1, "/nfs/turbo/boylelab/crone/test/sample_data", endpoint_id=endpoint_id)

print("Status : ", future.done())

print("Result : ", future.result())

