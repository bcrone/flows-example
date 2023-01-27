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

def calc_pval(model, x, y):
	y_len = len(y)
	beta_hat = [model.intercept_] + model.coef_.tolist()
	x1 = np.column_stack((np.ones(y_len), x))
	st_dev_noise = np.sqrt(np.sum(np.square(y - x1@beta_hat))/(y_len - x1.shape[1]))
	beta_cov = np.linalg.inv(x1.T@x1)

	t_val = beta_hat/(st_dev_noise*np.sqrt(np.diagonal(beta_cov)))
	p_vals = t.sf(np.abs(t_val), y_len-x1.shape[1])*2
	p = p_vals[-1]
	return p

def calc_R2(model, x, y, adj_R2):
	y_len = len(y)
	ss_residual = sum((y - model.predict(x))**2)
	ss_total = sum((y - np.mean(y))**2)
	R2 = 1 - float(ss_residual)/ss_total
	new_adj_R2 = 1 - (1 - R2)*(y_len-1)/(y_len-x.shape[1]-1)
	d_R2 = new_adj_R2 - adj_R2
	return new_adj_R2, d_R2

def initializeNullModel(covariates_path, samples_path, trait, isQuant):
	null_pheno = pd.read_table(covariates_path, sep='\t', index_col='FID')
	samples = pd.read_table(samples_path, sep=' ', header=None)
	samples.columns = ['FID','IID']
	null_pheno = null_pheno[null_pheno['IID'].isin(samples['IID'])]
	null_pheno = null_pheno[null_pheno[trait].notna()]
	x = (null_pheno[[
		"Age", "Sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"
		]]).to_numpy()
	if isQuant:
		null_pheno["norm"] = (null_pheno[trait] - null_pheno[trait].mean()) / null_pheno[trait].std()
		y = (null_pheno["norm"]).to_numpy()
	else:
		y = (null_pheno[f"{trait}"]).to_numpy()
	null_model = LinearRegression(n_jobs=-1).fit(x, y)
	null_adj_R2, null_d_R2 = calc_R2(null_model, x, y, 0)
	return [null_pheno, null_model, null_adj_R2]

def calculateCodingModel(threshold, coding_prefix, null_pheno, null_adj_R2, isQuant):
	coding_file = f"{coding_prefix}.{threshold}.sscore"
	try:
		coding_table = pd.read_table(coding_file, delim_whitespace=True, index_col="IID")
	except OSError as e:
		return
	coding_count = max(coding_table["ALLELE_CT"])
	coding_pheno = pd.merge(null_pheno, coding_table["SCORE1_AVG"], on="IID")
	x = (coding_pheno[[
		"Age", "Sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "SCORE1_AVG"
		]]).to_numpy()
	if isQuant:
		y = (coding_pheno["norm"]).to_numpy()
	else:
		y = (coding_pheno[f"{trait}"]).to_numpy()
	coding_model = LinearRegression(n_jobs=-1).fit(x, y)
	coding_adj_R2, coding_d_R2 = calc_R2(coding_model, x, y, null_adj_R2)
	coding_p = calc_pval(coding_model, x, y)
	return ["Coding", threshold, "NA", coding_adj_R2, coding_d_R2, coding_count, coding_p]

def addCodingModel(threshold, coding_prefix, null_pheno, null_adj_R2, isQuant):
	coding_file = f"{coding_prefix}.{threshold}.sscore"
	try:
		coding_table = pd.read_table(coding_file, delim_whitespace=True, index_col="IID")
	except OSError as e:
		return
	coding_count = max(coding_table['ALLELE_CT'])
	coding_pheno = pd.merge(null_pheno, coding_table["SCORE1_AVG"], on="IID")
	coding_pheno.rename(columns={'SCORE1_AVG':'CODING'},inplace=True)	
	if isQuant:
		coding_model = ols("norm ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING", data=coding_pheno).fit()
	else:
		coding_model = ols(f"{trait} ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING", data=coding_pheno).fit()
	coding_adj_R2 = coding_model.rsquared_adj
	coding_d_R2 = coding_adj_R2 - null_adj_R2
	coding_p = coding_model.pvalues['CODING']

	return [coding_pheno, coding_model, coding_adj_R2, coding_d_R2, coding_p, coding_count]

def updateNullModel(trait, iteration, traits_path, pheno, null_adj_R2, isQuant):
	null_table = pd.read_table(f"{traits_path}/{trait}/{trait}.TURF.ITERATION_{iteration}.sscore", delim_whitespace=True, index_col="IID")
	null_table.rename(columns={'SCORE1_AVG':'MASTER'},inplace=True)
	if iteration == 1: # 1st iteration logic
		master_pheno = pd.merge(pheno, null_table['MASTER'], on="IID")
	else:
		master_pheno = pd.merge(pheno.drop(columns='MASTER'), null_table['MASTER'], on="IID")
	if isQuant:
		master_model = ols("norm ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + MASTER", data=master_pheno).fit()
	else:
		master_model = ols(f"{trait} ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + MASTER", data=master_pheno).fit()
	master_adj_R2 = master_model.rsquared_adj
	master_adj_d_R2 = master_adj_R2 - null_adj_R2

	return [master_pheno, master_model, master_adj_R2, master_adj_d_R2]

def runComparison(trait, tissues, partitions, iteration, pheno, adj_R2, added_bin_paths, results_path, isQuant):  # 1:514.4733331203461

	def expand_tissues(tissues, partitions):
		return list(itertools.chain.from_iterable(itertools.repeat(x, len(partitions)) for x in tissues))

	def get_TURF_results(file, iteration, isQuant, tissue, partition, adj_R2):
		p = f"0.{file.split('.')[5]}"
		if p == "0.sscore":
			p = 1
		TURF_table = pd.read_table(file, delim_whitespace=True, index_col='IID')
		if TURF_table['SCORE1_AVG'].isnull().values.any():
			return {'Threshold':p, 'Tissue':tissue, 'Model':partition, "Adj R2":TURF_adj_R2, 
				"Delta Adj R2":np.nan, "P":np.nan, "Count":np.nan}
		TURF_count = max(TURF_table["ALLELE_CT"])
		TURF_pheno = pd.merge(pheno, TURF_table["SCORE1_AVG"], on="IID")
		if iteration == 1: # 1st iteration logic
			x = (TURF_pheno[[
				"Age", "Sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "CODING", "SCORE1_AVG"]]
				).to_numpy()
		else:
			x = (TURF_pheno[[
				"Age", "Sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "CODING", "SCORE1_AVG"]]
				).to_numpy()
		if isQuant:
			y = (TURF_pheno["norm"]).to_numpy()
		else:
			y = (TURF_pheno[f"{trait}"]).to_numpy()
		TURF_model = LinearRegression(n_jobs=-1).fit(x, y)
		TURF_adj_R2, TURF_d_R2 = calc_R2(TURF_model, x, y, adj_R2)
		TURF_p = calc_pval(TURF_model, x, y)
		row = {'Threshold':p, 'Tissue':tissue, 'Model':partition, "Adj R2":TURF_adj_R2, 
			"Delta Adj R2":TURF_d_R2, "P":TURF_p, "Count":TURF_count}
		return row

	def get_max_TURF(tissue, partition, results_path, iteration, isQuant, adj_R2):
		tissue_path = f"{results_path}/{tissue}"
		tissue_files = glob.glob(f"{tissue_path}/{trait}.{tissue}.{partition}.ITERATION_{iteration}.*.sscore")
		TURF_result = pd.DataFrame(
			[get_TURF_results(file, iteration, isQuant, tissue, partition, adj_R2) for file in tissue_files],
			columns=["Threshold","Tissue","Model","Adj R2","Delta Adj R2","P","Count"])
		TURF_result.dropna(inplace=True)
		TURF_bin_max = TURF_result[TURF_result['Delta Adj R2']==TURF_result['Delta Adj R2'].max()]
		return TURF_bin_max
	

	TURF_head = pd.DataFrame(columns=["Threshold","Tissue","Model","Adj R2","Delta Adj R2","P","Count"])
	TURF_max = [get_max_TURF(tissue, partition, results_path, iteration, isQuant, adj_R2) for tissue, partition in zip(
		expand_tissues(tissues, partitions), [str(p) for p in partitions]*len(tissues)) if f"{tissue}.{partition}" not in added_bin_paths]
	
	return TURF_head.append(TURF_max, ignore_index=True)

def runJtest(max_model, model, pheno, trait, iteration, results_path, isQuant):
	path=f"{results_path}/{max_model.Tissue.values[0]}"
	file = f"{path}/{trait}.{max_model.Tissue.values[0]}.{max_model.Model.values[0]}.ITERATION_{iteration}.{max_model.Threshold.values[0]}.sscore"
	model_table = pd.read_table(file, delim_whitespace=True, index_col="IID")
	model_count = max(model_table["ALLELE_CT"])
	model_pheno = pd.merge(pheno, model_table["SCORE1_AVG"], on="IID")
	if isQuant:
		model_model = ols("norm ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + SCORE1_AVG", data=model_pheno).fit()
	else:
		model_model = ols(f"{trait} ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + SCORE1_AVG", data=model_pheno).fit()
	return [compare_j(model_model, model)[1] <= 0.05, compare_j(model, model_model)[1] > 0.05]

def runANOVA(max_model, model, pheno, trait, iteration, results_path, isQuant):
	path = f"{results_path}/{max_model.Tissue.values[0]}"
	file = f"{path}/{trait}.{max_model.Tissue.values[0]}.{max_model.Model.values[0]}.ITERATION_{iteration}.{max_model.Threshold.values[0]}.sscore"
	model_table = pd.read_table(file, delim_whitespace=True, index_col="IID")
	model_count = max(model_table["ALLELE_CT"])
	model_pheno = pd.merge(pheno, model_table["SCORE1_AVG"], on="IID")
	if iteration == 1: # 1st iteration logic
		if isQuant:
			model_model = ols("norm ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + SCORE1_AVG", data=model_pheno).fit()
		else:
			model_model = ols(f"{trait} ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + SCORE1_AVG", data=model_pheno).fit()
	else:
		if isQuant:
			model_model = ols("norm ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + MASTER + SCORE1_AVG", data=model_pheno).fit()
		else:
			model_model = ols(f"{trait} ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + MASTER + SCORE1_AVG", data=model_pheno).fit()
	return [anova_lm(model,model_model)['Pr(>F)'][1] < 0.05,path]

def addPartitionSNPs(master, trait, tissue, partition, threshold, iteration, GWAS_path, results_path):
	snp_pvalue = pd.read_table(f"{GWAS_path}/{trait}/{trait}.PLINK.TITR.SNP.pvalue", delim_whitespace=True)
	snp_pvalue.rename(columns={'SNP':'ID'}, inplace=True)
	partition_snp = pd.read_table(f"{results_path}/{tissue}/{trait}.{tissue}.{partition}.ITERATION_{iteration}.snplist",names=['ID'])
	partition_snp_pvalue = pd.merge(partition_snp, snp_pvalue, on="ID")
	partition_snp_threshold = partition_snp_pvalue[partition_snp_pvalue["P"] < float(threshold)]
	[master.append(snp) for snp in partition_snp_threshold.ID.values.tolist() if snp not in master]

def tagClumpedSNPs(trait, tissue, partition, threshold, iteration, GWAS_path, results_path):
	clumped_snp = pd.read_table(f"{results_path}/{tissue}/{trait}.{tissue}.{partition}.ITERATION_{iteration}.clumped", delim_whitespace=True)
	clumped_snp_threshold = clumped_snp[clumped_snp["P"] < float(threshold)]
	clumped_snp_threshold['SP2'] = clumped_snp_threshold['SP2'].str.split(',').apply(lambda x: [e.strip('(1)') for e in x])
	return list(set(clumped_snp_threshold['SP2'].sum()))

def writeClumpedList(master, trait, ancestry, iteration, GWAS_path):
	with open(f"{GWAS_path}/{trait}/{ancestry}.{trait}.ITERATION_{iteration}.SNPs", "w") as f:
		for snp in master:
			f.write("%s\n" % snp)

def writeExcludedSNPs(excluded_list, trait, ancestry, iteration, results_path):
	with open(f"{results_path}/traits/{trait}/{trait}.TURF.ITERATION_{iteration}.exclude", "w") as f:
		for snp in excluded_list:
			f.write("%s\n" % snp)

def writeMasterResults(master_results, trait, results_path):
	master_results.to_csv(f"{results_path}/{trait}/{trait}.TURF.master_results.tsv", sep="\t", index=False)

def readMasterResults(trait, results_path):
	master_results = pd.read_table(f"{results_path}/{trait}/{trait}.TURF.master_results.tsv")
	return master_results

def writeMasterPheno(master_pheno, trait, iteration, results_path):
	master_pheno.to_csv(f"{results_path}/{trait}/{trait}.TURF.master_pheno.ITERATION_{iteration}.tsv", sep="\t", index=False)

def writeTURFMax(TURF_max, trait, iteration, results_path):
	TURF_max.to_csv(f"{results_path}/{trait}/{trait}.TURF_max.ITERATION_{iteration}.tsv", sep="\t", index=False)

def readCoefTable(trait, results_path):
	coef_table = pd.read_table(f"{results_path}/{trait}/{trait}.TURF.master_coefs.tsv")
	return coef_table

def writeCoefTable(coef_table, trait, results_path):
	coef_table.to_csv(f"{results_path}/{trait}/{trait}.TURF.master_coefs.tsv", sep="\t", index=False)

def readMasterPheno(trait, iteration, results_path, isQuant):
	master_pheno = pd.read_table(f"{results_path}/{trait}/{trait}.TURF.master_pheno.ITERATION_{iteration-1}.tsv")
	if isQuant:
		master_model = ols("norm ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + MASTER", data=master_pheno).fit()
	else:
		master_model = ols(f"{trait} ~ Age + Sex + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + CODING + MASTER", data=master_pheno).fit()
	return [master_pheno, master_model, master_model.rsquared_adj]

def readMasterSNPs(trait, iteration, results_path):
	with open(f"{results_path}/{trait}/{trait}.TURF.ITERATION_{iteration-1}.snplist") as f:
		master = f.readlines()
	master = list(map(lambda s: s.strip(), master))
	return master

def readExcludedSNPs(trait, iteration, results_path):
	with open(f"{results_path}/traits/{trait}/{trait}.TURF.ITERATION_{iteration-1}.exclude") as f:
		excluded_snps = f.readlines()
	excluded_snps = list(map(lambda s: s.strip(), excluded_snps))
	return excluded_snps

def readSNPList(trait, iteration, results_path):
	with open(f"{results_path}/{trait}/{trait}.TURF.ITERATION_{iteration}.snplist") as f:
		master = f.readlines()
	master = list(map(lambda s: s.strip(), master))
	return master

def buildTissuePartitionList(master_results):
	tissues = master_results['Tissue'].tolist()
	partitions = master_results['Model'].tolist()
	return [str(a) + "." + str(b) for a,b in zip(tissues, partitions)]

def runPlinkScore(ancestry, trait, iteration, root):
	plink_path = "/nfs/turbo/boylelab/plink2/plink2"
	ukb_path = f'{root}/data/UKB'
	GWAS_path = f'{root}/data/GWAS'
	results_path = f'{root}/results/traits'

	err = {}
	returncode = {}

	cmd = f"{plink_path} \
	--bed {ukb_path}/{ancestry}/ukb_imp_chrALL_v3.bed \
	--bim {ukb_path}/{ancestry}/ukb_imp_chrALL_v3.bim \
	--fam {ukb_path}/phenos/{ancestry}/{ancestry}.{trait}.fam \
	--score {GWAS_path}/{trait}/{trait}.PLINK.TITR 1 2 3 header no-mean-imputation \
	--extract {GWAS_path}/{trait}/{ancestry}.{trait}.ITERATION_{iteration}.SNPs \
	--keep {ukb_path}/phenos/{ancestry}/{ancestry}.{trait}.sample.IDs \
	--out {results_path}/{trait}/{trait}.TURF.ITERATION_{iteration} --allow-no-sex --write-snplist"

	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)

	err['score'] = p.communicate()[1]
	returncode['score'] = p.returncode
	return [returncode,err]

def evaluateFirstIteration(ancestry, trait, iteration, tissues, thresholds, partitions, isQuant, root):
	master = []
	results = []
	added_bin_paths = []
	master_results = pd.DataFrame(columns=["Threshold","Tissue","Model","Adj R2","Delta Adj R2","P","SNP Count","Null Adj R2"])

	results_path = f'{root}/results'
	traits_path = f'{results_path}/traits'
	GWAS_path = f'{root}/data/GWAS'
	covariates_path = f'{root}/data/UKB/phenos/{ancestry}/{ancestry}.{trait}.covariates'
	samples_path = f'{root}/data/UKB/phenos/{ancestry}/{ancestry}.{trait}.sample.IDs'
	coding_prefix = f'{root}/results/coding/{trait}/{trait}.coding'

	# Build null model
	null_pheno, null_model, null_adj_R2 = initializeNullModel(covariates_path, samples_path, trait, isQuant)

	# Calculate coding models
	for threshold in thresholds:
		results.append(calculateCodingModel(threshold, coding_prefix, null_pheno, null_adj_R2, isQuant))

	# Select best coding model (greatest dR2 over null)
	coding_results = pd.DataFrame(results, columns=["Model","Threshold","Partition","Adj R2","Delta R2","SNP Count","P"])
	best_coding_model = coding_results.iloc[coding_results['Delta R2'].idxmax()]
	best_coding_threshold = best_coding_model['Threshold']

	# Add coding model as covariate in model
	coding_pheno, coding_model, coding_adj_R2, coding_d_R2, coding_p, coding_count = addCodingModel(best_coding_threshold, coding_prefix, null_pheno, null_adj_R2, isQuant)
	row = {"Threshold":best_coding_threshold,"Tissue":"NULL","Model":"Coding","Adj R2":coding_adj_R2,"Delta Adj R2":coding_d_R2, 
		"P":coding_p, "SNP Count":coding_count, "Null Adj R2":null_adj_R2}
	master_results = master_results.append(row, ignore_index=True)
	TURF_max = runComparison(trait, tissues, partitions, iteration, coding_pheno, coding_adj_R2, added_bin_paths, results_path, isQuant)
	TURF_max = TURF_max.sort_values(by="Delta Adj R2",ascending=False)
	writeTURFMax(TURF_max, trait, iteration, traits_path)
	max_model = TURF_max[TURF_max["Delta Adj R2"] == TURF_max["Delta Adj R2"].max()]
	[pass_ANOVA,path] = runANOVA(max_model, coding_model, coding_pheno, trait, iteration, results_path, isQuant)
	
	if pass_ANOVA:
		addPartitionSNPs(master, trait, max_model.Tissue.values[0], max_model.Model.values[0], max_model.Threshold.values[0], iteration, GWAS_path, results_path)
		writeClumpedList(master, trait, ancestry, iteration, GWAS_path)

		isError, msg = runPlinkScore(ancestry, trait, iteration, root)
		if isError:
			print(msg)

		master_pheno, master_model, master_adj_R2, master_adj_d_R2 = updateNullModel(trait, iteration, traits_path, coding_pheno, coding_adj_R2, isQuant)
		coef_table = master_model.params.to_frame().transpose()
		writeCoefTable(coef_table, trait, traits_path)
		snplist = readSNPList(trait, iteration, traits_path)
		clumped_snps = tagClumpedSNPs(trait, max_model.Tissue.values[0], max_model.Model.values[0], max_model.Threshold.values[0], iteration, GWAS_path, results_path)
		excluded_snps = list(set(clumped_snps) - set(snplist))
		writeExcludedSNPs(excluded_snps, trait, ancestry, iteration, results_path)

		row = {"Threshold":max_model.Threshold.values[0],"Tissue":max_model.Tissue.values[0],"Model":max_model.Model.values[0],
		"Adj R2":master_adj_R2,"Delta Adj R2":master_adj_d_R2,"P":max_model['P'].values[0],"SNP Count":len(snplist),"Null Adj R2":null_adj_R2}

		master_results = master_results.append(row, ignore_index=True)
		print(f'Master results after {iteration} iterations for {trait}')
		print(master_results)
		writeMasterResults(master_results, trait, traits_path)
		writeMasterPheno(master_pheno, trait, iteration, traits_path)

	else:
		print('Failed ANOVA!')
		writeMasterResults(master_results, trait, traits_path)
		quit()

def evaluateIteration(ancestry, trait, iteration, tissues, thresholds, partitions, isQuant, root):
	results_path = f'{root}/results'
	traits_path = f'{results_path}/traits'
	GWAS_path = f'{root}/data/GWAS'
	covariates_path = f'{root}/data/UKB/phenos/{ancestry}/{ancestry}.{trait}.covariates'
	samples_path = f'{root}/data/UKB/phenos/{ancestry}/{ancestry}.{trait}.sample.IDs'
	coding_prefix = f'{root}/results/coding/{trait}/{trait}.coding'

	excluded_snps = readExcludedSNPs(trait, iteration, results_path)

	master_results = readMasterResults(trait, traits_path)
	print(f'Master results after {iteration-1} iterations for {trait}')
	print(master_results)
	master = readMasterSNPs(trait, iteration, traits_path)

	null_pheno, null_model, null_adj_R2 = initializeNullModel(covariates_path, samples_path, trait, isQuant)
	master_pheno, master_model, master_adj_R2 = readMasterPheno(trait, iteration, traits_path, isQuant)
	coef_table = readCoefTable(trait, traits_path)
	added_bin_paths = buildTissuePartitionList(master_results)

	TURF_max = runComparison(trait, tissues, partitions, iteration, master_pheno, master_adj_R2, added_bin_paths, results_path, isQuant)
	TURF_max = TURF_max.sort_values(by="Delta Adj R2",ascending=False)
	writeTURFMax(TURF_max, trait, iteration, traits_path)
	max_model = TURF_max[TURF_max["Delta Adj R2"] == TURF_max["Delta Adj R2"].max()]
	[fwd, bwd] = runJtest(max_model, master_model, master_pheno, trait, iteration, results_path, isQuant)
	print(f'Comparision complete - Max Model')
	print(max_model)

	if fwd and bwd:
		addPartitionSNPs(master, trait, max_model.Tissue.values[0], max_model.Model.values[0], max_model.Threshold.values[0], iteration, GWAS_path, results_path)
		writeClumpedList(master, trait, ancestry, iteration, GWAS_path)

		isError, msg = runPlinkScore(ancestry, trait, iteration, root)
		if isError:
			print(msg)

		master_pheno, master_model, master_adj_R2, master_adj_d_R2 = updateNullModel(trait, iteration, traits_path, master_pheno, master_adj_R2, isQuant)
		coef_table = pd.concat([coef_table,master_model.params.to_frame().transpose()])
		writeCoefTable(coef_table, trait, traits_path)
		snplist = readSNPList(trait, iteration, traits_path)
		clumped_snps = tagClumpedSNPs(trait, max_model.Tissue.values[0], max_model.Model.values[0], max_model.Threshold.values[0], iteration, GWAS_path, results_path)
		excluded_snps = list(set(clumped_snps) - set(snplist))
		writeExcludedSNPs(excluded_snps, trait, ancestry, iteration, results_path)
		added_bin_paths.append("{}.{}".format(max_model.Tissue.values[0],max_model.Model.values[0]))

		row = {"Threshold":max_model.Threshold.values[0],"Tissue":max_model.Tissue.values[0],"Model":max_model.Model.values[0],
		"Adj R2":master_adj_R2,"Delta Adj R2":master_adj_d_R2,"P":max_model['P'].values[0],"SNP Count":len(snplist),"Null Adj R2":null_adj_R2}

		master_results = master_results.append(row, ignore_index=True)
		print(f'Master results after {iteration} iterations for {trait}')
		print(master_results)
		writeMasterResults(master_results, trait, traits_path)
		writeMasterPheno(master_pheno, trait, iteration, traits_path)

	else:
		print('Failed J Test!')
		print(f'Forward: {fwd}')
		print(f'Backward: {bwd}')
		writeMasterResults(master_results, trait, traits_path)
		quit()

def main(ancestry, trait, iteration, isQuant, root):
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

	thresholds = ['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1',
				  '0.1','0.09','0.08','0.07','0.06','0.05','0.04','0.03','0.02','0.01',
				  '0.009','0.008','0.007','0.006','0.005','0.004','0.003','0.002','0.001',
				  '0.0009','0.0008','0.0007','0.0006','0.0005','0.0004','0.0003','0.0002','0.0001',
				  '0.00009','0.00008','0.00007','0.00006','0.00005','0.00004','0.00003','0.00002','0.00001']

	partitions = range(1,101)

	print(f'Evaluating iteration {iteration} for {trait}')

	if iteration == 1:
		evaluateFirstIteration(ancestry, trait, iteration, tissues, thresholds, partitions, isQuant, root)

	else:
		evaluateIteration(ancestry, trait, iteration, tissues, thresholds, partitions, isQuant, root)

future = fx.submit(main, "EUR", "height", 1, True, "/nfs/turbo/boylelab/crone/test/sample_data", endpoint_id=endpoint_id)

print("Status : ", future.done())

print("Result : ", future.result())

