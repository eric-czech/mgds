
CANCER_STUDY = 'cellline_nci60'
CASE_LIST_ID = 'cellline_nci60_all'

# Relative linear copy-number values
# ----------------------------------
# Description: Relative linear copy-number values for each gene (from Affymetrix SNP6).
PROF_COPY_NUMBER = 'cellline_nci60_linear_CNA'

# Putative copy-number alterations from discretization
# ----------------------------------------------------
# Description: Putative copy-number calls in 60 samples: -2 = homozygous deletion; -1 = hemizygous deletion;
# 0 = neutral / no change; 1 = gain; 2 = high level amplification. Putative copy-number calls in 972 samples
# determined by GISTIC2: -2 = homozygous deletion; -1 = hemizygous deletion; 0 = neutral / no change; 1 = gain;
# 2 = high level amplification. GISTIC2 was run using the GenePattern website.
PROF_COPY_NUMBER_PUTATIVE = 'cellline_nci60_CNA'

# mRNA Expression z-Scores (combined microarray)
# ----------------------------------------------
# Description: mRNA z-Scores (5 Platform Gene Transcript)
PROF_GENE_EXPRESSION = 'cellline_nci60_mrna_median_Zscores'

# Mutations
# ---------
# Description: Mutation data from exome sequencing (includes only type-2 mutations).
PROF_MUTATION = 'cellline_nci60_mutations'