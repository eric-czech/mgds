

CANCER_STUDY = 'cellline_ccle_broad'
CASE_LIST_ID = 'cellline_ccle_broad_all'

# Log2 copy-number values
# --------------------------------
# Description: Log2 copy-number values for each gene (from Affymetrix SNP6).
PROF_COPY_NUMBER = 'cellline_ccle_broad_log2CNA'

# Putative copy-number alterations
# --------------------------------
# Description: Putative copy-number calls in 972 samples determined by GISTIC2: -2 = homozygous deletion; -1 =
# hemizygous deletion; 0 = neutral / no change; 1 = gain; 2 = high level amplification. GISTIC2 was run using the
# GenePattern website, using the CCLE segmented data downloaded from the CCLE website and all the default
# parameters, except the confidence, which we increased from 0.75 to 0.99.
PROF_COPY_NUMBER_PUTATIVE = 'cellline_ccle_broad_CNA'

# mRNA Expression z-Scores (microarray)
# -------------------------------------
# Description: mRNA z-Scores (Affymetrix microarray)
PROF_GENE_EXPRESSION_ZSCORE = 'cellline_ccle_broad_mrna_median_Zscores'

# mRNA expression (microarray)
# ----------------------------
# Description: Expression levels for 16219 genes in 991 samples (Affymetrix microarray).
PROF_GENE_EXPRESSION = 'cellline_ccle_broad_mrna'

# Mutations
# ---------
# Description: Mutation data from targeted sequencing (1561 genes).
PROF_MUTATION = 'cellline_ccle_broad_mutations'

