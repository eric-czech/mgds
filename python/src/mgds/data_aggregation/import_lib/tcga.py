from mgds.data_aggregation import database as db
from mgds.data_aggregation import source as src
from mgds.data_aggregation import data_type as dtyp
from mgds.data_aggregation.import_lib import cgds
import pandas as pd
import logging
logger = logging.getLogger(__name__)

# CGDS cancer study ids for the most desirable versions of each dataset.  There multiple for each TCGA
# cohort, typically just the original study and then one "provisional" amendment, but some studies have
# 3 datasets.  These list was chosen as the single study per cohort that had the most complete data and
# was as recent as possible (see notebooks/data_aggregation/import/tcga/v1/tcga-cgds-meta.ipynb)
# for more details)
CANCER_STUDY_IDS = [
    'acc_tcga', 'blca_tcga', 'brca_tcga', 'cesc_tcga', 'chol_tcga',
    'coadread_tcga', 'dlbc_tcga', 'esca_tcga', 'gbm_tcga', 'hnsc_tcga',
    'kich_tcga', 'kirc_tcga', 'kirp_tcga', 'laml_tcga', 'lgg_tcga',
    'lihc_tcga', 'luad_tcga', 'lusc_tcga', 'meso_tcga', 'ov_tcga',
    'paad_tcga', 'pcpg_tcga', 'prad_tcga', 'sarc_tcga', 'skcm_tcga',
    'stad_tcga', 'tgct_tcga', 'thca_tcga', 'thym_tcga', 'ucec_tcga',
    'ucs_tcga', 'uvm_tcga'
]

# All case lists can be expressed as <cancer_study>_all
CASE_LIST_ID_FMT = '{}_all'

# *NOTE* All of the following descriptions and details differ by cancer study but usually only slightly
# and in inconsequential ways (or so it would seem).  Typically, one such description was chosen below
# as being representative of all:


# Methylation (HM450)
# --------------------------------
# Description: Methylation (HM450) beta-values for genes in $X cases. For genes with multiple methylation
# probes, the probe least correlated with expression.
PROF_FMT_METHYLATION = '{}_methylation_hm450'

# Mutations
# --------------------------------
# Description: Mutation data from whole exome sequencing.
PROF_FMT_MUTATIONS = '{}_mutations'


# ### Copy Number ###
# Putative copy-number alterations from GISTIC
# --------------------------------
# Description: Putative copy-number calls on X cases determined using GISTIC 2.0. Values: -2 = homozygous deletion;
#  -1 = hemizygous deletion; 0 = neutral / no change; 1 = gain; 2 = high level amplification.
PROF_FMT_CNA_PUTATIVE = '{}_gistic'
# Relative linear copy-number values
# --------------------------------
# Description: Relative linear copy-number values for each gene (from Affymetrix SNP6).
PROF_FMT_CNA = '{}_linear_CNA'


# ### RNA Expression ###
# mRNA Expression z-Scores (microarray)
# --------------------------------
# Description: mRNA z-Scores (Agilent microarray) compared to the expression distribution of each gene
# tumors that are diploid for this gene.
PROF_FMT_EXPRESSION_ZSCORE = '{}_mrna_median_Zscores'
# mRNA expression (microarray)
# --------------------------------
# Description: Expression levels (Agilent microarray).
PROF_FMT_EXPRESSION = '{}_mrna'


# ### RNA-Seq ###
# mRNA Expression z-Scores (RNA Seq V2 RSEM)
# --------------------------------
# Description: mRNA z-Scores (RNA Seq V2 RSEM) compared to the expression distribution of each
# gene tumors that are diploid for this gene.
PROF_FMT_RNASEQ_ZSCORE = '{}_rna_seq_v2_mrna_median_Zscores'
# mRNA expression (RNA Seq V2 RSEM)
# --------------------------------
# Description: Expression levels for $X genes in $Y $CANCER_STUDY cases (RNA Seq V2 RSEM).
PROF_FMT_RNASEQ = '{}_rna_seq_v2_mrna'


# ### RPPA ###
# Protein expression Z-scores (RPPA)
# --------------------------------
# Description: Protein expression, measured by reverse-phase protein array, Z-scores
PROF_FMT_RPPA_ZSCORE = '{}_rppa_Zscores'
# Protein expression (RPPA)
# --------------------------------
# Description: Protein expression measured by reverse-phase protein array
PROF_FMT_RPPA = '{}_rppa'


def import_genetic_profile_data(
        profile_fmt, data_type, gene_ids,
        case_list_fmt=CASE_LIST_ID_FMT, batch_size=50, cohorts=None):
    tables = []

    for i, cs in enumerate(CANCER_STUDY_IDS):

        # Split study ids named like "lusc_tcga" into two parts and extract first as the "cohort"
        assert len(cs.split('_')) == 2, 'Cancer study "{}" does not match expected format'.format(cs)
        cohort = cs.split('_')[0]

        if cohorts is not None and cohort not in cohorts:
            continue

        # Create a CGDS genetic profile like "lusc_tcga_mrna"
        genetic_profile_id = profile_fmt.format(cs)

        # Create a CGDS case list id like "lusc_tcga_all"
        case_list_id = case_list_fmt.format(cs)

        # Create table to save data as (eg lusc-gene-expression)
        table = '{}-{}'.format(cohort, data_type)
        tables.append(table)

        logger.info(
            'Importing data for study "{}" ({} of {}), cohort "{}", case list "{}", profile "{}", table "{}"'
            .format(cs, i + 1, len(CANCER_STUDY_IDS), cohort, case_list_id, genetic_profile_id, table)
        )

        op = lambda: cgds.get_genetic_profile_data(
            case_list_id, genetic_profile_id, gene_ids, gene_id_batch_size=batch_size
        )
        db.cache_raw_operation(op, src.TCGA_v1, table)

    # Return all names of raw, newly-created tables
    return tables


def import_clinical_data(case_list_id_fmt, cohorts=None):
    tables = []
    for i, cs in enumerate(CANCER_STUDY_IDS):

        # Split study ids named like "lusc_tcga" into two parts and extract first as the "cohort"
        assert len(cs.split('_')) == 2, 'Cancer study "{}" does not match expected format'.format(cs)
        cohort = cs.split('_')[0]

        if cohorts is not None and cohort not in cohorts:
            continue

        # Create a CGDS case list id like "lusc_tcga_all"
        case_list_id = case_list_id_fmt.format(cs)

        # Create table to save data as (eg lusc-gene-expression)
        table = '{}-cellline-meta'.format(cohort)
        tables.append(table)

        logger.info(
            'Importing data for study "{}" ({} of {}), cohort "{}", case list "{}", table "{}"'
            .format(cs, i + 1, len(CANCER_STUDY_IDS), cohort, case_list_id, table)
        )

        op = lambda: cgds.get_clinical_data(case_list_id)
        db.cache_raw_operation(op, src.TCGA_v1, table)
    return tables


def load_genetic_profile_data(data_type, cohorts=None, keep_na_value=False):
    tables = db.tables(src.TCGA_v1, db.RAW)
    d = []
    for table in tables:
        cohort = table.split('-')[0]
        if cohorts is not None and cohort not in cohorts:
            continue
        if data_type in table:
            df = db.load(src.TCGA_v1, db.RAW, table)
            if len(df) > 0:
                # Melt from wide format with cell line id in columns to long format (makes deduping easier)
                df = pd.melt(df, id_vars=['GENE_ID', 'COMMON'], var_name='CELL_LINE_ID', value_name='VALUE')

                # Drop null VALUE settings, if specified; if nothing remains log a warning and continue to next table
                if not keep_na_value:
                    df = df[df['VALUE'].notnull()]
                if len(df) == 0:
                    logger.warn('Data for table "{}" contained only null for the VALUE field'.format(table))
                    continue

                # Drop complete duplicates (the CGDS returns these for a small percentage of cases)
                n = len(df)
                df = df.drop_duplicates()
                if n != len(df):
                    logger.debug('Dropped {} completely duplicated records from table "{}"'.format(n - len(df), table))

                df['COHORT'] = cohort
                df = df.rename(columns={'GENE_ID': 'GENE_ID:CGDS', 'COMMON': 'GENE_ID:HGNC'})
                d.append(df)
    return pd.concat(d)


def load_clinical_data(cohorts=None):
    tables = db.tables(src.TCGA_v1, db.RAW)
    d = []
    for table in tables:
        cohort = table.split('-')[0]
        if cohorts is not None and cohort not in cohorts:
            continue
        if dtyp.CELLLINE_META in table:
            df = db.load(src.TCGA_v1, db.RAW, table)
            df['COHORT'] = cohort
            if len(df) > 0:
                d.append(df)
    return pd.concat(d)
