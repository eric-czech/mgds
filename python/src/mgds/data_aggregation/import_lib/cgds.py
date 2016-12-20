"""
Wrapper API for CGDS (cbioportal) Web Services

See http://www.cbioportal.org/web_api.jsp for more details.
"""
import pandas as pd
import numpy as np
from py_utils.collection_utils import to_batches
from py_utils.collection_utils import subset
import logging
import time

logger = logging.getLogger(__name__)

BASE_URL = 'http://www.cbioportal.org/public-portal/webservice.do?'


# #### Searching CGDS Metadata Examples ####
# pd.set_option('display.max_colwidth', 500)
# dt = cgds.get_genetic_profiles('cellline_ccle_broad')
# dt
# ####

def _to_url(cmd, data=None):
    url = '{}cmd={}'.format(BASE_URL, cmd)
    if data:
        url += '&' + '&'.join(['{}={}'.format(k, v) for k, v in data.items()])
    return url


def _get(cmd, data=None):
    url = _to_url(cmd, data)
    # print(url)
    return pd.read_csv(url, sep='\t', comment='#')


def _is_id(idv):
    return isinstance(idv, str) and len(idv) > 0


def _is_iterable(seq, check_empty=True):
    # Ensure sequence is not a string, which is technically iterable
    # but not desired in this context
    if isinstance(seq, str):
        return False
    try:
        _ = (e for e in seq)
        if check_empty and len(seq) == 0:
            return False
        return True
    except TypeError:
        return False


def get_cancer_studies():
    return _get('getCancerStudies')


def get_cancer_types():
    return _get('getTypesOfCancer')


def get_genetic_profiles(cancer_study_id):
    assert _is_id(cancer_study_id), \
        'Cancer Study ID must be a non-empty string (e.g. "cellline_ccle_broad")'
    return _get('getGeneticProfiles', {'cancer_study_id': cancer_study_id})


def get_case_lists(cancer_study_id):
    assert _is_id(cancer_study_id), \
        'Cancer Study ID must be a non-empty string (e.g. "cellline_ccle_broad")'
    return _get('getCaseLists', {'cancer_study_id': cancer_study_id})


def get_clinical_data(case_list_id):
    assert _is_id(case_list_id), \
        'Case list ID must be a non-empty string (e.g. "cellline_ccle_broad_all")'
    return _get('getClinicalData', {'case_set_id': case_list_id})


def _get_batch_batch_result(gene_ids, batch_size, cmd, args, print_progress=True):
    res = None
    gene_id_batches = list(to_batches(gene_ids, batch_size))
    n = len(gene_id_batches)
    m = max(int(n / 10.), 1)
    for i, gene_ids in enumerate(gene_id_batches):
        if print_progress and i % m == 0:
            logger.info('Processing batch {} of {}'.format(i + 1, n))
        data = dict(args)
        data['gene_list'] = ','.join(gene_ids)
        # Tracer()()

        attempts = 0
        while True:
            attempts += 1
            try:
                part = _get(cmd, data)
                break
            except:
                if attempts > 5:
                    raise
                logger.warn('An http error occurred.  Will try again in 30 seconds ...')
                time.sleep(30)

        if res is None:
            res = part
        else:
            res = res.append(part)
    return res


def get_genetic_profile_data(case_list_id, genetic_profile_id, gene_ids,
                             gene_id_batch_size=50, print_progress=True):
    assert _is_id(case_list_id), \
        'Case list ID must be a non-empty string (e.g. "cellline_ccle_broad_all")'
    assert _is_id(genetic_profile_id), \
        'Genetic Profile ID must be a non-empty string (e.g. "cellline_ccle_broad_log2CNA")'
    assert _is_iterable(gene_ids), 'Gene IDs must be iterable and non-empty'

    # Split given gene list into batches and combine results from each batch
    args = {
        'id_type': 'gene_symbol',
        'case_set_id': case_list_id,
        'genetic_profile_id': genetic_profile_id
    }
    return _get_batch_batch_result(gene_ids, gene_id_batch_size, 'getProfileData', args, print_progress=print_progress)


def get_mutation_data(case_list_id, genetic_profile_id, gene_ids,
                      gene_id_batch_size=50, print_progress=True):
    assert _is_id(case_list_id), \
        'Case list ID must be a non-empty string (e.g. "cellline_ccle_broad_all")'
    assert _is_id(genetic_profile_id), \
        'Genetic Profile ID must be a non-empty string (e.g. "cellline_ccle_broad_log2CNA")'
    assert _is_iterable(gene_ids), 'Gene IDs must be iterable and non-empty'

    args = {
        'id_type': 'gene_symbol',
        'case_set_id': case_list_id,
        'genetic_profile_id': genetic_profile_id
    }
    return _get_batch_batch_result(gene_ids, gene_id_batch_size, 'getMutationData', args, print_progress=print_progress)


def get_protein_array_info(cancer_study_id, array_type, gene_ids, gene_id_batch_size=500, print_progress=True):
    assert _is_id(cancer_study_id), \
        'Cancer Study ID must be a non-empty string (e.g. "cellline_ccle_broad")'
    array_types = ['protein_level', 'phosphorylation']
    assert array_type in array_types, \
        'Protein array type must be one of {}'.format(array_types)
    assert _is_iterable(gene_ids), 'Gene IDs must be iterable and non-empty'

    args = {
        'id_type': 'gene_symbol',
        'cancer_study_id': cancer_study_id,
        'protein_array_type': array_type
    }
    return _get_batch_batch_result(gene_ids, gene_id_batch_size, 'getProteinArrayInfo',
                                   args, print_progress=print_progress)


def get_protein_array_data(case_list_id, array_info):
    assert _is_id(case_list_id), \
        'Case list ID must be a non-empty string (e.g. "cellline_ccle_broad_all")'
    array_types = ['0', '1']
    assert array_info in array_types, \
        'Protein array info flag must be one of {}'.format(array_types)

    data = {
        'case_set_id': case_list_id,
        'array_info': array_info
    }
    return _get('getProteinArrayData', data)


def melt_raw_data(d, rm_null_values=True, value_numeric=True):
    """
    Transforms raw CGDS data from wide format (cell lines in columns) to long format.

    Note that the uniqueness of gene + cell line id combinations will also be ensured.

    :param d: Raw CGDS DataFrame
    :param rm_null_values: If true, rows with a VALUE of null will be removed after transformation to long format
    :param value_numeric: If true, VALUE will be assumed to be numeric and will be cast as such (and results
        will be checked for any non-convertible values)
    :return: Data in long format (ie CELL_LINE, GENE_ID, VALUE)
    """

    # Currently data is in format where gene ids are in rows and cell line ids are in columns -- to correct this,
    # first ensure that the two gene identifiers given do not contain any conflicts:
    assert d.groupby('COMMON')['GENE_ID'].nunique().max() == 1
    assert d.groupby('GENE_ID')['COMMON'].nunique().max() == 1

    # Next, conform the gene id names to something more informative and then pivot
    # the frame so that the gene ids are in rows along with cell line ids (pushed down out of columns):
    d = d.rename(columns={'GENE_ID': 'GENE_ID:CGDS', 'COMMON': 'GENE_ID:HGNC'})
    d = pd.melt(d, id_vars=['GENE_ID:CGDS', 'GENE_ID:HGNC'], var_name='CELL_LINE_ID', value_name='VALUE')

    # Ensure all non-value fields are non-null
    assert np.all(d[[c for c in d if c != 'VALUE']].notnull())

    # If specified drop rows with null values
    if rm_null_values:
        # Create mask for values to be ignored noting that the CGDS web service sometimes
        # returns string null for values
        mask = (d['VALUE'].notnull()) & (d['VALUE'].astype(str) != 'null')

        # Drop any records containing null values for any cell line + gene combination
        d = subset(d, lambda df: df[mask], subset_op='Remove null values for column "VALUE"')

        # Ensure no values are null
        assert np.all(d['VALUE'].notnull())

    # Cast value to numeric unless disabled (which should rarely be the case -- CGDS table results
    # are almost always numeric)
    if value_numeric:
        d['VALUE'] = pd.to_numeric(d['VALUE'])
        assert np.all(d['VALUE'].notnull())

    return d


DEFAULT_IGNORABLE_MUTATION_COLS =  [
    'mutation_status',
    'validation_status',
    'xvar_link', 'xvar_link_pdb', 'xvar_link_msa',
    'reference_read_count_normal',
    'variant_read_count_normal'
]


def prep_mutation_data(d, c_rm=DEFAULT_IGNORABLE_MUTATION_COLS):
    d_exp = d.drop(c_rm, axis=1).copy()

    # Convert Entrez ID to integer (and make sure this doesn't some how lead to differences)
    assert np.all(d_exp['entrez_gene_id'] == d_exp['entrez_gene_id'].astype(np.int64))
    d_exp['entrez_gene_id'] = d_exp['entrez_gene_id'].astype(np.int64)

    # Fill "Functional Impact Score" NAs with unknown (it is always one of 'L', 'M', 'N', or 'H')
    # d_exp['functional_impact_score'] = d_exp['functional_impact_score'].fillna('Unknown')
    # d_exp['sequencing_center'] = d_exp['sequencing_center'].fillna('Unknown')

    # Conform field names and convert to ucase
    d_exp = d_exp.rename(columns={
        'entrez_gene_id': 'GENE_ID:ENTREZ',
        'gene_symbol': 'GENE_ID:HGNC',
        'case_id': 'CELL_LINE_ID'
    })
    d_exp = d_exp.rename(columns=lambda c: c.upper())

    # Remove completely duplicated records
    d_exp = subset(d_exp, lambda df: df[~df.duplicated()], subset_op='Remove duplicate records')

    return d_exp


def aggregate(d, max_replicates=3):
    """ Aggregates measurements for replicates (possibly dupes) for cell line + gene combinations

    :param d: DataFrame from `melt_raw_data`
    :param max_replicates: Maximum expected number of replicates (seems to be around 3 at most); increase if an
        error occurs because this number is exceeded but still seems reasonable
    :return: DataFrame with same format and VALUE expanded to VALUE_MEAN and VALUE_STD as well as replicate count
        distribution as a Series
    """
    # Get number of records per gene + cell line combo
    d_rep_ct = d.groupby(['GENE_ID:HGNC', 'CELL_LINE_ID']).size()

    assert d_rep_ct.max() <= max_replicates, \
        'Number of Gene + Cell Line replicates exceeds max expected {}.  Distribution of replicate counts:\n{} '\
        .format(max_replicates, d_rep_ct.to_dict())

    # Given that the number of cell line + gene replicates is within the expected range,
    # group by those identifiers and determine the mean and standard deviation for each group
    d = d.groupby(['CELL_LINE_ID', 'GENE_ID:HGNC', 'GENE_ID:CGDS'])['VALUE']\
        .agg({'VALUE_MEAN': np.mean, 'VALUE_STD': np.std, 'VALUE_CT': 'count'})
    d['VALUE_STD'] = d['VALUE_STD'].fillna(0) # Replace STD for single replicates with 0
    d = d.reset_index()

    # Ensure that there are now no duplicate values for the same cell line + gene combination
    assert np.all(d.groupby(['GENE_ID:HGNC', 'CELL_LINE_ID']).size() == 1)

    # Compute replicate count distribution
    d_dist = d_rep_ct.value_counts()
    d_dist.name = 'Number of Replicates'

    return d, d_dist


def prep_clinical_data(d, keep_cols=None):
    # Subset raw CGDS clinical data with 100+ fields to only those that are most relevant
    c_clinical = [
        'CASE_ID',
        'AGE',
        'DAYS_TO_BIRTH',
        'DAYS_TO_COLLECTION',
        'DAYS_TO_DEATH',
        'ETHNICITY',
        'RACE',
        'GENDER',
        'HISTOLOGICAL_DIAGNOSIS',
        'HISTOLOGICAL_SUBTYPE',
        'INITIAL_PATHOLOGIC_DX_YEAR',
        'METHOD_OF_INITIAL_SAMPLE_PROCUREMENT',
        'OTHER_PATIENT_ID',
        'OTHER_SAMPLE_ID',
        'PRIMARY_SITE',
        'SAMPLE_TYPE',
        'TISSUE_SOURCE_SITE',
        'TUMOR_TISSUE_SITE',
        'VITAL_STATUS',
        'CANCER_TYPE',
        'CANCER_TYPE_DETAILED',
        'HISTOLOGY',
        'DATA_SOURCE',
        'TUMOR_TYPE'
    ]
    if keep_cols is not None:
        c_clinical += keep_cols
    assert 'CASE_ID' in d, 'Raw clinical data must contain "CASE_ID" field'
    d = d[np.intersect1d(c_clinical, d.columns.tolist())]

    # Rename case_id to cell_line_id, per the usual convention across data sources
    d = d.rename(columns={'CASE_ID': 'CELL_LINE_ID'})

    return d
