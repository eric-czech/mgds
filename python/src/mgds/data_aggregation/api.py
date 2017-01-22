
import pandas as pd
import numpy as np
from mgds.data_aggregation import database as db
from mgds.data_aggregation import source as src
from mgds.data_aggregation import entity
from mgds.data_aggregation import data_type as dtyp

import logging
logger = logging.getLogger(__name__)


def get_hugo_gene_ids():
    """ Get HUGO Gene Symbols"""
    d = db.load(src.HUGO_v1, db.IMPORT, 'gene-meta')
    d = d[~d['Approved Name'].str.lower().str.contains('symbol withdrawn|entry withdrawn')]
    d = d['Approved Symbol'].unique()
    return d


def get_preferred_drug_sensitivity_measurements():
    return dict([
        (src.CTD_v2, 'AUC'),
        (src.GDSC_v2, 'LN_IC50'),
        (src.NCI60_v2, 'LN_GI50'),
        (src.NCIDREAM_v1, 'LN_GI50')
    ])


def get_genomic_data_availability():
    datasets = [
        (src.CCLE_v1, dtyp.GENE_COPY_NUMBER),
        (src.CCLE_v1, dtyp.GENE_EXPRESSION),
        (src.CCLE_v1, dtyp.GENE_EXOME_SEQ),
        (src.GDSC_v2, dtyp.GENE_COPY_NUMBER),
        (src.GDSC_v2, dtyp.GENE_EXPRESSION),
        (src.GDSC_v2, dtyp.GENE_EXOME_SEQ),
        (src.NCI60_v2, dtyp.GENE_COPY_NUMBER),
        (src.NCI60_v2, dtyp.GENE_EXPRESSION),
        (src.NCI60_v2, dtyp.GENE_EXOME_SEQ),
        (src.NCIDREAM_v1, dtyp.GENE_COPY_NUMBER),
        (src.NCIDREAM_v1, dtyp.GENE_EXPRESSION),
        (src.NCIDREAM_v1, dtyp.GENE_EXOME_SEQ),
        (src.NCIDREAM_v1, dtyp.GENE_METHYLATION),
        (src.NCIDREAM_v1, dtyp.GENE_RNA_SEQ),
        (src.GDSC_v2, dtyp.DRUG_SENSITIVITY),
        (src.CTD_v2, dtyp.DRUG_SENSITIVITY),
        (src.NCI60_v2, dtyp.DRUG_SENSITIVITY),
        (src.NCIDREAM_v1, dtyp.DRUG_SENSITIVITY),
        # (src.NCIDREAM_v1, dtyp.GENE_RPPA)
    ]
    return datasets


def get_entity_mapping(entity_type):
    if entity_type == entity.CELL_LINE:
        return db.load(src.MGDS_v1, db.ENTITY, 'cellline-ids-by-typ')
    if entity_type == entity.PRIMARY_SITE:
        return db.load(src.MGDS_v1, db.ENTITY, 'primary-site-by-src')
    if entity_type == entity.DRUG:
        return db.load(src.MGDS_v1, db.ENTITY, 'drug-ids')
    else:
        raise ValueError('Mappings for entity type "{}" not yet supported'.format(entity_type))


def get_cellline_metadata(sources, mappings=None):
    d_id = []
    for source in sources:
        d = get_raw_genomic_data(source, dtyp.CELLLINE_META, mappings=mappings)
        if d is None:
            continue
        d_id.append(d.assign(SOURCE=source))
    d_id = pd.concat(d_id)
    assert np.all(d_id['CELL_LINE_ID:MGDS'].notnull()), \
        'Found null MGDS ID for cell line in metadata -- '\
        'this should not be possible and reflects a mapping omission or error'
    return d_id


def get_drug_sensitivity_data(sources, mappings=None):
    d_drug = []
    for source in sources:
        d = get_raw_genomic_data(source, dtyp.DRUG_SENSITIVITY, mappings=mappings)
        d_drug.append(d.assign(SOURCE=source))
    d_drug = pd.concat(d_drug)

    # TODO: Uncomment this when GDSC null cell lines fixed for drug data
    # assert np.all(d_drug['CELL_LINE_ID:MGDS'].notnull()), \
    #     'Found null MGDS ID for cell line in drug data -- '\
    #     'this should not be possible and reflects a mapping omission or error'

    assert np.all(d_drug['DRUG_NAME:MGDS'].notnull()), \
        'Found null MGDS drug name for cell line in drug data -- '\
        'this should not be possible and reflects a mapping omission or error'
    return d_drug


def get_raw_genomic_data(source, data_type, cell_line_taxonomy='COMMON', drug_name_taxonomy='COMMON', mappings=None):

    if not db.exists(source, db.IMPORT, data_type):
        return None

    d = db.load(source, db.IMPORT, data_type)

    # Determine the field (based on the cell line taxonomy) that should be joined to in order to get normalized ids
    c_cl_id = 'CELL_LINE_ID{}'.format('' if cell_line_taxonomy == 'COMMON' else ':' + cell_line_taxonomy)

    # If this data contains the necessary id field, join it on the normalized id mappings
    if c_cl_id in d:

        # Load cell line mapping data and subset to source, data type, and taxonomy
        if mappings is not None and entity.CELL_LINE in mappings:
            d_id = mappings[entity.CELL_LINE]
        else:
            d_id = get_entity_mapping(entity.CELL_LINE)

        assert cell_line_taxonomy in d_id, \
            'Cell line taxonomy "{}" not found in mapping data'.format(cell_line_taxonomy)
        d_id = d_id[cell_line_taxonomy]
        assert source in d_id, \
            'Source "{}" not found in cell line mapping data for taxonomy "{}"'.format(src, cell_line_taxonomy)
        d_id = d_id[source]
        assert data_type in d_id, \
            'Data type "{}" not found in cell line mapping data for taxonomy "{}" and source "{}"'\
            .format(data_type, cell_line_taxonomy, source)
        d_id = d_id[data_type].reset_index().dropna()

        # Cell line mapping data now has MGDS_ID and source specific identifiers as columns, so first rename the
        # source specific id field to the same name it should join to in the original data
        d_id = d_id.rename(columns={data_type: c_cl_id, 'MGDS_ID': 'CELL_LINE_ID:MGDS'})

        d = pd.merge(d, d_id, on=c_cl_id, how='left')

    if 'CELL_LINE_ID:MGDS' in d:
        assert 'PRIMARY_SITE:MGDS' not in d, 'Data should not already contain a normalized primary site field'
        assert 'PRIMARY_SITE:SOURCE' not in d, 'Data should not already contain a source specific primary site field'

        # Load primary site mapping data
        if mappings is not None and entity.PRIMARY_SITE in mappings:
            d_ps = mappings[entity.PRIMARY_SITE]
        else:
            d_ps = get_entity_mapping(entity.PRIMARY_SITE)

        # If the source is present in primary site mapping, join this mapping to the data
        if source in d_ps:
            d_ps = d_ps[source].reset_index().rename(columns={source: 'PRIMARY_SITE:SOURCE'})
            # Merge to primary site data, resulting in two extra fields (PRIMARY_SITE:[SOURCE|MGDS])
            d = pd.merge(d, d_ps, on=['CELL_LINE_ID:MGDS'], how='left')

        # Otherwise, log a warning that primary sites will not be available for this source + data type
        else:
            logger.warning(
                'Genomic data for source "{}" and data type "{}" has a cell line mapping '
                'but does not have a primary site mapping (so all primary site fields will be null)'
                .format(source, data_type)
            )
            for c in ['PRIMARY_SITE:SOURCE', 'PRIMARY_SITE:MGDS']:
                d[c] = None

    if data_type == dtyp.DRUG_SENSITIVITY:
        assert 'CELL_LINE_ID:MGDS' in d, \
            'Drug sensitivity data for source "{}" contains no normalized cell line '\
            'ids so it cannot be joined to drug name mappings'.format(source)
        assert 'DRUG_NAME' in d, \
            'Drug sensitivity data for source "{}" does not contain field "DRUG_NAME"'.format(source)

        # Load drug name mapping data
        if mappings is not None and entity.DRUG in mappings:
            d_rx = mappings[entity.DRUG]
        else:
            d_rx = get_entity_mapping(entity.DRUG)

        # Subset to given taxonomy
        assert drug_name_taxonomy in d_rx, \
            'Drug name taxonomy "{}" not found in mapping data'.format(drug_name_taxonomy)
        d_rx = d_rx[drug_name_taxonomy]

        # If the source is present in drug name mapping, join this mapping to the data
        if source in d_rx:
            d_rx = d_rx[source].reset_index().rename(columns={source: 'DRUG_NAME:SOURCE'})

            # Merge to raw data, resulting in two extra fields (DRUG_NAME:[SOURCE|MGDS])
            d = pd.merge(d, d_rx, left_on='DRUG_NAME', right_on='DRUG_NAME:SOURCE', how='left')

        # Otherwise, log a warning that drug names will not be available for this source
        else:
            logger.warning(
                'Drug sensitivity data for source "{}" contains no normalize drug name mapping'.format(source)
            )
            for c in ['DRUG_NAME:SOURCE', 'DRUG_NAME:MGDS']:
                d[c] = None
    return d


