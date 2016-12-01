
import pandas as pd
from mgds.data_aggregation import database as db
from mgds.data_aggregation import source as src
from mgds.data_aggregation import entity


def get_hugo_gene_ids():
    """ Get HUGO Gene Symbols"""
    d = db.load(src.HUGO_v1, db.IMPORT, 'gene-meta')
    d = d[~d['Approved Name'].str.lower().str.contains('symbol withdrawn|entry withdrawn')]
    d = d['Approved Symbol'].unique()
    return d


def get_entity_mapping(entity_type):
    if entity_type == entity.CELL_LINE:
        return db.load(src.MGDS_v1, db.ENTITY, 'cellline-ids-by-typ')
    else:
        raise ValueError('Mappings for entity type "{}" not yet supported'.format(entity_type))


def get_raw_genomic_data(source, data_type, cell_line_taxonomy='COMMON', mappings=None):

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

        # Mapping data now has MGDS_ID and source specific identifiers as columns, so first rename the
        # source specific id field to the same name it should join to in the original data
        d_id = d_id.rename(columns={data_type: c_cl_id, 'MGDS_ID': 'CELL_LINE_ID:MGDS'})

        d = pd.merge(d, d_id, on=c_cl_id, how='left')

    return d


