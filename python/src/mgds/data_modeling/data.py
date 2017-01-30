
import pandas as pd
from py_utils.pandas_utils import one_value
from py_utils import collection_utils
from mgds.data_aggregation import api
from mgds.data_aggregation import source as src
from mgds.data_aggregation import entity
from mgds.data_aggregation import data_type as dtyp
import logging
logger = logging.getLogger(__name__)


def get_feature_datasets(dataset_info):
    """
    Generate tabular feature data frame with multiple genomic data types
    :param dataset_info: Source + data type pairs (generally subset of api.get_genomic_data_availability)
    :return:
    """

    # Pre-load entity mappings needed to create raw feature sets
    entity_mappings = {
        entity.CELL_LINE: api.get_entity_mapping(entity.CELL_LINE),
        entity.PRIMARY_SITE: api.get_entity_mapping(entity.PRIMARY_SITE)
    }

    d = []
    m = []
    for (source, data_type) in dataset_info:
        logger.info('Unpacking source "{}", data set "{}"'.format(source, data_type))

        c_val = 'INDICATOR' if data_type in [dtyp.GENE_EXOME_SEQ] else 'VALUE'
        c_col = 'GENE_ID:HGNC' if data_type != dtyp.DRUG_SENSITIVITY else 'DRUG_NAME:MGDS'
        agg_func = 'mean' if data_type in [dtyp.GENE_EXOME_SEQ, dtyp.DRUG_SENSITIVITY] else one_value

        # Leaving this out due to lack of clarity on encoding (what is "-1,-1,-,-"?)
        if source == src.GDSC_v2 and data_type in [dtyp.GENE_COPY_NUMBER]:
            continue

        # These were both verified to show that the values associated with all duplicate MGDS cell lines
        # and genes are identical (so the mean will be used, but there is no variance)
        if source == src.CCLE_v1 and data_type in [dtyp.GENE_COPY_NUMBER, dtyp.GENE_EXPRESSION]:
            agg_func = 'mean'

        # If sensitivity data is being loaded, determine which measurement is preferred for this
        # particular source and assign that as the value to aggregate
        if data_type == dtyp.DRUG_SENSITIVITY:
            drug_measures = api.get_preferred_drug_sensitivity_measurements()
            if source not in drug_measures:
                raise ValueError(
                    'Source "{}" does not have a preferred sensitivity measurement assigned yet'
                    .format(source)
                )
            c_val = drug_measures[source]

        df = api.get_raw_genomic_data(source, data_type, mappings=entity_mappings)
        df['INDICATOR'] = 1

        if c_val == 'VALUE' and 'VALUE' not in df:
            c_val = 'VALUE_MEAN'

        na_cell_line = df['CELL_LINE_ID:MGDS'].isnull()
        if na_cell_line.any():
            df = collection_utils.subset(
                df, lambda data: data[~na_cell_line],
                subset_op='Removing records with null MGDS cell line ID',
                log=logger
            )

        c_primary_site = 'PRIMARY_SITE:MGDS:{}:{}'.format(source, data_type)
        m.append((
            df
            .filter(items=['CELL_LINE_ID:MGDS', 'PRIMARY_SITE:MGDS'])
            .drop_duplicates()
            .rename(columns={'PRIMARY_SITE:MGDS': c_primary_site})
            .set_index('CELL_LINE_ID:MGDS')
        ))

        # Pivot out of long form into wide form
        df = df.drop('PRIMARY_SITE:MGDS', axis=1).pivot_table(
            index='CELL_LINE_ID:MGDS',
            columns=c_col,
            values=c_val,
            aggfunc=agg_func
        )
        df.columns = pd.MultiIndex.from_tuples([(source, data_type, c) for c in df])
        d.append(df)

    def get_site(r):
        rd = r.dropna()
        if rd.nunique() > 1:
            raise ValueError('Found primary site combinations with conflicting values; row = "{}"'.format(r))
        return rd.iloc[0] if len(rd) > 0 else None
    primary_sites = pd.concat(m, axis=1).apply(get_site, axis=1).fillna('NULL')

    d = pd.concat(d, axis=1)
    d.index = d.index.set_names('CELL_LINE_ID:MGDS')
    d['PRIMARY_SITE:MGDS'] = d.index.to_series().map(primary_sites)
    d = d.set_index('PRIMARY_SITE:MGDS', append=True)

    return d
