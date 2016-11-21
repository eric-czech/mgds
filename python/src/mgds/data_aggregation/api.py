
from mgds.data_aggregation import database as db
from mgds.data_aggregation import source as src


def get_hugo_gene_ids():
    """ Get HUGO Gene Symbols"""
    d = db.load(src.HUGO_v1, db.IMPORT, 'gene-meta')
    d = d[~d['Approved Name'].str.lower().str.contains('symbol withdrawn|entry withdrawn')]
    d = d['Approved Symbol'].unique()
    return d
