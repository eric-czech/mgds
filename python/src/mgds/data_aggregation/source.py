

HUGO_v1 = 'hugo_v1'
CCLE_v1 = 'ccle_v1'
CTD_v1 = 'ctd_v1'
CTD_v2 = 'ctd_v2'
NCI60_v1 = 'nci60_v1'
NCI60_v2 = 'nci60_v2'
NCIDREAM_v1 = 'ncidream_v1'
# GDSC_v1 = 'gdsc_v1'
GDSC_v2 = 'gdsc_v2'
GTEX_v1 = 'gtex_v1'
BIOC_v1 = 'bioc_v1'
MGDS_v1 = 'mgds_v1'
TCGA_v1 = 'tcga_v1'


def get_pretty_name(source):
    return source.split('_')[0].upper()
