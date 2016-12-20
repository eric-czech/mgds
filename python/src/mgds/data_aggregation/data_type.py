
GENE_EXPRESSION = 'gene-expression'
GENE_COPY_NUMBER = 'gene-copy-number'
GENE_RNA_SEQ = 'gene-rna-seq'
GENE_EXOME_SEQ = 'gene-exome-seq'
GENE_RPPA = 'gene-rppa'
GENE_METHYLATION = 'gene-methylation'
DRUG_SENSITIVITY = 'drug-sensitivity'
CELLLINE_META = 'cellline-meta'

PRETTY_NAMES = {
    GENE_EXOME_SEQ: 'Exome Seq',
    GENE_RNA_SEQ: 'RNA-Seq',
    GENE_COPY_NUMBER: 'CNV',
    GENE_EXPRESSION: 'Expression',
    GENE_METHYLATION: 'Methylation',
    CELLLINE_META: 'Cell Line Metadata',
    DRUG_SENSITIVITY: 'Drug Sensitivity'
}


def add_normalized_modifier(data_type):
    return '{}-normalized'.format(data_type)


def add_putative_modifier(data_type):
    return '{}-putative'.format(data_type)


def get_pretty_name(data_type):
    return PRETTY_NAMES.get(data_type, ' '.join([v.title() for v in data_type.split('-')]))
