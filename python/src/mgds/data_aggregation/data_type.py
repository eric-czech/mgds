
GENE_EXPRESSION = 'gene-expression'
GENE_COPY_NUMBER = 'gene-copy-number'
GENE_RNA_SEQ = 'gene-rna-seq'
GENE_EXOME_SEQ = 'gene-exome-seq'
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


def get_pretty_name(data_type):
    return PRETTY_NAMES.get(data_type, ' '.join([v.title() for v in data_type.split('-')]))
