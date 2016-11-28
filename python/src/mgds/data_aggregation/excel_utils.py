
import logging
import pandas as pd
logger = logging.getLogger(__name__)

# Pattern for identifying gene/cell line ids that may have been
# serialize by excel as dates
# Note that the "?:" denotes that these are non-capturing groups, something
# necessary to avoid pandas warnings with regular expressions
DATE_REGEX = '^\d{1,2}-(?:JAN|FEB|MAR|APR|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$'

# Excel date to gene name mapping (from https://www.biostars.org/p/183018/)
GENE_DATE_MAP = {
    "1-SEP": "SEPT1",
    "2-SEP": "SEPT2",
    "3-SEP": "SEPT3",
    "4-SEP": "SEPT4",
    "5-SEP": "SEPT5",
    "6-SEP": "SEPT6",
    "7-SEP": "SEPT7",
    "8-SEP": "SEPT8",
    "9-SEP": "SEPT9",
    "10-SEP": "SEPT10",
    "11-SEP": "SEPT11",
    "12-SEP": "SEPT12",
    "13-SEP": "SEPT13",
    "14-SEP": "SEPT14",
    "3-OCT": "POU5F1",
    "4-OCT": "POU5F1",
    "7-OCT": "POU3F2",
    "9-OCT": "POU3F4",
    "11-OCT": "POU2F3",
    "1-DEC": "DEC1",
    "1-MAR": "MARCH1",
    "2-MAR": "MARCH2",
    "3-MAR": "MARCH3",
    "4-MAR": "MARCH4",
    "5-MAR": "MARCH5",
    "6-MAR": "MARCH6",
    "7-MAR": "MARCH7",
    "8-MAR": "MARCH8",
    "9-MAR": "MARCH9",
    "10-MAR": "MARCH10",
    "11-MAR": "MARCH11",
    "1-FEB": "FEB1",
    "2-FEB": "FEB2",
    "5-FEB": "FEB5",
    "6-FEB": "FEB6",
    "7-FEB": "FEB7",
    "9-FEB": "FEB9",
    "10-FEB": "FEB10",
    "2-APR": "FAM215A",
    "3-APR": "ATRAID",
    "1-APR": "MAGEH1",
    "1-MAY": "PRKCD",
    "2-NOV": "CTGF",
    "1-NOV": "C11ORF40"

    # These are intentionally ignored since they have multiple possible inverse mappings
    # "6-OCT": "POU3F1",
    # "6-OCT": "SLC22A16",
    # "2-OCT": "POU2F2",
    # "2-OCT": "SLC22A2",
}

NON_INVERTIBLE_DATES = ["6-OCT", "2-OCT"]


def is_excel_date(ids):
    """
    Returns boolean vector for given strings (usually cell line or gene ids) indicating whether or
    not each string conforms to a date pattern like "D-MMM" (eg 1-SEP).

    This is apparently a big problem with bioinformatics datasets transferred as excel files.  See [1] and [2]
    for more details.
    [1] https://www.biostars.org/p/183018/
    [2] http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-5-80
    :param ids: Series of string IDs
    :return: Boolean Series
    """
    return ids.str.upper().str.strip().str.contains(DATE_REGEX)


def convert_gene_ids(ids):
    r = []
    is_date = is_excel_date(ids)
    for i, id in enumerate(ids):
        if not is_date[i]:
            r.append(id)
        else:
            if id in NON_INVERTIBLE_DATES:
                logger.warning('Gene ID "{}" has multiple known inverse mappings so its value will be returned as null')
                r.append(None)
            else:
                r.append(GENE_DATE_MAP.get(id.upper().strip(), None))
    assert len(r) == len(ids)
    return pd.Series(r, index=ids.index)


def get_gene_conversions(original_ids, converted_ids):
    mask = (original_ids != converted_ids).values
    v1 = original_ids[mask]
    v1.name = 'ORIGINAL_GENE_ID'
    v2 = converted_ids[mask]
    v2.name = 'CONVERTED_GENE_ID'
    return pd.concat([v1, v2], axis=1)
