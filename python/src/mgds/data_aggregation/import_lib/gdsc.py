

def get_known_drug_mappings():
    """ Returns a set of non-obvious drug name conversions

    At TOW, these conversions were sourced from the following sources:
        1. GDSC file 'GDSC-CCLE-CTRP_conversion.xlsx'
            - Note that while this file contains ~80 mappings, only the ones that wouldn't be provided by a
                case-insensitive match after removing non-alphaunmeric characters are returned here.  In other
                words a mapping like GDSC = Erlotinib, CTRP = erlotinib will be ignored while one like
                GDSC = Mitomycin C, CTRP = mitomycin will be included.

    Also, the mapping is constructed in such a way to facilitate conversion to a later-stage name of any drug.  This
    means that if a drug has multiple names like "ABT-263" and "Navitoclax", then the mapping for this drug will
    be keyed by the laboratory name "ABT-263" with a value equal to "Navitoclax".  This will favor converting
    everything to the most friendly name possible.
    """
    return {
        # GDSC -> CCLE -> CTRP spreadsheet mappings
        'Mitomycin C': 'Mitomycin',
        'Obatoclax Mesylate': 'Obatoclax',
        'Bleomycin A2': 'Bleomycin',
        'SN-38': 'Camptothecin',
        'Cytarabine Hydrochloride': 'Cytarabine',
        'JNJ-26854165': 'Serdemetan',
        'AZD-2281': 'Olaparib'
    }
