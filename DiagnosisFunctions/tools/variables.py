def getVariableGroups(df=None):
    
    if df == None:
        # Standard variable groups
        diagnosis_variables = ['diagnosis_acne','diagnosis_actinic_keratosis','diagnosis_psoriasis','diagnosis_seborrheic_dermatitis','diagnosis_viral_warts','diagnosis_vitiligo']
        area_variables = ['area_acral_distribution','area_exposed_areas','area_extensor_sites','area_seborrheic_region']
        characteristics_variables = ['scale','plaque','pustule','patch','papule','dermatoglyph_disruption','open_comedo']
    else:
        # Custom variable groups based on dataframe
        diagnosis_variables = [x for x in df.columns if x.startswith('diagnosis_')]
        area_variables = [x for x in df.columns if x.startswith('area_')]
        characteristics_variables = [x for x in df.columns if not (x in diagnosis_variables or x in area_variables)]
    
    return diagnosis_variables, area_variables, characteristics_variables