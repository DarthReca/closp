CORINE_TO_DW = {
    # CORINE 'Artificial Surfaces' mapped to DW 'Built', 'Bare', or 'Grass'
    "Continuous urban fabric": "Built",
    "Discontinuous urban fabric": "Built",
    "Industrial or commercial units": "Built",
    "Road and rail networks and associated land": "Built",
    "Port areas": "Built",
    "Airports": "Built",
    "Mineral extraction sites": "Bare",
    "Dump sites": "Bare",
    "Construction sites": "Bare",
    "Green urban areas": "Grass",
    "Sport and leisure facilities": "Grass",
    # CORINE 'Agricultural Areas' mapped to DW 'Crops', 'Flooded vegetation', or 'Grass'
    "Non-irrigated arable land": "Crops",
    "Permanently irrigated land": "Crops",
    "Rice fields": "Flooded vegetation",  # Specific case of a crop that is flooded
    "Vineyards": "Crops",
    "Fruit trees and berry plantations": "Crops",  # Cultivated trees are considered crops
    "Olive groves": "Crops",  # Cultivated trees are considered crops
    "Pastures": "Grass",
    "Annual crops associated with permanent crops": "Crops",
    "Complex cultivation patterns": "Crops",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Crops",
    "Agro-forestry areas": "Trees",  # A mix, but 'Trees' is a strong component
    # CORINE 'Forest and Semi-Natural Areas' mapped to DW 'Trees', 'Grass', 'Shrub and Scrub', 'Bare', or 'Snow and Ice'
    "Broad-leaved forest": "Trees",
    "Coniferous forest": "Trees",
    "Mixed forest": "Trees",
    "Natural grassland": "Grass",
    "Moors and heathland": "Shrub and Scrub",
    "Sclerophyllous vegetation": "Shrub and Scrub",
    "Transitional woodland/shrub": "Shrub and Scrub",
    "Beaches, dunes, sands": "Bare",
    "Bare rock": "Bare",
    "Sparsely vegetated areas": "Bare",
    "Burnt areas": "Burned Area",
    "Glaciers and perpetual snow": "Snow and Ice",
    # CORINE 'Wetlands' mapped to DW 'Flooded vegetation', 'Bare', or 'Water'
    "Inland marshes": "Flooded vegetation",
    "Peatbogs": "Flooded vegetation",
    "Salt marshes": "Flooded vegetation",
    "Salines": "Bare",  # Often bare ground for salt evaporation
    "Intertidal flats": "Bare",  # Bare when tide is out
    # CORINE 'Water Bodies' mapped to DW 'Water'
    "Water courses": "Water",
    "Water bodies": "Water",
    "Coastal lagoons": "Water",
    "Estuaries": "Water",
    "Sea and ocean": "Water",
}
