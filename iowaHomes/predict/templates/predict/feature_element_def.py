import datetime

elements = {

    "MSSubClass":
         {"name": "MSSubClass",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "20", "text": "1-STORY 1946 & NEWER ALL STYLES"},
              {"value": "30", "text": "1-STORY 1945 & OLDER"},
              {"value": "40", "text": "1-STORY W/FINISHED ATTIC ALL AGES"},
              {"value": "45", "text": "1-1/2 STORY - UNFINISHED ALL AGES"},
              {"value": "50", "text": "1-1/2 STORY FINISHED ALL AGES"},
              {"value": "60", "text": "2-STORY 1946 & NEWER"},
              {"value": "70", "text": "2-STORY 1945 & OLDER"},
              {"value": "75", "text": "2-1/2 STORY ALL AGES"},
              {"value": "80", "text": "SPLIT OR MULTI-LEVEL"},
              {"value": "85", "text": "SPLIT FOYER"},
              {"value": "90", "text": "DUPLEX - ALL STYLES AND AGES"},
              {"value": "120", "text": "1-STORY PUD (Planned Unit Development) - 1946 & NEWER"},
              {"value": "150", "text": "1-1/2 STORY PUD - ALL AGES"},
              {"value": "160", "text": "2-STORY PUD - 1946 & NEWER"},
              {"value": "180", "text": "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER"},
              {"value": "190", "text": "2 FAMILY CONVERSION - ALL STYLES AND AGES"},

          ],
          "labelText": "Zoning Classification",
          "tooltip": "Identifies the type of dwelling involved in the sale."},

    "MSZoning":
         {"name": "MSZoning",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "A", "text": "Agriculture"},
              {"value": "C", "text": "Commercial"},
              {"value": "FV", "text": "Floating Village"},
              {"value": "I", "text": "Industrial"},
              {"value": "RH", "text": "Residential - High Density"},
              {"value": "RM", "text": "Residential - Medium Density"},
              {"value": "RP", "text": "Residential - Low Density Park"},
              {"value": "RL", "text": "Residential - Low Density"},
          ],
          "labelText": "Zoning Classification",
          "tooltip": "Identifies the general zoning classification"},

    "LotFrontage":
         {"name": "LotFrontage",
          "type": "slider",
          "min": '10',
          "max": '350',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Lot Frontage",
          "tooltip": "Linear feet of street connected to property"
          },

    "LotArea":
         {"name": "LotArea",
          "type": "slider",
          "min": '0',
          "max": '40000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Lot Area",
          "tooltip": "Lot size in square feet"
          },

    "Street":
         {"name": "Street",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Grvl", "text": "Gravel"},
              {"value": "Pave", "text": "Paved"},

          ],
          "labelText": "Street",
          "tooltip": "Type of road access to property"},

    "Alley":
         {"name": "Alley",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Grvl", "text": "Gravel"},
              {"value": "Pave", "text": "Paved"},
              {"value": "NA", "text": "No alley access"},

          ],
          "labelText": "Alley",
          "tooltip": "Type of alley access to property"},

    "LotShape":
         {"name": "LotShape",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Reg", "text": "Regular"},
              {"value": "IR1", "text": "Slightly irregular"},
              {"value": "IR2", "text": "Moderately Irregular"},
              {"value": "IR3", "text": "Irregular"},

          ],
          "labelText": "Lot Shape",
          "tooltip": "General shape of property"},

    "LandContour":
         {"name": "LandContour",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Lvl", "text": "Near Flat/Level"},
              {"value": "Bnk", "text": "Banked - Quick and significant rise from street grade to building"},
              {"value": "HLS", "text": "Hillside - Significant slope from side to side"},
              {"value": "Low", "text": "Depression"},

          ],
          "labelText": "Land Contour",
          "tooltip": "Flatness of the property"},

    "Utilities":
         {"name": "Utilities",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "AllPub", "text": "All public Utilities (E,G,W,& S)"},
              {"value": "NoSewr", "text": "Electricity, Gas, and Water (Septic Tank)"},
              {"value": "NoSeWa", "text": "Electricity and Gas Only"},
              {"value": "ELO", "text": "Electricity only"},

          ],
          "labelText": "Utilities",
          "tooltip": "Type of utilities available"},

    "LotConfig":
         {"name": "LotConfig",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Inside", "text": "Inside lot"},
              {"value": "Corner", "text": "Corner lot)"},
              {"value": "CulDSac", "text": "Cul-de-sac"},
              {"value": "FR2", "text": "Frontage on 2 sides of property"},
              {"value": "FR3", "text": "Frontage on 3 sides of property"},

          ],
          "labelText": "Lot Config",
          "tooltip": "Lot configuration"},

    "LandSlope":
         {"name": "LandSlope",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Gtl", "text": "Gentle slope"},
              {"value": "Mod", "text": "Moderate Slope)"},
              {"value": "Sev", "text": "Severe Slope"},

          ],
          "labelText": "Land Slope",
          "tooltip": "Slope of property"},

    "Neighborhood":
         {"name": "Neighborhood",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Blmngtn", "text": "Bloomington Heights"},
              {"value": "Blueste", "text": "Bluestem)"},
              {"value": "SBrDaleev", "text": "Briardale"},
              {"value": "BrkSide", "text": "Brookside"},
              {"value": "ClearCr", "text": "Clear Creek"},
              {"value": "CollgCr", "text": "College Creek"},
              {"value": "Crawfor", "text": "Crawford"},
              {"value": "Edwards", "text": "Edwards"},
              {"value": "Gilbert", "text": "Gilbert"},
              {"value": "IDOTRR", "text": "Iowa DOT and Rail Road"},
              {"value": "MeadowV", "text": "Meadow Village"},
              {"value": "Mitchel", "text": "Mitchell"},
              {"value": "Names", "text": "North Ames"},
              {"value": "NoRidge", "text": "Northridge"},
              {"value": "NPkVill", "text": "Northpark Villa"},
              {"value": "NridgHt", "text": "Northridge Heights"},
              {"value": "NWAmes", "text": "Northwest Ames"},
              {"value": "OldTown", "text": "Old Town"},
              {"value": "SWISU", "text": "South & West of Iowa State University"},
              {"value": "Sawyer", "text": "Sawyer"},
              {"value": "SawyerW", "text": "Sawyer West"},
              {"value": "Somerst", "text": "Somerset"},
              {"value": "StoneBr", "text": "Stone Brook"},
              {"value": "Timber", "text": "Timberland"},
              {"value": "Veenker", "text": "Veenker"},

          ],
          "labelText": "Neighborhood",
          "tooltip": "Physical locations within Ames city limits"},

    "Condition1":
         {"name": "Condition1",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Artery", "text": "Adjacent to arterial street"},
              {"value": "Feedr", "text": "Adjacent to feeder street)"},
              {"value": "Norm", "text": "Normal"},
              {"value": "RRNn", "text": "Within 200' of North-South Railroad"},
              {"value": "RRAn", "text": "Adjacent to North-South Railroad"},
              {"value": "PosN", "text": "Near positive off-site feature--park, greenbelt, etc."},
              {"value": "PosA", "text": "Adjacent to postive off-site feature"},
              {"value": "RRNe", "text": "Within 200' of East-West Railroad"},
              {"value": "RRAe", "text": "Adjacent to East-West Railroad"},
          ],
          "labelText": "Condition1",
          "tooltip": "Proximity to various conditions"},

    "Condition2":
         {"name": "Condition2",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Artery", "text": "Adjacent to arterial street"},
              {"value": "Feedr", "text": "Adjacent to feeder street)"},
              {"value": "Norm", "text": "Normal"},
              {"value": "RRNn", "text": "Within 200' of North-South Railroad"},
              {"value": "RRAn", "text": "Adjacent to North-South Railroad"},
              {"value": "PosN", "text": "Near positive off-site feature--park, greenbelt, etc."},
              {"value": "PosA", "text": "Adjacent to postive off-site feature"},
              {"value": "RRNe", "text": "Within 200' of East-West Railroad"},
              {"value": "RRAe", "text": "Adjacent to East-West Railroad"},
          ],
          "labelText": "Condition2",
          "tooltip": "Proximity to various conditions (if more than one is present)"},

    "BldgType":
         {"name": "BldgType",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "1Fam", "text": "Single-family Detached"},
              {"value": "2FmCon", "text": "Two-family Conversion; originally built as one-family dwelling)"},
              {"value": "Duplx", "text": "Duplex"},
              {"value": "TwnhsE", "text": "Townhouse End Unit"},
              {"value": "TwnhsI", "text": "Townhouse Inside Unit"},

          ],
          "labelText": "Dwelling",
          "tooltip": "Type of dwelling"},

    "HouseStyle":
         {"name": "HouseStyle",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "1Story", "text": "One story"},
              {"value": "1.5Fin", "text": "One and one-half story: 2nd level finished)"},
              {"value": "1.5Unf", "text": "One and one-half story: 2nd level unfinished"},
              {"value": "2Story", "text": "Two story"},
              {"value": "2.5Fin", "text": "Two and one-half story: 2nd level finished"},
              {"value": "2.5Unf", "text": "Two and one-half story: 2nd level unfinished"},
              {"value": "SFoyer", "text": "Split Foyer"},
              {"value": "SLvl", "text": "Split Level"},
          ],
          "labelText": "House Style",
          "tooltip": "Style of dwelling"},

    "OverallQual":
         {"name": "OverallQual",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "10", "text": "Very Excellent"},
              {"value": "9", "text": "Excellent"},
              {"value": "8", "text": "Very Good"},
              {"value": "7", "text": "Good"},
              {"value": "6", "text": "Above Average"},
              {"value": "5", "text": "Average"},
              {"value": "4", "text": "Below Average"},
              {"value": "3", "text": "Fair"},
              {"value": "2", "text": "Poor"},
              {"value": "1", "text": "Very Poor"},
          ],
          "labelText": "Overall Quality",
          "tooltip": "Rates the overall material and finish of the house"},

    "OverallCond":
         {"name": "OverallCond",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "10", "text": "Very Excellent"},
              {"value": "9", "text": "Excellent"},
              {"value": "8", "text": "Very Good"},
              {"value": "7", "text": "Good"},
              {"value": "6", "text": "Above Average"},
              {"value": "5", "text": "Average"},
              {"value": "4", "text": "Below Average"},
              {"value": "3", "text": "Fair"},
              {"value": "2", "text": "Poor"},
              {"value": "1", "text": "Very Poor"},
          ],
          "labelText": "Overall Condition",
          "tooltip": " Rates the overall condition of the house"},

    "YearBuilt":
         {"name": "YearBuilt",
          "type": "slider",
          "min": '1870',
          "max": str(datetime.datetime.today().year),
          "step": '1',
          "value": str(datetime.datetime.today().year),
          "unit": "",
          "labelText": "Year Built",
          "tooltip": "Original construction date"
          },

    "YearRemodAdd":
         {"name": "YearRemodAdd",
          "type": "slider",
          "min": '1870',
          "max": str(datetime.datetime.today().year),
          "step": '1',
          "value": str(datetime.datetime.today().year),
          "unit": "",
          "labelText": "Remodel/Addition Year",
          "tooltip": "Remodel date (same as construction date if no remodeling or additions)"
          },

    "RoofStyle":
         {"name": "RoofStyle",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Flat", "text": "Flat"},
              {"value": "Gable", "text": "Gable"},
              {"value": "Gambrel", "text": "Gabrel (Barn)"},
              {"value": "Hip", "text": "Hip"},
              {"value": "Mansard", "text": "Mansard"},
              {"value": "Shed", "text": "Shed"},

          ],
          "labelText": "Roof Style",
          "tooltip": "Type of roof"},

    "RoofMatl":
         {"name": "RoofMatl",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "ClyTile", "text": "Clay or Tile"},
              {"value": "CompShg", "text": "Standard (Composite) Shingle"},
              {"value": "Membran", "text": "Membrane"},
              {"value": "Metal", "text": "Metal"},
              {"value": "Roll", "text": "Roll"},
              {"value": "Tar&Grv", "text": "Gravel & Tar"},
              {"value": "WdShake", "text": "Wood Shakes"},
              {"value": "WdShngl", "text": "Wood Shingles"},

          ],
          "labelText": "Roof Material",
          "tooltip": ""},

    "Exterior1st":
         {"name": "Exterior1st",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "AsbShng", "text": "Asbestos Shingles"},
              {"value": "AsphShn", "text": "Asphalt Shingles"},
              {"value": "BrkComm", "text": "Brick Common"},
              {"value": "BrkFace", "text": "Brick Face"},
              {"value": "CBlock", "text": "Cinder Block"},
              {"value": "CemntBd", "text": "Cement Board"},
              {"value": "HdBoard", "text": "Hard Board"},
              {"value": "ImStucc", "text": "Imitation Stucco"},
              {"value": "MetalSd", "text": "Metal Siding"},
              {"value": "Other", "text": "Other"},
              {"value": "Plywood", "text": "Plywood"},
              {"value": "PreCast", "text": "PreCast"},
              {"value": "Stone", "text": "Stone"},
              {"value": "Stucco", "text": "Stucco"},
              {"value": "VinylSd", "text": "Vinyl Siding"},
              {"value": "Wd Sdng", "text": "Wood Siding"},
              {"value": "WdShing", "text": "Wood Shingles"},

          ],
          "labelText": "Exterior covering",
          "tooltip": "Exterior covering on house"},

    "Exterior2nd":
         {"name": "Exterior2nd",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "AsbShng", "text": "Asbestos Shingles"},
              {"value": "AsphShn", "text": "Asphalt Shingles"},
              {"value": "BrkComm", "text": "Brick Common"},
              {"value": "BrkFace", "text": "Brick Face"},
              {"value": "CBlock", "text": "Cinder Block"},
              {"value": "CemntBd", "text": "Cement Board"},
              {"value": "HdBoard", "text": "Hard Board"},
              {"value": "ImStucc", "text": "Imitation Stucco"},
              {"value": "MetalSd", "text": "Metal Siding"},
              {"value": "Other", "text": "Other"},
              {"value": "Plywood", "text": "Plywood"},
              {"value": "PreCast", "text": "PreCast"},
              {"value": "Stone", "text": "Stone"},
              {"value": "Stucco", "text": "Stucco"},
              {"value": "VinylSd", "text": "Vinyl Siding"},
              {"value": "Wd Sdng", "text": "Wood Siding"},
              {"value": "WdShing", "text": "Wood Shingles"},

          ],
          "labelText": "Exterior covering",
          "tooltip": "Exterior covering on house (if more than one material)"},

    "MasVnrType":
         {"name": "MasVnrType",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "BrkCmn", "text": "Brick Common"},
              {"value": "BrkFace", "text": "Brick Face"},
              {"value": "CBlock", "text": "Cinder Block"},
              {"value": "None", "text": "None"},
              {"value": "Stone", "text": "Stone"},

          ],
          "labelText": "Masonry Veneer Type",
          "tooltip": ""},

    "MasVnrArea":
         {"name": "MasVnrArea",
          "type": "slider",
          "min": '0',
          "max": '5000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Masonry Area",
          "tooltip": " Masonry veneer area in square feet"
          },

    "ExterQual":
         {"name": "ExterQual",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Average/Typical"},
              {"value": "Fa", "text": "Fair"},
              {"value": "Po", "text": "Poor"},
          ],
          "labelText": "Exterior Quality",
          "tooltip": "Evaluates the quality of the material on the exterior"
          },

    "ExterCond":
         {"name": "ExterCond",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Average/Typical"},
              {"value": "Fa", "text": "Fair"},
              {"value": "Po", "text": "Poor"},
          ],
          "labelText": "Exterior Quality",
          "tooltip": "Evaluates the present condition of the material on the exterior"
          },

    "Foundation":
         {"name": "Foundation",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "BrkTil", "text": "Brick & Tile"},
              {"value": "CBlock", "text": "Cinder Block"},
              {"value": "PConc", "text": "Poured Contrete"},
              {"value": "Slab", "text": "Slab"},
              {"value": "Stone", "text": "Stone"},
              {"value": "Wood", "text": "Wood"},
          ],
          "labelText": "Foundation",
          "tooltip": "Type of foundation"
          },

    "BsmtQual":
         {"name": "BsmtQual",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent (100+ inches)"},
              {"value": "Gd", "text": "Good (90-99 inches)"},
              {"value": "TA", "text": "Typical (80-89 inches)"},
              {"value": "Fa", "text": "Fair (70-79 inches)"},
              {"value": "Po", "text": "Poor (<70 inches)"},
              {"value": "NA", "text": "No Basement"},
          ],
          "labelText": "Basement Height",
          "tooltip": "Evaluates the height of the basement"},

    "BsmtCond":
         {"name": "BsmtCond",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Typical - slight dampness allowed"},
              {"value": "Fa", "text": "Fair - dampness or some cracking or settling"},
              {"value": "Po", "text": "Poor - Severe cracking, settling, or wetness"},
              {"value": "NA", "text": "No Basement"},
          ],
          "labelText": "Basement Height",
          "tooltip": "Evaluates the general condition of the basement"},

    "BsmtExposure":
         {"name": "BsmtExposure",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Gd", "text": "Good Exposure"},
              {"value": "Av", "text": "Average Exposure (split levels or foyers typically score average or above)"},
              {"value": "Mn", "text": "Mimimum Exposure"},
              {"value": "No", "text": "No Exposure"},
              {"value": "NA", "text": "No Basement"},
          ],
          "labelText": "Basement Exposure",
          "tooltip": "Refers to walkout or garden level walls"},

    "BsmtFinType1":
         {"name": "BsmtFinType1",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "GLQ", "text": "Good Living Quarters"},
              {"value": "ALQ", "text": "Average Living Quarters"},
              {"value": "BLQ", "text": "Below Average Living Quarters"},
              {"value": "Rec", "text": "Average Rec Room"},
              {"value": "LwQ", "text": "Low Quality"},
              {"value": "Unf", "text": "Unfinshed"},
              {"value": "NA", "text": "No Basement"},
          ],
          "labelText": "Basement Finished Area Rating 1",
          "tooltip": " Rating of basement finished area"},

    "BsmtFinSF1":
         {"name": "BsmtFinSF1",
          "type": "slider",
          "min": '0',
          "max": '2000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Finished Basment Area",
          "tooltip": "Type 1 finished square feet"
          },

    "BsmtFinType2":
         {"name": "BsmtFinType2",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "GLQ", "text": "Good Living Quarters"},
              {"value": "ALQ", "text": "Average Living Quarters"},
              {"value": "BLQ", "text": "Below Average Living Quarters"},
              {"value": "Rec", "text": "Average Rec Room"},
              {"value": "LwQ", "text": "Low Quality"},
              {"value": "Unf", "text": "Unfinshed"},
              {"value": "NA", "text": "No Basement"},
          ],
          "labelText": "Basement Finished Area Rating 2",
          "tooltip": " Rating of basement finished area"},

    "BsmtFinSF2":
         {"name": "BsmtFinSF2",
          "type": "slider",
          "min": '0',
          "max": '2000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Finished Basment Area",
          "tooltip": "Type 2 finished square feet"
          },

    "BsmtUnfSF":
         {"name": "BsmtUnfSF",
          "type": "slider",
          "min": '0',
          "max": '2500',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Basment Unfinished Area",
          "tooltip": "Unfinished square feet of basement area"
          },

    "TotalBsmtSF":
         {"name": "TotalBsmtSF",
          "type": "slider",
          "min": '0',
          "max": '8000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Basement Area",
          "tooltip": "Total square feet of basement area"
          },

    "Heating":
         {"name": "Heating",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Floor", "text": "Floor Furnace"},
              {"value": "GasA", "text": "Gas forced warm air furnace"},
              {"value": "GasW", "text": "Gas hot water or steam heat"},
              {"value": "Grav", "text": "Gravity furnace"},
              {"value": "OthW", "text": "Hot water or steam heat other than gas"},
              {"value": "Wall", "text": "Wall furnace"},
          ],
          "labelText": "Heating Quality",
          "tooltip": "Type of heating"},

    "HeatingQC":
         {"name": "HeatingQC",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Average/Typical"},
              {"value": "Fa", "text": "Fair"},
              {"value": "Po", "text": "Poor"},
          ],
          "labelText": "Heating Quality",
          "tooltip": "Heating quality and condition"},

    "CentralAir":
         {"name": "CentralAir",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Y", "text": "Yes"},
              {"value": "N", "text": "No"},
          ],
          "labelText": "Central Air",
          "tooltip": "Central air conditioning"},

    "Electrical":
         {"name": "Electrical",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "SBrkr", "text": "Standard Circuit Breakers & Romex"},
              {"value": "FuseA", "text": "Fuse Box over 60 AMP and all Romex wiring (Average)"},
              {"value": "FuseF", "text": "60 AMP Fuse Box and mostly Romex wiring (Fair)"},
              {"value": "FuseP", "text": "60 AMP Fuse Box and mostly knob & tube wiring (poor)"},
              {"value": "Mix", "text": "Mixed"},
          ],
          "labelText": "Heating quality",
          "tooltip": "Electrical system"},

    "1stFlrSF":
         {"name": "1stFlrSF",
          "type": "slider",
          "min": '0',
          "max": '3500',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "First Floor Area",
          "tooltip": "First Floor area in square feet"
          },

    "2ndFlrSF":
         {"name": "2ndFlrSF",
          "type": "slider",
          "min": '0',
          "max": '2000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Second Floor Area",
          "tooltip": "Second Floor area in square feet"
          },

    "LowQualFinSF":
         {"name": "LowQualFinSF",
          "type": "slider",
          "min": '0',
          "max": '600',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Low Quality Finish Area",
          "tooltip": " Low quality finished square feet (all floors)"
          },

    "GrLivArea":
         {"name": "GrLivArea",
          "type": "slider",
          "min": '0',
          "max": '3700',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Above Grade Area",
          "tooltip": "Above grade (ground) living area square feet"
          },

    "BsmtFullBath":
         {"name": "BsmtFullBath",
          "type": "slider",
          "min": '0',
          "max": '10',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Basement Full Bathrooms",
          "tooltip": "Amount of full basement bathrooms"
          },

    "BsmtHalfBath":
         {"name": "BsmtHalfBath",
          "type": "slider",
          "min": '0',
          "max": '5',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Basement Half Bathrooms",
          "tooltip": ""
          },

    "FullBath":
         {"name": "FullBath",
          "type": "slider",
          "min": '0',
          "max": '5',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Full Bathrooms",
          "tooltip": "Full bathrooms above grade"
          },

    "HalfBath":
         {"name": "HalfBath",
          "type": "slider",
          "min": '0',
          "max": '5',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Half Bathrooms",
          "tooltip": "Half bathrooms above grade"
          },

    "BedroomAbvGr":
         {"name": "BedroomAbvGr",
          "type": "slider",
          "min": '0',
          "max": '10',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Bedrooms",
          "tooltip": "Bedrooms above grade (does NOT include basement bedrooms)"
          },

    "KitchenAbvGr":
         {"name": "KitchenAbvGr",
          "type": "slider",
          "min": '0',
          "max": '5',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Kitchens",
          "tooltip": "Kitchens above grade"
          },

    "KitchenQual":
         {"name": "KitchenQual",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Average/Typical"},
              {"value": "Fa", "text": "Fair"},
              {"value": "Po", "text": "Poor"},
          ],
          "labelText": "Kitchen Quality",
          "tooltip": "Kitchen quality rating"},

    "TotRmsAbvGrd":
         {"name": "TotRmsAbvGrd",
          "type": "slider",
          "min": '0',
          "max": '20',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Total Rooms",
          "tooltip": "Total rooms above grade (does not include bathrooms)"
          },

    "Functional":
         {"name": "Functional",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Typ", "text": "Typical Functionality"},
              {"value": "Min1", "text": "Minor Deductions 1"},
              {"value": "Min2", "text": "Minor Deductions 2"},
              {"value": "Mod", "text": "Moderate Deductions"},
              {"value": "Maj1", "text": "Major Deductions 1"},
              {"value": "Maj2", "text": "Major Deductions 2"},
              {"value": "Sev", "text": "Severely Damaged"},
              {"value": "Sal", "text": "Salvage only"},
          ],
          "labelText": "Functionality Rating",
          "tooltip": "Home functionality (Assume typical unless deductions are warranted)"},

    "Fireplaces":
         {"name": "Fireplaces",
          "type": "slider",
          "min": '0',
          "max": '15',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Fireplaces",
          "tooltip": "Number of fireplaces"
          },

    "FireplaceQu":
         {"name": "FireplaceQu",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent - Exceptional Masonry Fireplace"},
              {"value": "Gd", "text": "Good - Masonry Fireplace in main level"},
              {"value": "TA",
               "text": "Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement"},
              {"value": "Fa", "text": "Fair - Prefabricated Fireplace in basement"},
              {"value": "Po", "text": "Poor - Ben Franklin Stove"},
              {"value": "NA", "text": "No Fireplace"},
          ],
          "labelText": "Fireplace Quality",
          "tooltip": "Fireplace Quality Rating"},

    "GarageType":
         {"name": "GarageType",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "2Types", "text": "More than one type of garage"},
              {"value": "Attchd", "text": "Attached to home"},
              {"value": "Basment", "text": "Basement Garage"},
              {"value": "BuiltIn", "text": "Built-In (Garage part of house - typically has room above garage)"},
              {"value": "CarPort", "text": "Car Port"},
              {"value": "Detchd", "text": "Detached from home"},
              {"value": "NA", "text": "No Garage"},
          ],
          "labelText": "Garage Type",
          "tooltip": "Garage location"},

    "GarageYrBlt":
         {"name": "GarageYrBlt",
          "type": "slider",
          "min": '1870',
          "max": str(datetime.datetime.today().year),
          "step": '1',
          "value": str(datetime.datetime.today().year),
          "unit": "",
          "labelText": "Garage Year Built",
          "tooltip": "Year garage was built"
          },

    "GarageFinish":
         {"name": "GarageFinish",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Fin", "text": "Finished"},
              {"value": "RFn", "text": "Rough Finished"},
              {"value": "Unf", "text": "Unfinished"},
              {"value": "NA", "text": "No Garage)"},

          ],
          "labelText": "Garage Finish",
          "tooltip": "Interior finish of the garage"},

    "GarageCars":
         {"name": "GarageCars",
          "type": "slider",
          "min": '0',
          "max": '15',
          "step": '1',
          "value": '0',
          "unit": "",
          "labelText": "Garage Capacity",
          "tooltip": "Size of garage in car capacity"
          },

    "GarageArea":
         {"name": "GarageArea",
          "type": "slider",
          "min": '0',
          "max": '3000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Garage Area",
          "tooltip": "Size of garage in square feet"
          },

    "GarageQual":
         {"name": "GarageQual",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Average/Typical"},
              {"value": "Fa", "text": "Fair"},
              {"value": "Po", "text": "Poor"},
              {"value": "NA", "text": "No Garage"},
          ],
          "labelText": "Garage quality",
          "tooltip": ""},

    "GarageCond":
         {"name": "GarageCond",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Ex", "text": "Excellent"},
              {"value": "Gd", "text": "Good"},
              {"value": "TA", "text": "Average/Typical"},
              {"value": "Fa", "text": "Fair"},
              {"value": "Po", "text": "Poor"},
              {"value": "NA", "text": "No Garage"},
          ],
          "labelText": "Garage condition",
          "tooltip": ""},

    "PavedDrive":
         {"name": "PavedDrive",
          "type": "dropdown",
          "fields": [
              {"value": "", "text": ""},
              {"value": "Y", "text": "Paved"},
              {"value": "P", "text": "Partial Pavement"},
              {"value": "N", "text": "Dirt/Gravel"},
          ],
          "labelText": "Driveway Type",
          "tooltip": ""},

    "WoodDeckSF":
         {"name": "WoodDeckSF",
          "type": "slider",
          "min": '0',
          "max": '2000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Wood Deck Area",
          "tooltip": "Wood deck area in square feet"
          },

    "OpenPorchSF":
         {"name": "OpenPorchSF",
          "type": "slider",
          "min": '0',
          "max": '1250',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Open Porch Area",
          "tooltip": "Open porch area in square feet"
          },

    "EnclosedPorch":
         {"name": "EnclosedPorch",
          "type": "slider",
          "min": '0',
          "max": '1000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Enclosed porch area",
          "tooltip": "Enclosed porch area in square feet"
          },

    "3SsnPorch":
         {"name": "3SsnPorch",
          "type": "slider",
          "min": '0',
          "max": '1000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Three season porch area",
          "tooltip": "Three season porch area in square feet"
          },

    "ScreenPorch":
         {"name": "ScreenPorch",
          "type": "slider",
          "min": '0',
          "max": '1000',
          "step": '5',
          "value": '0',
          "unit": "sq. ft",
          "labelText": "Screen porch area",
          "tooltip": "Screen porch area in square feet"
          },

}
