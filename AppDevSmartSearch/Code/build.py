import subprocess
from subprocess import call
call(["python", "config.py"])

import config
GLOBAL_SECTION_UPDATE = config.GLOBAL_SECTION_UPDATE
PREPROCESSING_SECTION_UPDATE = config.PREPROCESSING_SECTION_UPDATE
IMPLEMENTATION_SECTION_UPDATE = config.IMPLEMENTATION_SECTION_UPDATE
INTERFACE_SECTION_UPDATE = config.INTERFACE_SECTION_UPDATE

if(GLOBAL_SECTION_UPDATE == 1):
    call(["python", "Smart_Search_Preprocessing.py"])
    call(["python", "Smart_Search_Implementation.py"])
    call(["python", "Smart_Search_Interface.py"])
    
else:
    if(PREPROCESSING_SECTION_UPDATE == 1):
        call(["python", "Smart_Search_Preprocessing.py"])

    if(IMPLEMENTATION_SECTION_UPDATE == 1):
        call(["python", "Smart_Search_Implementation.py"])

    if(INTERFACE_SECTION_UPDATE == 1):
        call(["python", "Smart_Search_Interface.py"])

