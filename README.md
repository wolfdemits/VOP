# Vakoverschrijdend project
Development of a Denoising Deep Learning framework for micro-MRI

## Used file structure:

|_ Data
    |_ DICOM
        |_ HIGH RES
            |_ MOUSE[i]
                |_ HEAD-THORAX
                    |_ Coronal
                    |_ Sagittal
                    |_ Transax
                |_ THORAX-ABDOMEN
                    |_ Coronal
                    |_ Sagittal
                    |_ Transax
        |_ LOW RES
            |_ MOUSE[i]
                |_ HEAD-THORAX
                    |_ Coronal
                    |_ Sagittal
                    |_ Transax
                |_ THORAX-ABDOMEN
                    |_ Coronal
                    |_ Sagittal
                    |_ Transax
    |_ ZARR_NON-PREPROCESSED
    |_ ZARR_PREPROCESSED
    |_ BLACKLIST.json
|_ PRE-PROCESSING
   |_ Visualizer.py

**Use Visualizer.py to inspect and remove slices**

## Used slice ID format:
ID: MMLPXXR
-> MM: Mouse ID
-> L: Location     H = HEAD-THORAX; T = THORAX-ABDOMEN
-> P: Plane        C = Coronal; S = Sagittal; T = Transax
-> XX: Slice ID
-> R: Resoulution  H = HIGH RES; L = LOW RES

Example: 01HC01L = 1st low-resolution slice of coronal Head-Thorax image of mouse 1

BLACKLIST.json contains IDs in format: MMLPXX (where resolution parameter is dropped (because both get blacklisted))
