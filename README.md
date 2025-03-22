# Vakoverschrijdend project
Development of a Denoising and Super-resolution Deep Learning framework for micro-MRI

## Used file structure:
<pre>
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
    |_ ZARR_PREPROCESSED
        |_ ... (same as DICOM)
    |_ BLACKLIST.json
|_ PRE-PROCESSING
    |_ visualizer.py
    |_ Slicemanager.py
    |_ preprocessing.py
</pre>

Zarr array file structure: <br>
Resolution -> location -> mouse -> plane -> index (same as DICOM structure)

**Use Visualizer.py to inspect and remove slices**

## Used slice ID format:
ID: MMLPXXR <br>
-> MM: Mouse ID <br>
-> L: Location     H = HEAD-THORAX; T = THORAX-ABDOMEN <br>
-> P: Plane        C = Coronal; S = Sagittal; T = Transax <br>
-> XX: Slice ID <br>
-> R: Resoulution  H = HIGH RES; L = LOW RES <br>
<br>
Example: 01HC01L = 1st low-resolution slice of coronal Head-Thorax image of mouse 1<br>
<br>
BLACKLIST.json contains IDs in format: MMLPXX (where resolution parameter is dropped (because both get blacklisted))

## Data
**Data/ directory on onedrive**:
https://ugentbe-my.sharepoint.com/:f:/r/personal/florencemarie_muller_ugent_be/Documents/2025_VOP_Project_MicroMRI/Data?csf=1&web=1&e=6nPbnh
