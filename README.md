- Using the CHB-MIT datasets found in (https://physionet.org/content/chbmit/1.0.0/#files-panel), added to the TUH Corpus Dataset from Temple University (it requires access), this is the source of it (https://isip.piconepress.com/projects/tuh_eeg).

- The data preprocessing phase depends on the Spatial Artifact Removal Technique (SART) for the artifacts removal applied with some enhancements such as:
  •  Preprocessing phase with filters(DC offset - notch filter - optimized band pass filter).
  •  then enhanced implementation of SART.
  •  using the windowing technique with 4s and 50% overlap.
  • post-processing with a low-pass filter.

- Wavelet transformation using the Mexican Wavelet as the mother wavelet , producing the 3D tensors of the time , frequency, and channel number dimensions stored in HDF5.

- Some of the graphs demonstrating the results you will find in the *snaps* file.


