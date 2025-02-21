# ssvep

still image on layer 4 without normalization: all filters showed an activation of magnitude 17500 at frequency 0
still image on layer 4 with filter normalization: all filters except filter 9 showed an activation of magnitude 30 at frequency 0. activation of filter 9 was constantly 0 on all frequencies.
whole image flickering with a frequency of 5 on layer 4 with filter normalization: ?

showed output mean of all filters across frames for whole image flickering with a frequency of 5 on layer 4: they show sinusoidal patterns with various amplitudes. they dont have exactly 5 as frequency.

showed output mean of all filters across frames for still image on layer 4: each filter shows constant values across all frames.

changed activation array from rgb to lsh
removed flicker functions conversion to rgb since bgr to hsi is implemented in get activations

BGR= opencv's blue green read image processing thats the defoult for .cv2.imread()