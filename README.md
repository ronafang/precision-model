## Introduction

Attention (conceptully) lets us know where to focus on.

What if we used this to fetch more information for where we should focus, and remove information where we shouldn't?

I explored this idea with audio data by trying to use attention values to adjust the size of frequency bins to pass in as input in a layer.

I wanted to see what it would take to take a general idea to a custom ML model. There is more to go than what has been done, but it runs!

## Overview

We have data in the form of songs and their instrumental counterpart.

I take 4096 samples at a time, apply FFT, and apply a cumulative sum across the frequency domain with positional encoding. 

In the model, we start out with giving it evenly distributed segments (arbitrary choice, i.e you could make it logarithmic, or more randomized), and use attention values to adjust the frequency bins. For example, we could have 10 bins between 100Hz and 200Hz, or we could have 3. That's what the purpose of the layer is. 

Because of the requirement of a constant number of arguments, I use differentiable indexing by taking the cumulative sum of the attention values and multiplying it by 4095, and pass them to an interpolated function for the range sums. 

The final result is interpreted as a ratio for each sample of the original FFT values, and then the audio is reconstructed. 