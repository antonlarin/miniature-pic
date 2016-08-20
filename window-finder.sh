#!/usr/bin/env bash

mv defs.py defs.py.bak
sed "s/^ITERATIONS = .*/ITERATIONS = 973/" < defs.py.bak |
    sed "s/^OUTPUT_PERIOD = .*/OUTPUT_PERIOD = 972/" > defs.py.bak2

for fft_window_size in 3 ; do
    sed "s/FFT_WINDOW_SIZE = [0-9]*/FFT_WINDOW_SIZE = $fft_window_size/" < \
        defs.py.bak2 > defs.py
    echo "Testing window size $fft_window_size"
    ./mpic.py 1 > /dev/null
    mv 000001.png window-$fft_window_size.png
done

rm defs.py.bak2
mv defs.py.bak defs.py

