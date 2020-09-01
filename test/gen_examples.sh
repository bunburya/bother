pipenv run ../bother/bother.py -ot mallorca.tif -b 39.195468 2.272535 39.998336 3.584351 mallorca.png
pipenv run ../bother/bother.py -it mallorca.tif --crop 1574x787 c -si 2048x1024 mallorca_no_processing_resized.png

pipenv run ../bother/bother.py -sd 0.5 -l 100 -rl -b 51.296094 -11.017116 55.596810 -4.403352 ireland_resampled_lakes_low_raised.png
pipenv run ../bother/bother.py -mb 170 -sd 0.5 -l 100 -rl -b 51.296094 -11.017116 55.596810 -4.403352 ireland_resampled_lakes_low_raised_flat.png

pipenv run ../bother/bother.py -l -b -16.970092 -70.246331 -15.032316 -68.334710 titicaca_lakes.png
pipenv run ../bother/bother.py --epsg 3857 -l -b -16.970092 -70.246331 -15.032316 -68.334710 titicaca_lakes_3857.png

pipenv run ../bother/bother.py -b 46.408240 9.555657 46.975534 10.378602 alps.png
