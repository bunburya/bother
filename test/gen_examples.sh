pipenv run ../bother -ot examples/mallorca.tif -b 39.195468 2.272535 39.998336 3.584351 examples/mallorca.png
pipenv run ../bother -ot examples/ireland.tif -b 51.296094 -11.017116 55.596810 -4.403352 examples/ireland.png
pipenv run ../bother -ot examples/titicaca.tif -b -16.970092 -70.246331 -15.032316 -68.334710 examples/titicaca.png
pipenv run ../bother -ot examples/constance.tif -b 47.321556 8.830936 47.980224 9.860904 examples/constance.png
pipenv run ../bother -ot examples/alps.tif -b 46.408240 9.555657 46.975534 10.378602 examples/alps.png

pipenv run ../bother -it examples/mallorca.tif --crop 1574x787 c -si 2048x1024 examples/mallorca_no_processing_resized.png

pipenv run ../bother -sd 0.5 -l 100 -rl -b 51.296094 -11.017116 55.596810 -4.403352 examples/ireland_resampled_lakes_low_raised.png
pipenv run ../bother -mb 170 -sd 0.5 -l 100 -rl -b 51.296094 -11.017116 55.596810 -4.403352 examples/ireland_resampled_lakes_low_raised_flat.png

pipenv run ../bother -l -b -16.970092 -70.246331 -15.032316 -68.334710 examples/titicaca_lakes.png
pipenv run ../bother --epsg 3857 -l -b -16.970092 -70.246331 -15.032316 -68.334710 examples/titicaca_lakes_3857.png
