for gamma in 0 1 2 3
do
    python callNeuralClosure.py \
		--folder=sh_mk11_m1_2d_g$gamma \
		--loadModel=0 \
		--degree=1 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=11 \
		--epoch=20 \
		--batch=32 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=1 \
		--decorrInput=0 \
		--objective=0 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=250 \
		--networkdepth=2 \
		--rotated=1

	python callNeuralClosure.py \
		--folder=sh_mk11_m1_2d_g$gamma \
		--loadModel=1 \
		--degree=1 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=11 \
		--epoch=30 \
		--batch=64 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=1 \
		--decorrInput=0 \
		--objective=2 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=250 \
		--networkdepth=2 \
		--rotated=1

	python callNeuralClosure.py \
		--folder=sh_mk11_m1_2d_g$gamma \
		--loadModel=1 \
		--degree=1 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=11 \
		--epoch=1000 \
		--batch=256 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=1 \
		--decorrInput=0 \
		--objective=2 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=250 \
		--networkdepth=2 \
		--rotated=1

done

