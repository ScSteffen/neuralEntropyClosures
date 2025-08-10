for gamma in   0 1 2 3
do
    python callNeuralClosure.py \
		--folder=rot_Harmonic_Mk11_M2_2D_gamma$gamma \
		--loadModel=0 \
		--degree=2 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=11 \
		--epoch=20 \
		--batch=32 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=0 \
		--decorrInput=1 \
		--objective=0 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=350 \
		--networkdepth=3 \
		--rotated=0 \
		--max_alpha_norm=8.0

	python callNeuralClosure.py \
		--folder=rot_Harmonic_Mk11_M2_2D_gamma$gamma \
		--loadModel=1 \
		--degree=2 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=11 \
		--epoch=30 \
		--batch=64 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=0 \
		--decorrInput=1 \
		--objective=2 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=350 \
		--networkdepth=3 \
		--rotated=0 \
		--max_alpha_norm=8.0

	python callNeuralClosure.py \
		--folder=rot_Harmonic_Mk11_M2_2D_gamma$gamma \
		--loadModel=1 \
		--degree=2 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=11 \
		--epoch=1000 \
		--batch=256 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=0 \
		--decorrInput=1 \
		--objective=2 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=350 \
		--networkdepth=3 \
		--rotated=0 \
		--max_alpha_norm=8.0

done

