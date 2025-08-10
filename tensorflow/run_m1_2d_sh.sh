for gamma in 0 # 1 2 3
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
		--epoch=5 \
		--batch=64 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=0 \
		--decorrInput=1 \
		--objective=1 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=250 \
		--networkdepth=1
	
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
		--scaledOutput=0 \
		--decorrInput=1 \
		--objective=2 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=250 \
		--networkdepth=1

done


