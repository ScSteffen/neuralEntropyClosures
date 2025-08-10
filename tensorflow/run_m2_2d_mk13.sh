mk=12

for gamma in 3
do

	python callNeuralClosure.py \
		--folder=test$gamma \
		--loadModel=1 \
		--degree=2 \
		--basis=spherical_harmonics \
		--spatialDimension=2 \
		--gammalevel=$gamma \
		--sampling=2 \
		--model=$mk \
		--epoch=2000 \
		--batch=512 \
		--curriculum=1 \
		--normalized=1 \
		--scaledOutput=0 \
		--decorrInput=0 \
		--objective=2 \
		--processingmode=1 \
		--training=1 \
		--verbosity=2 \
		--networkwidth=300 \
		--networkdepth=6 \
		--rotated=0 \
		--max_alpha_norm=10.0
done

  #python callNeuralClosure.py \
	#	--folder=test$gamma \
	#	--loadModel=1 \
	#	--degree=2 \
	#	--basis=spherical_harmonics \
	#	--spatialDimension=2 \
	#	--gammalevel=$gamma \
	#	--sampling=2 \
	#	--model=$mk \
	#	--epoch=100 \
	#	--batch=512 \
	#	--curriculum=1 \
	#	--normalized=1 \
	#	--scaledOutput=0 \
	#	--decorrInput=0 \
	#	--objective=1 \
	#	--processingmode=1 \
	#	--training=1 \
	#	--verbosity=2 \
	#	--networkwidth=300 \
	#	--networkdepth=6 \
	#	--rotated=0 \
	#	--max_alpha_norm=14.0
