# python test_al_gl.py --dataset blobs --metric raw --K 8 --config config.yaml --resultsdir results 
# python test_al_gl_voptfull.py --dataset blobs --metric raw --K 8 --config config_full.yaml --resultsdir results
# python accuracy_al_gl.py --dataset blobs --metric raw --config config.yaml --resultsdir results
# python compile_summary.py --dataset blobs --resultsdir results

python test_al_gl.py --dataset paviasub --metric hsi --K 15 --config config.yaml --resultsdir results 
python accuracy_al_gl.py --dataset paviasub --metric hsi --config config.yaml --resultsdir results
python compile_summary.py --dataset paviasub --resultsdir results

python test_al_gl.py --dataset salinassub --metric hsi --K 15 --config config.yaml --resultsdir results 
python accuracy_al_gl.py --dataset salinassub --metric hsi --config config.yaml --resultsdir results
python compile_summary.py --dataset salinassub --resultsdir results


# python test_al_gl.py --dataset mnist-mod3 --metric vae --K 20 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset mnist-mod3 --metric vae --config config.yaml --resultsdir results
# python compile_summary.py --dataset mnist-mod3 --resultsdir results

# python test_al_gl.py --dataset fashionmnist-mod3 --metric vae --K 20 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset fashionmnist-mod3 --metric vae --config config.yaml --resultsdir results
# python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results


# python test_al_gl.py --dataset mnistsmall-mod3 --metric vae --K 20 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset mnistsmall-mod3 --metric vae --config config.yaml --resultsdir results
# python compile_summary.py --dataset mnistsmall-mod3 --resultsdir results

# python test_al_gl.py --dataset fashionmnistsmall-mod3 --metric vae --K 20 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset fashionmnistsmall-mod3 --metric vae --config config.yaml --resultsdir results
# python compile_summary.py --dataset fashionmnistsmall-mod3 --resultsdir results



