LEONHARD = True
# ./test.sh > out 2>&1 &
hiddensizes = [200]
lrs = [0.001]
dropouts = [0.8]
thetas = [1e-4, 2e-4, 4e-4, 1e-3]
gammas = [1e-4, 2e-4, 4e-4, 1e-3]

cnt = 0
for theta in thetas:
	for gamma in gammas:
		for h in hiddensizes:
			for lr in lrs:
				for d in dropouts:
					if cnt == 0:
						print('module load python_gpu/3.6.4 &&')
						print('module load cuda/9.0.176 cudnn/7.3 &&')
					print('bsub -W 24:00 -n 2 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=8192,ngpus_excl_p=1]" python main.py --pre_embedding --hidden_size {}'
					      ' --max_steps 1000 --gen_bidirectional --gen_layers 2 --drop_out {} --batch_size 100 --test_batch_size 100 --learning_rate {} --epochs 200 --theta {} --gamma {} &&'
					      .format(h, d, lr, theta, gamma))
					cnt += 1
					if cnt % 13 == 0:
						print('sleep 7200 &&')
						print('module load python_gpu/3.6.4 &&')
						print('module load cuda/9.0.176 cudnn/7.3 &&')


