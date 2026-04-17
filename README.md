I want to rewrite /home/data1/musong/workspace/python/spen_recons/script/0415_unsupervised_train.py, which is very hard to train, i think we may use a supervised train instead to get the good phase map.

You can refer to /home/data1/musong/workspace/python/spen_recons/skill/spen_recons_skill.md

You only need to read /home/data1/musong/workspace/python/spen_recons/script/0415_unsupervised_train.py and the md file

you can use the groud truth phase map, that is what we want to learn, you should note that good_img by the inv a is the ground truth, we only need to correct the phase artifact

write the new script /home/data1/musong/workspace/python/spen_recons/script/0416_supervised_phase_train.py

you should make the network paramater larger
you should first use inv a to correct and then try to predict the phase