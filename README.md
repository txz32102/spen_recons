请参考`/home/data1/musong/workspace/python/spen_recons/skill/周报.md`
参考如下python脚本，md文件和log文件
```txt
/home/data1/musong/workspace/python/spen_recons/script/0415_unsupervised_train.md


如下脚本非常难训练
/home/data1/musong/workspace/python/spen_recons/script/0415_unsupervised_train.py
可参考/home/data1/musong/workspace/python/spen_recons/log/04162054_spen_unsupervised_0415/training.log和
/home/data1/musong/workspace/python/spen_recons/log/04162054_spen_unsupervised_0415/images/epoch_093.png

对于下面这两个脚本，虽然是监督的，但你需要参考/home/data1/musong/workspace/python/spen_recons/script/0415_unsupervised_train.md把它说成找到了一个好的初始化，本质上还能算是无监督的，只是一开始给了一点点引导
/home/data1/musong/workspace/python/spen_recons/script/0417_supervised_train_v1.py
/home/data1/musong/workspace/python/spen_recons/script/0417_supervised_train_v2.py


/home/data1/musong/workspace/python/spen_recons/log/04171007_supervised_phase_0417_v1/training.log
/home/data1/musong/workspace/python/spen_recons/log/04171007_supervised_phase_0417_v1/images/epoch_094.png
/home/data1/musong/workspace/python/spen_recons/log/04171007_supervised_phase_0417_v2/training.log
/home/data1/musong/workspace/python/spen_recons/log/04171007_supervised_phase_0417_v2/images/epoch_060.png（这个图片效果已经非常好了，但是phase map看上去还是会有一点可见的artifact，不过也很难变得更好了）
```

请将结果写入/home/data1/musong/workspace/python/spen_recons/work.md