I want to use an unsupervised method to reconstruct the spen image, mainly need to recover a phase map, and then the traditional method can better do the reconstruction.

Refer to `/home/data1/musong/workspace/python/spen_recons/skill/spen_recons_skill.md` for the coding standard and `/home/data1/musong/workspace/python/spenpy/skill.md` for the spen library.

One naive implementation is by my mentor is like this:

hr -> spen+phase problem -> lr

lr -> phase map + inva -> hr

hr -> awhole -> lr

If the two lr differ much, that mean the predicted phase map is bad

Our network is intended to predict the good phase map