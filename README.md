# Pytorch_Ext
Extension of Pytorch, including new layers, functionals, etc.

----

This is Pytorch_Ext, an extension package to Pytorch.

Recently Facebook has published a new deep learning framework Pytorch. I have to say I love it so much! It finally fills the gap between Torch and the whole Python scientific computing ecology. As an old Theano user, I'm really impressed by the ease of use also the astonishing speed provided by the dynamic computation power of Pytorch.

Though Pytorch is still far from mature, I've already transferred all of my works on it. During this process, I re-organize the corresponding codes and pack some useful stuff into a standalone project, namely the Pytorch_Ext. It's an extension package to Pytorch, and in a very early dev. stage, for now it features:

* **CTC objective in pure pytorch**. CTC (Connectionist Temporal Classification) is a very important objective in speech / handwriting recognition training. This is a direct pytorch translation of my previous Theano implementation and it runs much faster.
* **Center layer for center loss regulation**, refer to [pdf] for details.

Any issue and suggestion are welcome.

Enjoy :)

