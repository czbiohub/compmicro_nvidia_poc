# DDL MNIST Example

```bash
- Install PyTorch ([pytorch.org](http://pytorch.org))
    - `conda create -n pytorch_env pytorch
    - `conda activate pytorch_env`
- Copy Pytorch example scripts
    - `pytorch-install-samples ./`
```

Basic mnist test to emulate distributed data parallel using DDL.
```bash
ddlrun -H host1,host2  python mnist.py
```

Example mnist test utilizing Pytorch's DistributedDataParallel Module using DDL
```bash
ddlrun -H host1,host2 python mnist_ddp.py --dist-backend ddl
```
