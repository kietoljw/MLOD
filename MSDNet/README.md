```python
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi BH
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi BH
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi BH
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi BH

python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi adaBH
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi adaBH
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi adaBH
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi adaBH

python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi BY
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi BY
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi BY
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi BY

python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi Fisher
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi Fisher
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi Fisher
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi Fisher

python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi Cauchy
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi Cauchy
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi Cauchy
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi Cauchy

python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi all -temp 10
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi all -temp 10
python main1.py -ms energy -ml 5 -ma 1 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi all -temp 10
python main1.py -ms energy -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi all -temp 10

python main1.py -ms mahalanobis -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi all
python main1.py -ms odin -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi all
python main1.py -ms msp -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar10.pth.tar -mi cifar10 -multi all
python main1.py -ms mahalanobis -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi all
python main1.py -ms msp -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi all
python main1.py -ms odin -ml 5 -ma 0 -mc png -mf trained_model/msdnet_cifar100.pth.tar -mi cifar100 -multi all



python main1.py -ms mahalanobis -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar10.pth.tar -mi cifar10 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
python main1.py -ms energy -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar10.pth.tar -mi cifar10 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
python main1.py -ms msp -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar10.pth.tar -mi cifar10 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
python main1.py -ms odin -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar10.pth.tar -mi cifar10 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'



python main1.py -ms mahalanobis -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar100.pth.tar -mi cifar100 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
python main1.py -ms energy -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar100.pth.tar -mi cifar100 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
python main1.py -ms msp -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar100.pth.tar -mi cifar100 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
python main1.py -ms odin -ml 6 -ma 0 -mc png -mf trained_model/ranet_cifar100.pth.tar -mi cifar100 -multi all -a ranet --nChannels 16 --nBlocks 2 --stepmode 'lg' --step 2 --grFactor '4-2-1' --bnFactor '4-2-1'
```





