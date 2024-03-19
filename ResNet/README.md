# CIFAR
```python
python cifar_KnnMulti.py --in-dataset CIFAR-10  --out-datasets mnist kmnist fasionmnist lsun svhn dtd isun lsunR --name resnet18_cifar  --model-arch resnet18_cifar --multi-method all
python cifar_KnnMulti.py --in-dataset CIFAR-10  --out-datasets mnist kmnist fasionmnist lsun svhn dtd isun lsunR --name resnet34_cifar  --model-arch resnet34_cifar --multi-method all --gpu 1
python cifar_KnnMulti.py --in-dataset CIFAR-10  --out-datasets mnist kmnist fasionmnist lsun svhn dtd isun lsunR --name resnet50_cifar  --model-arch resnet50_cifar --multi-method all


python cifar_KnnMulti.py --in-dataset CIFAR-100  --out-datasets mnist kmnist fasionmnist lsun svhn dtd isun lsunR --name resnet18_cifar  --model-arch resnet18_cifar --multi-method all
python cifar_KnnMulti.py --in-dataset CIFAR-100  --out-datasets mnist kmnist fasionmnist lsun svhn dtd isun lsunR --name resnet34_cifar  --model-arch resnet34_cifar --multi-method all
```




# ImageNet

```python
python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets mnist kmnist fasionmnist lsun svhn dtd isun lsunR  --name resnet50_imagenet  --model-arch resnet50_imagenet --K 100 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets fasionmnist lsun svhn dtd --name resnet50_imagenet  --model-arch resnet50_imagenet --K 100 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets fasionmnist lsun svhn dtd --name resnet50_imagenet  --model-arch resnet50_imagenet --K 100 --multi-method all



python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets fasionmnist lsun svhn dtd --name resnet50_imagenet  --model-arch resnet50_imagenet --K 50 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets sun50 inat places50 dtd --name resnet50_imagenet  --model-arch resnet50_imagenet --K 100 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets dtd sun50 inat places50 --name resnet18_imagenet  --model-arch resnet18_imagenet --K 50 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets sun50 inat places50 dtd --name resnet50_imagenet  --model-arch resnet50_imagenet --K 1000 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets sun50 inat places50 dtd --name resnet50-supcon  --model-arch resnet50-supcon --K 1000 --multi-method all

python ImageNet_Knn_mult.py --in-dataset imagenet  --out-datasets sun50 inat places50 dtd --name resnet101_imagenet  --model-arch resnet101_imagenet --K 1000 --multi-method all

```