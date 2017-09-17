import torchvision.transforms as transforms


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

DEFAULT_TRANSFORMS = (
    transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])
)

TEST_TRANSFORMS = (
    transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        NORMALIZE
    ])
)
