from FoMo.model_zoo import multimodal_mae
import torch
import torch.nn as nn
import pyjson5 as json
import argparse
from PIL import Image
from torchvision import transforms


def construct_fomo_configs(args):
    '''
        Construct configurations for FoMo_1 model
    '''

    configs = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.mlp_dim,
        "num_classes": args.num_classes,
        "single_embedding_layer": True,
    }

    # Update configs with modality specific configurations as defined during pretraining

    modality_configs = json.load(open(args.modality_configs, 'r'))
    configs.update(modality_configs)

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint_path",
                        default="/home/wushixuan/yujun/data/weights/fomo_single_embedding_layer_weights.pt")
    parser.add_argument("--modality_configs",
                        default="/home/wushixuan/桌面/07/work2-exper/FoMo/configs/datasets/fomo_pretraining_datasets.json")
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--heads", default=12, type=int)
    parser.add_argument("--mlp_dim", default=2048, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--single_embedding_layer", default=True, type=bool)
    parser.add_argument("--image_path", default='/home/wushixuan/桌面/07/work2-exper/drawio/img/sar_input.jpg',
                        help="Path to a real image for testing")

    args = parser.parse_args()

    configs = construct_fomo_configs(args)

    # Initialize FoMo model
    v = multimodal_mae.MultiSpectralViT(
        image_size=configs["image_size"],
        patch_size=configs["patch_size"],
        channels=1,
        num_classes=configs["num_classes"],
        dim=configs["dim"],
        depth=configs["depth"],
        heads=configs["heads"],
        mlp_dim=configs["mlp_dim"],
        configs=configs,
    )
    try:
        v.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
        print("Model checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        exit()

    print("Keys after loading checkpoint:", v.state_dict().keys())

    # --- 使用一张真实图片进行测试 ---
    if args.image_path:
        print(f"Testing the loaded model with image: {args.image_path}")

        # 1. 定义预处理转换
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),  # 调整到模型期望的大小
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图，因为 channels=1
            transforms.ToTensor(),  # 转换为 Tensor
            # 如果需要，可以添加归一化
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # --- 关键修改点 ---
        # 根据 forward 函数的签名，需要一个 (img, keys) 的元组
        # 我们的模型是单通道的，所以 keys 只需要一个元素
        # 这个 key 的值需要与模型中预期的 key 匹配
        # 假设一个合理的 key 是 '0'，因为它在 modality_channels 中存在
        # 你可能需要根据实际情况调整这个 key
        # 例如，如果你想使用'sentinel-1-vv'作为通道，那么key就应该是'sentinel-1-vv'
        # 这里的'0'只是一个合理的猜测，你需要根据你的数据集和预训练设置来确定
        # 更好的方法是从 configs 字典中动态查找索引
        modality_to_index = configs["dataset_modality_index"]["tallos"]  # 假设是tallos数据集
        channel_name = 'sentinel-1-vv'  # 假设我们的图片代表这个通道
        channel_index = modality_to_index[channel_name]

        keys_list = [channel_index]  # 现在 keys_list 包含一个整数

        # 打印验证
        print(f"Using keys_list for inference: {keys_list}")

        # 2. 加载图片
        img = Image.open(args.image_path)

        # 3. 预处理图片
        img_tensor = transform(img)

        # 4. 添加批次维度
        # 模型通常期望 (batch_size, channels, height, width) 的输入
        img_tensor = img_tensor.unsqueeze(0)

        # 5. 将模型设置为评估模式
        v.eval()

        # 6. 进行推理
        with torch.no_grad():
            output = v((img_tensor, keys_list))
            encoded_features = v((img_tensor, keys_list), pool=False)

        print("Inference successful!")
        print("Output shape:", output.shape)

        # 打印编码特征信息
        print("\nEncoded Features Shape:", encoded_features.shape)
        print("Encoded Features (first patch, first 5 values):", encoded_features[0, 0, :5])

        # try:
        #
            # 打印预测的类别（如果模型是用于分类的）
        #     if output.dim() == 2 and output.shape[1] == args.num_classes:
        #         predicted_class = torch.argmax(output, dim=1).item()
        #         print("Predicted class index:", predicted_class)
        #
        # except FileNotFoundError:
        #     print(f"Error: The image file at {args.image_path} was not found.")
        # except Exception as e:
        #     print(f"An error occurred during image processing or inference: {e}")
    # else:
    #     print("No image path provided. Model loading verification successful, but no inference test was performed.")
