from PIL import Image
import numpy as np


def create_mosaic(image_path, output_path, block_size=16):
    """
    给一张图片添加马赛克效果并保存
    block_size (int): 马赛克块的大小（像素）。值越大，马赛克越严重。
    """
    try:
        # 1. 加载图片并转换为numpy数组
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        height, width, channels = img_array.shape

        # 2. 创建一个空的数组来存放结果
        mosaic_array = np.zeros_like(img_array)

        # 3. 遍历每个马赛克块
        for r in range(0, height, block_size):
            for c in range(0, width, block_size):
                # 确定当前块的边界
                r_end = min(r + block_size, height)
                c_end = min(c + block_size, width)

                # 提取当前块
                block = img_array[r:r_end, c:c_end]

                # 计算块的平均颜色（R, G, B三个通道分别计算平均值）
                if block.size > 0:
                    average_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                else:
                    continue

                # 用平均颜色填充结果数组中的对应块
                mosaic_array[r:r_end, c:c_end] = average_color

        # 4. 从数组创建图片对象并保存
        mosaic_img = Image.fromarray(mosaic_array)
        mosaic_img.save(output_path)
        print(f"成功生成马赛克图片，并保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {image_path}")
    except Exception as e:
        print(f"发生错误: {e}")


# --- 使用示例 ---
# 假设你有一张名为 'original.jpg' 的图片
# 我们将马赛克块大小设置为16x16像素
create_mosaic('D:\work2-diffusion\drawio\img\sar_input.jpg', 'D:\work2-diffusion\drawio\img\mosaic_image.jpg',
              block_size=12)
