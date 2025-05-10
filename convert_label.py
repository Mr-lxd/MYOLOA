import os
from pathlib import Path


def invert_labels(input_dir, output_dir):
    """
    遍历目录中的所有txt文件，反转标注中的类别0和1

    Args:
        input_dir: 输入文件目录
        output_dir: 输出文件目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 遍历所有txt文件
    for txt_file in Path(input_dir).glob('*.txt'):
        # 读取文件内容
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        # 处理每一行
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # 确保行格式正确
                # 反转类别（0变1，1变0）
                class_id = int(parts[0])
                if class_id == 0:
                    parts[0] = '1'
                elif class_id == 1:
                    parts[0] = '0'
                # 重新组合行
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)
            else:
                # 如果行格式不正确，保持原样
                new_lines.append(line)

        # 写入新文件
        output_file = Path(output_dir) / txt_file.name
        with open(output_file, 'w') as f:
            f.writelines(new_lines)

        print(f"Processed: {txt_file.name}")


# 使用示例
if __name__ == "__main__":
    input_directory = r"D:\navigationData\annotations_COCO\testing\labels"  # 替换为你的输入目录
    output_directory = r"D:\navigationData\annotations_COCO\testing\labels"  # 替换为你的输出目录

    invert_labels(input_directory, output_directory)