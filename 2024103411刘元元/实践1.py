
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

json_folder = 'C:/Users/Lenovo/Desktop/New folder1'  # 请替换为实际文件夹路径

# 获取文件夹中所有JSON文件的路径
json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder)
                if f.endswith('.json')]

    # 检查是否有JSON文件
if not json_files:
    print(f"错误：在文件夹 '{json_folder}' 中未找到JSON文件！")
    exit(1)

    # 为每个JSON文件创建单独的可视化
for json_file in json_files:
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        # 提取三角面数据
        triangles = [data[key] for key in data]
        triangles = np.array(triangles)

        # 创建 3D 图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制三角面
        poly3d = Poly3DCollection(triangles, alpha=0.5, linewidths=0.5, edgecolors='k')
        ax.add_collection3d(poly3d)

        # 设置坐标轴范围
        x_coords = triangles[:, :, 0].flatten()
        y_coords = triangles[:, :, 1].flatten()
        z_coords = triangles[:, :, 2].flatten()
        ax.set_xlim([np.min(x_coords), np.max(x_coords)])
        ax.set_ylim([np.min(y_coords), np.max(y_coords)])
        ax.set_zlim([np.min(z_coords), np.max(z_coords)])

        # 设置坐标轴标签和标题
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'蛋白质结构: {os.path.basename(json_file)}', fontsize=14)

            # 优化视角
        ax.view_init(elev=30, azim=45)

            # 保存图片
        output_filename = os.path.splitext(os.path.basename(json_file))[0] + '_visualization.png'
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)

            # 关闭图形以释放内存
        plt.close(fig)

        print(f"已生成 {output_filename}")

    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {e}")

print(f"所有 {len(json_files)} 个JSON文件处理完成！")