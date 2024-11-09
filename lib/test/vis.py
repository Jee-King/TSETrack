import matplotlib.pyplot as plt
import numpy as np

# def pltshow(pred_map, name )

def pltshow(pred_map, name):
    plt.figure(2)
    pred_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    pred_name = './' + name + '.png'
    # plt.savefig(pred_name)
    plt.savefig(pred_name, bbox_inches = 'tight', pad_inches = 0, dpi = 150)
    plt.close(2)


canvas = np.zeros((160, 160))

# 定义一个中心坐标和半径
center_x, center_y = 80, 80
radius = 40

# 创建一个环绕的图案，中心区域值较高，环绕区域值较低
for i in range(160):
    for j in range(160):
        distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
        if distance < radius:
            # 中心区域
            canvas[i, j] = 1
        elif distance < radius + 10:
            # 环绕区域，可以根据需要调整宽度
            canvas[i, j] = 0.5

pltshow(canvas, 'test')