"""
Pygame手写输入界面
提供可视化手写数字输入、实时预览和识别结果展示
"""

import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import os
import platform

# 全局字体缓存
_cn_font = None

def get_chinese_font(size=36):
    """获取中文字体"""
    global _cn_font
    if _cn_font is not None:
        return _cn_font
    
    # 尝试多个中文字体
    font_paths = []
    system = platform.system()
    
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',      # 黑体
            'C:/Windows/Fonts/msyh.ttc',        # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',      # 宋体
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        ]
    
    # 尝试加载字体
    for path in font_paths:
        if os.path.exists(path):
            try:
                _cn_font = ImageFont.truetype(path, size)
                return _cn_font
            except:
                continue
    
    # 如果都找不到，使用默认字体（可能不支持中文）
    _cn_font = ImageFont.load_default()
    return _cn_font

def draw_text_surface(text, size=36, color=(50, 50, 50)):
    """使用PIL渲染文本，返回pygame Surface"""
    font = get_chinese_font(size)
    # 获取文本尺寸
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # 创建PIL图像
    img = Image.new('RGBA', (width + 4, height + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, font=font, fill=(*color, 255))
    
    # 转换为pygame Surface
    str_buf = img.tobytes('raw', 'RGBA')
    surface = pygame.image.fromstring(str_buf, img.size, 'RGBA')
    return surface


class HandwritingCanvas:
    """手写画布组件"""

    def __init__(self, width=280, height=280, canvas_size=28):
        self.width = width
        self.height = height
        self.canvas_size = canvas_size
        self.scale = width // canvas_size

        self.canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        self.drawing = False
        self.last_pos = None

        self.brush_size = 2

    def reset(self):
        """清空画布"""
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

    def get_canvas(self):
        """获取当前画布内容"""
        return self.canvas.copy()

    def draw_point(self, pos):
        """在指定位置绘制点"""
        # pygame: pos = (x_horizontal, y_vertical)
        # canvas[row, col] = canvas[y_vertical, x_horizontal]
        x = pos[0] // self.scale   # 水平方向 -> 列
        y = pos[1] // self.scale   # 垂直方向 -> 行

        for dx in range(-self.brush_size + 1, self.brush_size):
            for dy in range(-self.brush_size + 1, self.brush_size):
                nx, ny = y + dy, x + dx
                if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                    dist = dx * dx + dy * dy
                    if dist < self.brush_size * self.brush_size:
                        self.canvas[nx, ny] = 255

    def draw_line(self, pos):
        """从上一个点到当前位置绘制线"""
        if self.last_pos is None:
            self.draw_point(pos)
        else:
            # pygame: pos = (x_horizontal, y_vertical)
            # canvas[row, col] = canvas[y_vertical, x_horizontal]
            x0, y0 = self.last_pos[0] // self.scale, self.last_pos[1] // self.scale
            x1, y1 = pos[0] // self.scale, pos[1] // self.scale

            points = self._bresenham_line(x0, y0, x1, y1)
            for x, y in points:
                if 0 <= y < self.canvas_size and 0 <= x < self.canvas_size:
                    for dx in range(-self.brush_size + 1, self.brush_size):
                        for dy in range(-self.brush_size + 1, self.brush_size):
                            nx, ny = y + dy, x + dx
                            if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                                dist = dx * dx + dy * dy
                                if dist < self.brush_size * self.brush_size:
                                    self.canvas[nx, ny] = 255

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham算法绘制直线"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def handle_event(self, event):
        """处理 pygame 事件"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.drawing = True
                self.last_pos = event.pos
                self.draw_point(event.pos)

        elif event.type == pygame.MOUSEMOTION:
            if self.drawing:
                self.draw_line(event.pos)
                self.last_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.drawing = False
                self.last_pos = None

    def get_pil_image(self):
        """获取 PIL Image 对象"""
        return Image.fromarray(self.canvas)


class HandwritingUI:
    """手写数字识别用户界面"""

    def __init__(self, model, device='cpu', canvas_size=280):
        pygame.init()

        self.model = model
        self.model.eval()
        self.device = device

        # 画布尺寸
        self.canvas_size = canvas_size
        self.canvas = HandwritingCanvas(canvas_size, canvas_size, canvas_size // 10)

        # 界面尺寸
        self.screen_width = 420
        self.screen_height = 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('LeNet-5 Handwritten Digit Recognition')

        # 按钮
        self.bg_color = (45, 45, 55)
        self.canvas_bg = (20, 20, 25)
        self.button_color = (70, 130, 180)
        self.button_hover = (90, 150, 200)
        self.text_color = (220, 220, 220)
        self.result_color = (100, 220, 100)

        self.buttons = {
            'predict': pygame.Rect(30, 320, 110, 45),
            'clear': pygame.Rect(155, 320, 110, 45),
            'quit': pygame.Rect(280, 320, 110, 45),
        }

        self.last_prediction = None
        self.last_confidence = 0.0
        self.inference_time = 0.0

        self.update_preview()

    def update_preview(self):
        """更新预览图像"""
        canvas_array = self.canvas.get_canvas()
        # 转置: (height, width) -> (width, height) for pygame
        preview = canvas_array.T
        preview_rgb = np.stack([preview] * 3, axis=-1)  # (width, height, 3)
        self.preview_surface = pygame.surfarray.make_surface(preview_rgb)
        self.preview_surface = pygame.transform.scale(
            self.preview_surface, (self.canvas_size, self.canvas_size)
        )

    def preprocess_canvas(self):
        """预处理画布内容用于模型输入"""
        canvas = self.canvas.get_canvas()

        image = Image.fromarray(canvas)
        image = image.resize((28, 28))

        image_array = np.array(image, dtype=np.float32) / 255.0

        mean = 0.1307
        std = 0.3081
        image_array = (image_array - mean) / std

        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self):
        """执行预测"""
        import time

        image_tensor = self.preprocess_canvas()

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        self.inference_time = (time.time() - start_time) * 1000

        self.last_prediction = predicted.item()
        self.last_confidence = confidence.item() * 100

        self.update_preview()

    def save_canvas(self, directory='./data/user_digits'):
        """保存画布内容"""
        os.makedirs(directory, exist_ok=True)

        canvas = self.canvas.get_canvas()
        image = Image.fromarray(255 - canvas)

        timestamp = int(np.random.randint(10000, 99999))
        filename = os.path.join(directory, f'digit_{timestamp}.png')
        image.save(filename)

        return filename

    def draw_button(self, surface, rect, text, hover=False):
        """绘制按钮"""
        color = self.button_hover if hover else self.button_color
        pygame.draw.rect(surface, color, rect, border_radius=8)
        
        text_surface = draw_text_surface(text, 24, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw(self):
        """绘制界面"""
        self.screen.fill(self.bg_color)

        # 画布
        canvas_rect = pygame.Rect(20, 20, self.canvas_size, self.canvas_size)
        pygame.draw.rect(self.screen, self.canvas_bg, canvas_rect, border_radius=4)
        if self.preview_surface:
            self.screen.blit(self.preview_surface, (canvas_rect.x, canvas_rect.y))
        pygame.draw.rect(self.screen, (100, 100, 100), canvas_rect, 2)

        # 按钮
        mouse_pos = pygame.mouse.get_pos()
        for name, rect in self.buttons.items():
            hover = rect.collidepoint(mouse_pos)
            self.draw_button(self.screen, rect, name.upper(), hover)

        # 结果展示
        if self.last_prediction is not None:
            result_text = f'Result: {self.last_prediction}'
            result_surface = draw_text_surface(result_text, 48, self.result_color)
            self.screen.blit(result_surface, (20, 390))

            conf_text = f'Confidence: {self.last_confidence:.1f}%'
            conf_surface = draw_text_surface(conf_text, 28, self.text_color)
            self.screen.blit(conf_surface, (20, 440))

        pygame.display.flip()

    def run(self):
        """运行主循环"""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for name, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            if name == 'predict':
                                self.predict()
                            elif name == 'clear':
                                self.canvas.reset()
                                self.last_prediction = None
                                self.last_confidence = 0.0
                                self.update_preview()
                            elif name == 'save':
                                filename = self.save_canvas()
                                print(f"图像已保存: {filename}")
                            elif name == 'quit':
                                running = False

                self.canvas.handle_event(event)
                self.update_preview()

            self.draw()
            clock.tick(60)

        pygame.quit()


def create_demo_ui(model, device='cpu'):
    """创建演示UI的工厂函数"""
    return HandwritingUI(model, device)


if __name__ == "__main__":
    print("请通过 main.py 运行主程序来启动手写识别界面")
