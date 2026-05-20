"""
Pygame手写输入界面
提供可视化手写数字输入、实时预览和识别结果展示
"""

import pygame
import numpy as np
from PIL import Image, ImageDraw
import torch
import io
import os


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
        x, y = pos[1] // self.scale, pos[0] // self.scale

        for dx in range(-self.brush_size + 1, self.brush_size):
            for dy in range(-self.brush_size + 1, self.brush_size):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                    dist = dx * dx + dy * dy
                    if dist < self.brush_size * self.brush_size:
                        self.canvas[nx, ny] = 255

    def draw_line(self, pos):
        """从上一个点到当前位置绘制线"""
        if self.last_pos is None:
            self.draw_point(pos)
        else:
            x0, y0 = self.last_pos[1] // self.scale, self.last_pos[0] // self.scale
            x1, y1 = pos[1] // self.scale, pos[0] // self.scale

            points = self._bresenham_line(x0, y0, x1, y1)
            for x, y in points:
                if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                    for dx in range(-self.brush_size + 1, self.brush_size):
                        for dy in range(-self.brush_size + 1, self.brush_size):
                            nx, ny = x + dx, y + dy
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

        self.canvas_width = canvas_size
        self.canvas_height = canvas_size
        self.canvas = HandwritingCanvas(canvas_size, canvas_size)

        self.screen_width = 600
        self.screen_height = 450
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('LeNet-5 手写数字识别')

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.bg_color = (240, 240, 240)
        self.button_color = (70, 130, 180)
        self.button_hover = (100, 160, 210)
        self.text_color = (50, 50, 50)

        self.buttons = {
            'predict': pygame.Rect(320, 40, 120, 45),
            'clear': pygame.Rect(460, 40, 100, 45),
            'save': pygame.Rect(320, 100, 120, 45),
            'quit': pygame.Rect(460, 100, 100, 45),
        }

        self.last_prediction = None
        self.last_confidence = 0.0
        self.inference_time = 0.0

        self.preview_surface = None
        self.update_preview()

    def update_preview(self):
        """更新预览图像"""
        canvas_array = self.canvas.get_canvas()
        preview = np.zeros((self.canvas_width, self.canvas_height), dtype=np.uint8)
        for i in range(self.canvas_width):
            for j in range(self.canvas_height):
                preview[i, j] = canvas_array[i * self.canvas.canvas_size // self.canvas_width,
                                            j * self.canvas.canvas_size // self.canvas_height]

        preview_pil = Image.fromarray(preview)
        preview_rgb = np.array(preview_pil.resize((self.canvas_width, self.canvas_height)))
        preview_rgb = np.transpose(preview_rgb, (1, 0, 2))

        self.preview_surface = pygame.surfarray.make_surface(preview_rgb)

    def preprocess_canvas(self):
        """预处理画布内容用于模型输入"""
        canvas = self.canvas.get_canvas()

        inverted = 255 - canvas

        image = Image.fromarray(inverted)
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

        text_surface = self.small_font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw(self):
        """绘制界面"""
        self.screen.fill(self.bg_color)

        canvas_rect = pygame.Rect(20, 20, self.canvas_width, self.canvas_height)
        pygame.draw.rect(self.screen, (255, 255, 255), canvas_rect)
        if self.preview_surface:
            self.screen.blit(self.preview_surface, (canvas_rect.x, canvas_rect.y))

        pygame.draw.rect(self.screen, (100, 100, 100), canvas_rect, 2)

        mouse_pos = pygame.mouse.get_pos()
        for name, rect in self.buttons.items():
            hover = rect.collidepoint(mouse_pos)
            self.draw_button(self.screen, rect, name.capitalize(), hover)

        info_y = 320
        if self.last_prediction is not None:
            result_text = f'识别结果: {self.last_prediction}'
            result_surface = self.font.render(result_text, True, (0, 100, 0))
            self.screen.blit(result_surface, (20, info_y))

            conf_text = f'置信度: {self.last_confidence:.2f}%'
            conf_surface = self.font.render(conf_text, True, (50, 50, 150))
            self.screen.blit(conf_surface, (20, info_y + 40))

            time_text = f'推理时间: {self.inference_time:.2f}ms'
            time_surface = self.font.render(time_text, True, (100, 100, 100))
            self.screen.blit(time_surface, (20, info_y + 80))

            prob_text = "各数字概率:"
            prob_surface = self.small_font.render(prob_text, True, self.text_color)
            self.screen.blit(prob_surface, (320, info_y))

            image_tensor = self.preprocess_canvas()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            for i, prob in enumerate(probs):
                bar_x = 320
                bar_y = info_y + 25 + i * 20
                bar_width = int(prob * 200)
                pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, 200, 15))
                pygame.draw.rect(self.screen, (70, 130, 180), (bar_x, bar_y, bar_width, 15))
                num_surface = self.small_font.render(f'{i}: {prob*100:.1f}%', True, self.text_color)
                self.screen.blit(num_surface, (530, bar_y))

        instructions = [
            "使用鼠标左键在画布上书写数字",
            "Predict: 识别当前数字",
            "Clear: 清空画布",
            "Save: 保存当前图像"
        ]
        for i, text in enumerate(instructions):
            surface = self.small_font.render(text, True, (120, 120, 120))
            self.screen.blit(surface, (20, self.screen_height - 100 + i * 20))

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
