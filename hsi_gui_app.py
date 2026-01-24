import sys
import os
import cv2
import numpy as np

# 导入 PySide6 核心组件
from PySide6.QtCore import QThread, Signal, Qt, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QListWidget,
                               QSlider, QCheckBox, QFileDialog, QSplitter,
                               QGroupBox, QMessageBox, QFrame, QLineEdit, QFormLayout)

# 导入后端引擎
from hsi_predictor_core import HSIPredictor


class InferenceWorker(QThread):
    """
    后台推理工人线程
    """
    finished = Signal(object, object, dict)
    error = Signal(str)

    def __init__(self, predictor, file_path, bright_thresh, conf_thresh):
        super().__init__()
        self.predictor = predictor
        self.file_path = file_path
        self.bright_thresh = bright_thresh
        self.conf_thresh = conf_thresh

    def run(self):
        if not self.file_path or not os.path.exists(self.file_path):
            self.error.emit(f"文件不存在: {self.file_path}")
            return

        try:
            raw_rgb, res_rgb, info = self.predictor.predict_image(
                self.file_path,
                brightness_thresh=self.bright_thresh,
                conf_thresh=self.conf_thresh
            )

            if raw_rgb is None:
                self.error.emit(info.get('error', '未知错误'))
            else:
                self.finished.emit(raw_rgb, res_rgb, info)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高光谱 AI 分选系统 - 自动批处理版")
        self.resize(1250, 900)  # 稍微加宽一点以适应图例

        # 1. 初始化变量
        self.predictor = None
        self.current_file_path = None
        self.model_path = None

        # 批量处理相关状态
        self.is_batch_running = False
        self.batch_index = 0

        # 2. 搭建界面布局
        self.setup_ui()

        # 3. 预填默认路径
        self.edit_white.setText(r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe")
        self.edit_dark.setText(r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe")
        self.edit_input.setText(r"E:\SPEDATA\高谱相机数据集\VAL-PET")
        self.edit_output.setText(r"D:\RESULT\1.22test1.2\testpet-0.01-0.50")

        # 自动加载文件列表
        if os.path.exists(self.edit_input.text()):
            self.refresh_file_list(self.edit_input.text())

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === 左侧控制栏 ===
        control_panel = QGroupBox("系统设置与控制")
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(360)

        # --- 1. 模型选择 ---
        self.btn_select_model = QPushButton("🔍 选择模型文件 (.h5/.onnx)")
        self.btn_select_model.clicked.connect(self.select_model_file)
        control_layout.addWidget(self.btn_select_model)

        # --- 2. 路径配置区 ---
        path_group = QGroupBox("校准路径")
        path_layout = QFormLayout(path_group)
        self.edit_white = QLineEdit()
        path_layout.addRow("白板:", self.edit_white)
        self.edit_dark = QLineEdit()
        path_layout.addRow("黑板:", self.edit_dark)
        control_layout.addWidget(path_group)

        # --- 3. 初始化按钮 ---
        self.btn_init_engine = QPushButton("🚀 初始化 AI 引擎")
        self.btn_init_engine.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_init_engine.clicked.connect(self.init_engine)
        self.btn_init_engine.setEnabled(False)
        control_layout.addWidget(self.btn_init_engine)

        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        control_layout.addWidget(line)

        # --- 4. 输入输出设置 ---
        io_group = QGroupBox("输入输出")
        io_layout = QVBoxLayout(io_group)

        # 输入目录
        inp_layout = QHBoxLayout()
        self.edit_input = QLineEdit()
        btn_browse_in = QPushButton("...")
        btn_browse_in.setFixedWidth(30)
        btn_browse_in.clicked.connect(self.browse_input)
        inp_layout.addWidget(self.edit_input)
        inp_layout.addWidget(btn_browse_in)
        io_layout.addWidget(QLabel("输入文件夹 (INPUT):"))
        io_layout.addLayout(inp_layout)

        btn_refresh = QPushButton("🔄 刷新文件列表")
        btn_refresh.clicked.connect(lambda: self.refresh_file_list(self.edit_input.text()))
        io_layout.addWidget(btn_refresh)

        # 输出目录
        out_layout = QHBoxLayout()
        self.edit_output = QLineEdit()
        btn_browse_out = QPushButton("...")
        btn_browse_out.setFixedWidth(30)
        btn_browse_out.clicked.connect(self.browse_output)
        out_layout.addWidget(self.edit_output)
        out_layout.addWidget(btn_browse_out)
        io_layout.addWidget(QLabel("结果保存文件夹 (OUTPUT):"))
        io_layout.addLayout(out_layout)

        self.chk_auto_save = QCheckBox("推理完成后自动保存结果图")
        self.chk_auto_save.setChecked(True)
        io_layout.addWidget(self.chk_auto_save)

        control_layout.addWidget(io_group)

        # --- 5. 批量分类控制区 ---
        batch_group = QGroupBox("自动分选控制")
        batch_layout = QHBoxLayout(batch_group)

        self.btn_start_batch = QPushButton("▶ 开始批量分类")
        self.btn_start_batch.setStyleSheet("font-weight: bold; color: green; font-size: 10pt;")
        self.btn_start_batch.clicked.connect(self.start_batch)

        self.btn_stop_batch = QPushButton("⏹ 停止分类")
        self.btn_stop_batch.setStyleSheet("font-weight: bold; color: red; font-size: 10pt;")
        self.btn_stop_batch.clicked.connect(self.stop_batch)
        self.btn_stop_batch.setEnabled(False)

        batch_layout.addWidget(self.btn_start_batch)
        batch_layout.addWidget(self.btn_stop_batch)
        control_layout.addWidget(batch_group)

        # --- 6. 文件列表 ---
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_clicked)
        control_layout.addWidget(self.file_list)

        # --- 7. 参数调整 ---
        param_group = QGroupBox("实时参数 (Batch时请勿调整)")
        param_layout = QVBoxLayout(param_group)

        # 置信度
        param_layout.addWidget(QLabel("置信度 (Confidence):"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(50)
        self.slider_conf.sliderReleased.connect(self.trigger_update)
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.slider_conf)
        self.lbl_conf_val = QLabel("0.50")
        conf_row.addWidget(self.lbl_conf_val)
        param_layout.addLayout(conf_row)
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v / 100:.2f}"))

        # 亮度
        param_layout.addWidget(QLabel("亮度过滤 (Brightness):"))
        self.slider_bright = QSlider(Qt.Horizontal)
        self.slider_bright.setRange(0, 100)
        self.slider_bright.setValue(10)
        self.slider_bright.sliderReleased.connect(self.trigger_update)
        bri_row = QHBoxLayout()
        bri_row.addWidget(self.slider_bright)
        self.lbl_bright_val = QLabel("0.01")
        bri_row.addWidget(self.lbl_bright_val)
        param_layout.addLayout(bri_row)
        self.slider_bright.valueChanged.connect(lambda v: self.lbl_bright_val.setText(f"{v / 1000:.3f}"))

        control_layout.addWidget(param_group)

        # === 右侧显示区 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        lbl_title = QLabel("AI 分选结果可视化 (实时保存中...)")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        right_layout.addWidget(lbl_title)

        self.lbl_res = QLabel("等待指令...")
        self.lbl_res.setAlignment(Qt.AlignCenter)
        self.lbl_res.setStyleSheet("background-color: #111; border: 2px solid #555;")
        self.lbl_res.setMinimumSize(600, 600)
        self.lbl_res.setScaledContents(False)
        right_layout.addWidget(self.lbl_res)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)

        # 状态栏
        self.status_label = QLabel("准备就绪")
        self.statusBar().addWidget(self.status_label)

    # ================= 基础逻辑 =================

    def select_model_file(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "AI Models (*.h5 *.onnx)")
        if fpath:
            self.model_path = fpath
            self.btn_select_model.setText(f"已选: {os.path.basename(fpath)}")
            self.btn_init_engine.setEnabled(True)

    def browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "选择输入目录", self.edit_input.text())
        if d:
            self.edit_input.setText(d)
            self.refresh_file_list(d)

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.edit_output.text())
        if d: self.edit_output.setText(d)

    def refresh_file_list(self, folder):
        self.file_list.clear()
        if not os.path.exists(folder): return
        import glob
        files = glob.glob(os.path.join(folder, "*.spe"))
        for f in files:
            self.file_list.addItem(f)
        self.status_label.setText(f"已加载 {len(files)} 个文件")

    def init_engine(self):
        if not self.model_path: return
        w_path = self.edit_white.text()
        d_path = self.edit_dark.text()

        if not os.path.exists(w_path) or not os.path.exists(d_path):
            QMessageBox.warning(self, "路径错误", "白板或黑板文件路径不存在！")
            return

        self.status_label.setText("⏳ 正在加载模型...")
        self.btn_init_engine.setEnabled(False)
        QApplication.processEvents()

        try:
            config_path = "best_bands_config.json"
            if not os.path.exists(config_path):
                alt_path = os.path.join(os.path.dirname(self.model_path), "best_bands_config.json")
                if os.path.exists(alt_path): config_path = alt_path

            self.predictor = HSIPredictor(
                model_path=self.model_path,
                config_path=config_path,
                white_ref_path=w_path,
                dark_ref_path=d_path
            )
            self.status_label.setText("✅ 引擎初始化成功")
            self.btn_init_engine.setText("🚀 引擎运行中")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败:\n{str(e)}")
            self.btn_init_engine.setEnabled(True)

    def on_file_clicked(self, item):
        if self.is_batch_running: return
        self.current_file_path = item.text()
        self.trigger_update()

    def trigger_update(self):
        if self.is_batch_running and self.sender() in [self.slider_conf, self.slider_bright]:
            return

        if not self.predictor or not self.current_file_path: return

        conf = self.slider_conf.value() / 100.0
        bright = self.slider_bright.value() / 1000.0

        self.status_label.setText(f"⏳ 正在推理: {os.path.basename(self.current_file_path)}...")

        self.worker = InferenceWorker(self.predictor, self.current_file_path, bright, conf)
        self.worker.finished.connect(self.update_display)
        self.worker.error.connect(lambda err: self.status_label.setText(f"❌ {err}"))
        self.worker.start()

    def add_colorbar(self, img_rgb):
        """
        [新增] 在图像右侧添加 Jet 颜色条 (0.0 Blue -> 1.0 Red)
        """
        h, w, c = img_rgb.shape
        bar_w = 25  # 颜色条宽度
        text_w = 40  # 文字区域宽度
        margin = 5  # 间距

        # 1. 生成梯度条 (从上到下: 255 -> 0)
        # 对应 Jet: 255=Red(1.0), 0=Blue(0.0)
        gradient = np.linspace(255, 0, h).astype(np.uint8)
        # 扩展宽度
        gradient_img = np.tile(gradient[:, np.newaxis], (1, bar_w))

        # 应用伪彩色 (结果是 BGR)
        bar_bgr = cv2.applyColorMap(gradient_img, cv2.COLORMAP_JET)
        # 转回 RGB 以便拼接
        bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)

        # 2. 生成文字背景 (黑色)
        text_bg = np.zeros((h, text_w, 3), dtype=np.uint8)

        # 3. 添加刻度文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        color = (255, 255, 255)  # White
        thick = 1

        # Top (1.0)
        cv2.putText(text_bg, "1.0", (2, 25), font, scale, color, thick)
        # Mid (PET)
        cv2.putText(text_bg, "PET", (2, h // 2), font, scale, color, thick)
        # Bottom (0.0)
        cv2.putText(text_bg, "0.0", (2, h - 10), font, scale, color, thick)

        # 4. 拼接 (Image | Gap | Bar | Text)
        gap = np.zeros((h, margin, 3), dtype=np.uint8)  # 黑色间隔

        combined = np.hstack((img_rgb, gap, bar_rgb, text_bg))
        return combined

    @Slot(object, object, dict)
    def update_display(self, raw_arr, res_arr, info):
        # === [修改] 添加图例处理逻辑 ===

        # 1. 给结果图加个边框 (图例)
        # 注意: res_arr 是 RGB 格式
        final_img = self.add_colorbar(res_arr)

        # 2. 更新 GUI 可视化 (显示带图例的图)
        h, w, ch = final_img.shape
        bytes_per_line = ch * w
        qt_res = QImage(final_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_res.setPixmap(QPixmap.fromImage(qt_res).scaled(
            self.lbl_res.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.status_label.setText(f"✅ 完成 | 耗时: {info['total_time']:.3f}s | PET像素: {info['pet_pixels']}")

        # 3. 保存逻辑 (保存带图例的图)
        if self.chk_auto_save.isChecked():
            out_dir = self.edit_output.text()
            if not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir)
                except:
                    pass

            if os.path.exists(out_dir):
                fname = info['filename'] + "_result.png"
                save_path = os.path.join(out_dir, fname)

                # 注意：OpenCV 保存需要 BGR，而 final_img 是 RGB
                bgr_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_img)

        # 4. 批量循环
        if self.is_batch_running:
            self.process_next_batch_image()

    # ================= 批量处理逻辑 =================

    def start_batch(self):
        if not self.predictor:
            QMessageBox.warning(self, "提示", "请先初始化 AI 引擎！")
            return

        if self.file_list.count() == 0:
            QMessageBox.warning(self, "提示", "文件列表为空！")
            return

        self.is_batch_running = True
        self.batch_index = 0

        self.btn_start_batch.setEnabled(False)
        self.btn_stop_batch.setEnabled(True)
        self.chk_auto_save.setChecked(True)
        self.file_list.setEnabled(False)

        self.process_batch_step()

    def stop_batch(self):
        self.is_batch_running = False
        self.status_label.setText("🛑 已请求停止，将在当前图片处理完后终止...")
        self.btn_stop_batch.setEnabled(False)

    def process_next_batch_image(self):
        if not self.is_batch_running:
            self.finish_batch()
            return

        self.batch_index += 1
        if self.batch_index < self.file_list.count():
            QTimer.singleShot(100, self.process_batch_step)
        else:
            self.finish_batch()

    def process_batch_step(self):
        if not self.is_batch_running: return

        item = self.file_list.item(self.batch_index)
        self.file_list.setCurrentItem(item)
        self.file_list.scrollToItem(item)

        self.current_file_path = item.text()

        conf = self.slider_conf.value() / 100.0
        bright = self.slider_bright.value() / 1000.0

        self.status_label.setText(
            f"🔄 [批量 {self.batch_index + 1}/{self.file_list.count()}] 处理中: {os.path.basename(self.current_file_path)}")

        self.worker = InferenceWorker(self.predictor, self.current_file_path, bright, conf)
        self.worker.finished.connect(self.update_display)
        self.worker.error.connect(lambda err: self.status_label.setText(f"❌ {err}"))
        self.worker.start()

    def finish_batch(self):
        self.is_batch_running = False
        self.btn_start_batch.setEnabled(True)
        self.btn_stop_batch.setEnabled(False)
        self.file_list.setEnabled(True)
        self.status_label.setText("✅ 批量处理任务已结束")
        QMessageBox.information(self, "完成", "批量分选任务已完成！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())