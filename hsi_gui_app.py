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
    finished = Signal(object, object, dict)
    error = Signal(str)

    def __init__(self, predictor, file_path, min_bright, max_bright, conf_thresh):
        super().__init__()
        self.predictor = predictor
        self.file_path = file_path
        self.min_bright = min_bright
        self.max_bright = max_bright
        self.conf_thresh = conf_thresh

    def run(self):
        if not self.file_path or not os.path.exists(self.file_path):
            self.error.emit(f"文件不存在: {self.file_path}")
            return
        try:
            # 传递参数 (现在都是 0.0-1.0 的相对比率)
            _, res_rgb, info = self.predictor.predict_image(
                self.file_path,
                brightness_thresh=self.min_bright,
                high_brightness_thresh=self.max_bright,
                conf_thresh=self.conf_thresh
            )
            self.finished.emit(None, res_rgb, info)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高光谱 AI 分选系统 - 专业演示版")
        self.resize(1300, 850)

        self.predictor = None
        self.current_file_path = None
        self.model_path = None
        self.is_batch_running = False
        self.batch_index = 0

        self.setup_ui()

        # 预填默认路径
        self.edit_white.setText(r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe")
        self.edit_dark.setText(r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe")
        self.edit_input.setText(r"E:\SPEDATA\高谱相机数据集\测试集\PET")
        self.edit_output.setText(r"D:\RESULT\1.22test1.2\testpet-0.01-0.50")

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

        # 1. 模型
        self.btn_select_model = QPushButton("🔍 选择模型文件 (.h5/.onnx)")
        self.btn_select_model.clicked.connect(self.select_model_file)
        control_layout.addWidget(self.btn_select_model)

        # 2. 路径
        path_group = QGroupBox("校准路径")
        path_layout = QFormLayout(path_group)
        self.edit_white = QLineEdit()
        path_layout.addRow("白板:", self.edit_white)
        self.edit_dark = QLineEdit()
        path_layout.addRow("黑板:", self.edit_dark)
        control_layout.addWidget(path_group)

        # 3. 初始化
        self.btn_init_engine = QPushButton("🚀 初始化 AI 引擎")
        self.btn_init_engine.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_init_engine.clicked.connect(self.init_engine)
        self.btn_init_engine.setEnabled(False)
        control_layout.addWidget(self.btn_init_engine)

        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        control_layout.addWidget(line)

        # 4. IO
        io_group = QGroupBox("输入输出")
        io_layout = QVBoxLayout(io_group)

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

        # 5. 批量
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

        # 6. 列表
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_clicked)
        control_layout.addWidget(self.file_list)

        # 7. 参数 (百分比制)
        param_group = QGroupBox("实时参数")
        param_layout = QVBoxLayout(param_group)

        # A. 置信度
        param_layout.addWidget(QLabel("置信度 (Confidence):"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(50)
        self.slider_conf.sliderReleased.connect(self.trigger_update)
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.slider_conf)
        self.lbl_conf_val = QLabel("0.65")
        conf_row.addWidget(self.lbl_conf_val)
        param_layout.addLayout(conf_row)
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v / 100:.2f}"))

        # B. 亮度下限 (Min % of Max)
        param_layout.addWidget(QLabel("亮度下限 (Min - 过滤背景 %):"))
        self.slider_min_bright = QSlider(Qt.Horizontal)
        self.slider_min_bright.setRange(0, 100)
        self.slider_min_bright.setValue(10)  # 默认 15%
        self.slider_min_bright.sliderReleased.connect(self.trigger_update)
        min_bri_row = QHBoxLayout()
        min_bri_row.addWidget(self.slider_min_bright)
        self.lbl_min_bright_val = QLabel("0.15")
        min_bri_row.addWidget(self.lbl_min_bright_val)
        param_layout.addLayout(min_bri_row)
        self.slider_min_bright.valueChanged.connect(lambda v: self.lbl_min_bright_val.setText(f"{v / 100:.2f}"))

        # C. 亮度上限 (Max % of Max)
        param_layout.addWidget(QLabel("亮度上限 (Max - 过滤高光 %):"))
        self.slider_max_bright = QSlider(Qt.Horizontal)
        self.slider_max_bright.setRange(0, 100)
        self.slider_max_bright.setValue(99)  # 默认 95%
        self.slider_max_bright.sliderReleased.connect(self.trigger_update)
        max_bri_row = QHBoxLayout()
        max_bri_row.addWidget(self.slider_max_bright)
        self.lbl_max_bright_val = QLabel("0.95")
        max_bri_row.addWidget(self.lbl_max_bright_val)
        param_layout.addLayout(max_bri_row)
        self.slider_max_bright.valueChanged.connect(lambda v: self.lbl_max_bright_val.setText(f"{v / 100:.2f}"))

        control_layout.addWidget(param_group)

        # === 右侧显示区 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        lbl_title = QLabel("AI 分选结果可视化 (Relative Brightness)")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        right_layout.addWidget(lbl_title)

        self.lbl_res = QLabel("等待指令...")
        self.lbl_res.setAlignment(Qt.AlignCenter)
        self.lbl_res.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc;")
        self.lbl_res.setMinimumSize(800, 400)
        self.lbl_res.setScaledContents(True)
        right_layout.addWidget(self.lbl_res)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)

        self.status_label = QLabel("准备就绪")
        self.statusBar().addWidget(self.status_label)

    # ================= 逻辑 =================

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
        if self.is_batch_running: return

        if not self.predictor or not self.current_file_path: return

        conf = self.slider_conf.value() / 100.0
        # 修改: 映射为 0.0 ~ 1.0 的比率
        min_bright = self.slider_min_bright.value() / 100.0
        max_bright = self.slider_max_bright.value() / 100.0

        self.status_label.setText(f"⏳ 正在推理: {os.path.basename(self.current_file_path)}...")

        self.worker = InferenceWorker(self.predictor, self.current_file_path, min_bright, max_bright, conf)
        self.worker.finished.connect(self.update_display)
        self.worker.error.connect(lambda err: self.status_label.setText(f"❌ {err}"))
        self.worker.start()

    @Slot(object, object, dict)
    def update_display(self, _, res_arr, info):
        res_arr = np.ascontiguousarray(res_arr)

        h, w, ch = res_arr.shape
        bytes_per_line = ch * w
        qt_res = QImage(res_arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_res.setPixmap(QPixmap.fromImage(qt_res))

        self.status_label.setText(f"✅ 完成 | 耗时: {info['total_time']:.3f}s | PET像素: {info['pet_pixels']}")

        if self.chk_auto_save.isChecked():
            out_dir = self.edit_output.text()
            if not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir)
                except:
                    pass

            if os.path.exists(out_dir):
                fname = info['filename'] + ".png"
                save_path = os.path.join(out_dir, fname)
                bgr_img = cv2.cvtColor(res_arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_img)

        if self.is_batch_running:
            self.process_next_batch_image()

    # ================= 批量处理 =================
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
        self.status_label.setText("🛑 已请求停止...")
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
        min_bright = self.slider_min_bright.value() / 100.0
        max_bright = self.slider_max_bright.value() / 100.0

        self.status_label.setText(f"🔄 [批量 {self.batch_index + 1}/{self.file_list.count()}] 处理中...")

        self.worker = InferenceWorker(self.predictor, self.current_file_path, min_bright, max_bright, conf)
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