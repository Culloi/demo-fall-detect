#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI 层: PySide6 (首选) / Tkinter / OpenCV 窗口 / 无界面 四级回退.

UI 与推理线程通过一个共享的 State 字典交互 (UI 线程只读, 推理线程只写).
"""
import threading

import cv2

from pose_detector import draw_pose


# ------------------------------------------------------------------
# 中英文切换 (i18n)
# ------------------------------------------------------------------
_STRINGS = {
    "zh": {
        "win.title": "跌倒检测 + 异常声音告警 Demo (RK3576)",
        "header.title": "跌倒检测监控系统",
        "header.subtitle": "Fall Detection · RK3576 NPU",
        "lang.btn": "EN",
        "badge.normal": "● 正常监控",
        "badge.normal_muted": "● 正常监控 (静音)",
        "badge.fall": "⚠ 检测到跌倒!",
        "badge.fall_alarm": "跌倒(告警中)",
        "card.fps": "帧率",
        "card.infer": "推理耗时",
        "card.person": "画面人数",
        "card.noise": "环境底噪",
        "card.sound": "声音强度",
        "btn.test_alarm": "⚠ 测试告警",
        "btn.mute": "静音 / 取消",
        "btn.manual": "手动存录像",
        "btn.quit": "退出",
        "log.caption": "事件日志",
        "event.sound": "异常声音 {db}dB",
        "event.fall": "检测到人体跌倒!",
        "event.manual": "手动触发录像",
        "status.init": "状态: 初始化",
        "status.normal": "状态: 正常监控",
        "status.down": "状态: ⚠ 检测到跌倒!",
        "status.down_alarm": "状态: 跌倒(告警中)",
        "status.metrics": "FPS: {fps:.1f}  推理: {ms:.0f}ms  人数: {persons}",
        "status.metrics_none": "FPS: -  推理: -ms  人数: 0",
        "status.sound": "声音: {db:.0f}dB  底噪: {nf:.0f}dB",
        "status.sound_none": "声音: --dB  底噪: --dB",
        "cv.hint": "demo_fall_detect (按 q 退出, a=测试告警, m=静音, "
                   "r=手动录像, l=中英文)",
    },
    "en": {
        "win.title": "Fall Detection + Abnormal Sound Alarm Demo (RK3576)",
        "header.title": "Fall Detection Monitor",
        "header.subtitle": "Fall Detection · RK3576 NPU",
        "lang.btn": "中文",
        "badge.normal": "● Monitoring",
        "badge.normal_muted": "● Monitoring (Muted)",
        "badge.fall": "⚠ Fall Detected!",
        "badge.fall_alarm": "Fall (Alarm Active)",
        "card.fps": "FPS",
        "card.infer": "Inference",
        "card.person": "Persons",
        "card.noise": "Noise Floor",
        "card.sound": "Sound Level",
        "btn.test_alarm": "⚠ Test Alarm",
        "btn.mute": "Mute / Unmute",
        "btn.manual": "Manual Record",
        "btn.quit": "Quit",
        "log.caption": "Event Log",
        "event.sound": "Unusual sound {db}dB",
        "event.fall": "Human fall detected!",
        "event.manual": "Manual recording triggered",
        "status.init": "Status: Initializing",
        "status.normal": "Status: Monitoring",
        "status.down": "Status: ⚠ Fall Detected!",
        "status.down_alarm": "Status: Fall (Alarm Active)",
        "status.metrics": "FPS: {fps:.1f}  Inference: {ms:.0f}ms  "
                          "Persons: {persons}",
        "status.metrics_none": "FPS: -  Inference: -ms  Persons: 0",
        "status.sound": "Sound: {db:.0f}dB  Noise: {nf:.0f}dB",
        "status.sound_none": "Sound: --dB  Noise: --dB",
        "cv.hint": "demo_fall_detect (q=quit, a=test alarm, m=mute, "
                   "r=manual record, l=language)",
    },
}
_lang = {"code": "zh"}


def get_lang():
    return _lang["code"]


def set_lang(code):
    if code in _STRINGS:
        _lang["code"] = code


def tr(key, **kwargs):
    table = _STRINGS.get(_lang["code"], _STRINGS["zh"])
    text = table.get(key) or _STRINGS["zh"].get(key, key)
    return text.format(**kwargs) if kwargs else text


def format_event(ev):
    """事件 (时间, 文案 key, 参数) -> 当前语言的可显示文本."""
    ts, key, params = ev
    return f"[{ts}] {tr(key, **(params or {}))}"


class AppState:
    """线程间共享状态."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None          # 标注后的显示帧 (BGR)
        self.fps = 0.0
        self.infer_ms = 0.0
        self.person_count = 0
        self.fall_state = "NORMAL"   # NORMAL / DOWN
        self.sound_db = -90.0
        self.noise_db = -50.0
        self.alarm_active = False
        self.muted = False
        self.events = []           # [(time_str, 文案key, 参数)]
        self.quit = False


class BaseUI:
    def __init__(self, state, app):
        self.state = state
        self.app = app             # 主应用句柄 (提供 trigger/stop/mute 等)

    def run(self):
        pass

    def close(self):
        pass


# ------------------------------------------------------------------
# PySide6 UI
# ------------------------------------------------------------------
QSS = """
QWidget { background: #16181d; color: #e8eaed; font-size: 14px; }
#header { background: #1e2128; border-bottom: 1px solid #2a2e37; }
#title { font-size: 18px; font-weight: 700; color: #f2f3f5; }
#subtitle { font-size: 12px; color: #8b919c; }
#video { background: #000; border: 1px solid #2a2e37; border-radius: 8px; }
#badge {
    border-radius: 14px; padding: 6px 18px;
    font-size: 14px; font-weight: 700; color: #fff; background: #2e9e5b;
}
#badge[alarm="true"] { background: #d93025; }
QFrame#card { background: #1e2128; border: 1px solid #2a2e37; border-radius: 8px; }
QLabel#cardTitle { color: #8b919c; font-size: 12px; }
QLabel#cardValue { font-size: 20px; font-weight: 700; color: #f2f3f5; }
QPushButton {
    background: #2a2e37; border: 1px solid #363b46; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; color: #e8eaed;
}
QPushButton:hover { background: #363b46; }
QPushButton:pressed { background: #232730; }
QPushButton#warn { background: #b3261e; border-color: #b3261e; font-weight: 700; }
QPushButton#warn:hover { background: #d93025; }
QProgressBar {
    background: #2a2e37; border: none; border-radius: 5px;
    height: 10px; text-align: center; color: transparent;
}
QProgressBar::chunk { background: #f9ab00; border-radius: 5px; }
QListWidget {
    background: #1e2128; border: 1px solid #2a2e37; border-radius: 8px;
    font-size: 13px; padding: 4px;
}
QListWidget::item { padding: 3px 6px; }
QListWidget::item:selected { background: #2a2e37; }
"""


class QtUI(BaseUI):
    def __init__(self, state, app, window_w=1024):
        super().__init__(state, app)
        from PySide6 import QtCore, QtGui, QtWidgets
        self.QtCore, self.QtGui, self.QtWidgets = QtCore, QtGui, QtWidgets

        self.window_w = window_w
        self._qimage = None  # 持有像素缓冲引用, 防止被 GC

        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.qt_app.setStyleSheet(QSS)
        self.qt_app.setFont(QtGui.QFont("Noto Sans CJK SC", 10))

        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(tr("win.title"))
        avail = self.win.screen().availableGeometry()
        self.win.setMinimumSize(min(640, avail.width()),
                                min(480, avail.height()))
        self.win.resize(min(window_w + 280, avail.width() - 20),
                        min(748, avail.height() - 20))

        root = QtWidgets.QVBoxLayout(self.win)
        root.setContentsMargins(12, 0, 12, 12)
        root.setSpacing(8)

        # ---- 顶栏: 标题 + 状态徽标 ----
        header = QtWidgets.QFrame(objectName="header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(4, 6, 4, 6)
        tw = QtWidgets.QVBoxLayout()
        tw.setSpacing(0)
        t = QtWidgets.QLabel(tr("header.title"), objectName="title")
        st = QtWidgets.QLabel(tr("header.subtitle"), objectName="subtitle")
        self._title_lb = t
        self._subtitle_lb = st
        tw.addWidget(t)
        tw.addWidget(st)
        hl.addLayout(tw)
        hl.addStretch(1)
        self.badge = QtWidgets.QLabel(tr("badge.normal"), objectName="badge")
        hl.addWidget(self.badge)
        self.lang_btn = QtWidgets.QPushButton(tr("lang.btn"))
        self.lang_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.lang_btn.clicked.connect(self._toggle_lang)
        hl.addWidget(self.lang_btn)
        root.addWidget(header)

        # ---- 中部: 视频 + 右侧面板 ----
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(10)
        self.video = QtWidgets.QLabel(objectName="video")
        self.video.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(360, 240)
        mid.addWidget(self.video, stretch=1)

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(4)
        self.cards = {}
        self.card_titles = {}
        for key in ("fps", "infer", "person", "noise"):
            side.addWidget(self._make_card(key))
        # 声音强度卡
        sound_card = QtWidgets.QFrame(objectName="card")
        sv = QtWidgets.QVBoxLayout(sound_card)
        sv.setContentsMargins(14, 8, 14, 8)
        sv.setSpacing(4)
        self.sound_title = QtWidgets.QLabel(tr("card.sound"),
                                            objectName="cardTitle")
        sv.addWidget(self.sound_title)
        self.sound_value = QtWidgets.QLabel("-- dB", objectName="cardValue")
        sv.addWidget(self.sound_value)
        self.sound_bar = QtWidgets.QProgressBar()
        self.sound_bar.setMinimumHeight(6)
        self.sound_bar.setRange(0, 60)
        self.sound_bar.setTextVisible(False)
        sv.addWidget(self.sound_bar)
        side.addWidget(sound_card)
        side.addStretch(1)

        self.buttons = {}
        for key, slot, obj in (("btn.test_alarm", self._test, "warn"),
                               ("btn.mute", self._mute, None),
                               ("btn.manual", self._manual, None),
                               ("btn.quit", self._on_close, None)):
            b = QtWidgets.QPushButton(tr(key))
            if obj:
                b.setObjectName(obj)
            b.clicked.connect(slot)
            side.addWidget(b)
            self.buttons[key] = b
        mid.addLayout(side)
        root.addLayout(mid, stretch=1)

        # ---- 底部: 事件日志 ----
        log_frame = QtWidgets.QVBoxLayout()
        self.log_cap = QtWidgets.QLabel(tr("log.caption"))
        self.log_cap.setStyleSheet("color:#8b919c; font-size:12px;")
        log_frame.addWidget(self.log_cap)
        self.log = QtWidgets.QListWidget()
        self.log.setMinimumHeight(50)
        log_frame.addWidget(self.log)
        root.addLayout(log_frame)

        self._n_events = -1

        self.timer = QtCore.QTimer(self.win)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _make_card(self, key):
        QtWidgets = self.QtWidgets
        card = QtWidgets.QFrame(objectName="card")
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(14, 6, 14, 6)
        v.setSpacing(2)
        title = QtWidgets.QLabel(tr("card." + key), objectName="cardTitle")
        v.addWidget(title)
        val = QtWidgets.QLabel("-", objectName="cardValue")
        v.addWidget(val)
        self.cards[key] = val
        self.card_titles[key] = title
        return card

    # ---- 中英文切换 ----
    def _toggle_lang(self):
        set_lang("en" if get_lang() == "zh" else "zh")
        self._apply_lang()

    def _apply_lang(self):
        self.win.setWindowTitle(tr("win.title"))
        self._title_lb.setText(tr("header.title"))
        self._subtitle_lb.setText(tr("header.subtitle"))
        self.lang_btn.setText(tr("lang.btn"))
        self.log_cap.setText(tr("log.caption"))
        self.sound_title.setText(tr("card.sound"))
        for key, lb in self.card_titles.items():
            lb.setText(tr("card." + key))
        for key, b in self.buttons.items():
            b.setText(tr(key))
        self._n_events = -1          # 强制下一帧按新语言重绘事件日志

    # ---- 按钮动作 ----
    def _test(self):
        self.app.trigger_alarm(duration=5)

    def _mute(self):
        self.app.toggle_mute()

    def _manual(self):
        self.app.manual_record()

    def _on_close(self):
        self.state.quit = True

    # ---- 主循环 ----
    def run(self):
        self.win.show()
        self.qt_app.aboutToQuit.connect(lambda: setattr(self.state, "quit", True))
        self.qt_app.exec()

    def close(self):
        try:
            self.qt_app.quit()
        except Exception:
            pass

    def _tick(self):
        QtWidgets = self.QtWidgets
        if self.state.quit:
            self.qt_app.quit()
            return
        st = self.state
        with st.lock:
            frame = None if st.frame is None else st.frame.copy()
            fps, ms = st.fps, st.infer_ms
            persons = st.person_count
            fall = st.fall_state
            db, nf = st.sound_db, st.noise_db
            alarm = st.alarm_active
            muted = st.muted
            events = list(st.events)

        # 视频帧
        if frame is not None:
            QtGui = self.QtGui
            h, w = frame.shape[:2]
            vw, vh = self.video.width() - 2, self.video.height() - 2
            if vw > 16 and vh > 16:
                scale = min(vw / w, vh / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self._qimage = QtGui.QImage(rgb.data, w, h, ch * w,
                                        QtGui.QImage.Format.Format_RGB888)
            self.video.setPixmap(QtGui.QPixmap.fromImage(self._qimage))

        # 状态徽标
        if fall == "DOWN":
            self.badge.setText(tr("badge.fall") if alarm
                               else tr("badge.fall_alarm"))
            self.badge.setProperty("alarm", "true")
        else:
            self.badge.setText(tr("badge.normal_muted") if muted
                               else tr("badge.normal"))
            self.badge.setProperty("alarm", "false")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)

        # 指标卡
        self.cards["fps"].setText(f"{fps:.1f}")
        self.cards["infer"].setText(f"{ms:.0f} ms")
        self.cards["person"].setText(str(persons))
        self.cards["noise"].setText(f"{nf:.0f} dB")
        self.sound_value.setText(f"{db:.0f} dB")
        self.sound_bar.setValue(int(max(0, db) + 60) if db > -60 else 0)

        # 事件日志
        if len(events) != self._n_events:
            self._n_events = len(events)
            self.log.clear()
            for ev in reversed(events):
                self.log.addItem(format_event(ev))


# ------------------------------------------------------------------
# Tkinter UI (PySide6 不可用时回退)
# ------------------------------------------------------------------
class TkUI(BaseUI):
    def __init__(self, state, app, window_w=1024):
        super().__init__(state, app)
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(tr("win.title"))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        from PIL import Image, ImageTk
        self.PIL = (Image, ImageTk)
        self.window_w = window_w

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side="top", fill="both", expand=True)

        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="top", fill="x")

        self.status_var = tk.StringVar(value=tr("status.init"))
        self.fps_var = tk.StringVar(value=tr("status.metrics_none"))
        self.sound_var = tk.StringVar(value=tr("status.sound_none"))
        ttk.Label(bar, textvariable=self.status_var, font=("Sans", 13, "bold"),
                  foreground="#c0392b").pack(side="left", padx=8)
        ttk.Label(bar, textvariable=self.fps_var).pack(side="left", padx=8)
        ttk.Label(bar, textvariable=self.sound_var).pack(side="left", padx=8)

        # 声音强度条
        self.pb = ttk.Progressbar(bar, length=180, maximum=60)
        self.pb.pack(side="left", padx=8)

        btns = ttk.Frame(bar)
        btns.pack(side="right")
        self.buttons = {}
        for key, slot in (("btn.test_alarm", self._test),
                          ("btn.mute", self._mute),
                          ("btn.manual", self._manual),
                          ("btn.quit", self._on_close)):
            b = ttk.Button(btns, text=tr(key), command=slot)
            b.pack(side="left", padx=4)
            self.buttons[key] = b
        self.lang_btn = ttk.Button(bar, text=tr("lang.btn"),
                                   command=self._toggle_lang)
        self.lang_btn.pack(side="right", padx=4)

        self.log_frame = ttk.LabelFrame(self.root, text=tr("log.caption"),
                                        padding=4)
        self.log_frame.pack(side="bottom", fill="x")
        self.listbox = tk.Listbox(self.log_frame, height=5)
        self.listbox.pack(fill="x")
        self._force_log = False

    def _test(self):
        self.app.trigger_alarm(duration=5)

    def _mute(self):
        self.app.toggle_mute()

    def _manual(self):
        self.app.manual_record()

    def _on_close(self):
        self.state.quit = True
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---- 中英文切换 ----
    def _toggle_lang(self):
        set_lang("en" if get_lang() == "zh" else "zh")
        self._apply_lang()

    def _apply_lang(self):
        self.root.title(tr("win.title"))
        self.lang_btn.configure(text=tr("lang.btn"))
        self.log_frame.configure(text=tr("log.caption"))
        for key, b in self.buttons.items():
            b.configure(text=tr(key))
        self._force_log = True

    def run(self):
        self.root.after(33, self._tick)
        self.root.mainloop()

    def _tick(self):
        if self.state.quit:
            # shutdown (SIGTERM/退出按钮) 后销毁窗口, 让 mainloop 返回进程正常退出
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        st = self.state
        with st.lock:
            frame = None if st.frame is None else st.frame.copy()
            fps, ms = st.fps, st.infer_ms
            persons = st.person_count
            fall = st.fall_state
            db, nf = st.sound_db, st.noise_db
            alarm = st.alarm_active
            muted = st.muted
            events = list(st.events)

        if frame is not None:
            Image, ImageTk = self.PIL
            h, w = frame.shape[:2]
            scale = self.window_w / w
            frame = cv2.resize(frame, (self.window_w, int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self._photo)

        if fall == "DOWN":
            self.status_var.set(tr("status.down") if alarm
                                else tr("status.down_alarm"))
        else:
            self.status_var.set(tr("status.normal"))
        self.fps_var.set(tr("status.metrics", fps=fps, ms=ms,
                            persons=persons))
        self.sound_var.set(tr("status.sound", db=db, nf=nf))
        self.pb["value"] = max(0, db + 60)

        if self._force_log or self.listbox.size() != len(events):
            self._force_log = False
            self.listbox.delete(0, "end")
            for ev in reversed(events):
                self.listbox.insert("end", format_event(ev))

        self.root.after(33, self._tick)


# ------------------------------------------------------------------
# OpenCV 窗口 UI (无 PySide6/tkinter/PIL 时回退)
# ------------------------------------------------------------------
class CvUI(BaseUI):
    def __init__(self, state, app, window_w=1024):
        super().__init__(state, app)
        self.window_w = window_w
        self.win = tr("cv.hint")
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    def run(self):
        while not self.state.quit:
            with self.state.lock:
                frame = None if self.state.frame is None else self.state.frame.copy()
                fall = self.state.fall_state
            if frame is None:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    self.state.quit = True
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (self.window_w, int(h * self.window_w / w)))
            cv2.putText(frame, f"STATE: {fall}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255) if fall == "DOWN" else (0, 200, 0), 2)
            cv2.imshow(self.win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.state.quit = True
            elif key == ord("a"):
                self.app.trigger_alarm(5)
            elif key == ord("m"):
                self.app.toggle_mute()
            elif key == ord("r"):
                self.app.manual_record()
            elif key == ord("l"):
                set_lang("en" if get_lang() == "zh" else "zh")
                cv2.setWindowTitle(self.win, tr("cv.hint"))
        cv2.destroyAllWindows()

    def close(self):
        cv2.destroyAllWindows()


class HeadlessUI(BaseUI):
    def run(self):
        import time
        last = ""
        while not self.state.quit:
            with self.state.lock:
                fall, db = self.state.fall_state, self.state.sound_db
            line = f"[headless] state={fall} sound={db:.0f}dB"
            if line != last:
                print(line)
                last = line
            time.sleep(1)


def make_ui(state, app, ui_cfg, no_ui=False):
    if no_ui:
        return HeadlessUI(state, app)
    try:
        import PySide6  # noqa
        import os
        if not os.environ.get("DISPLAY") and os.name == "posix":
            raise RuntimeError("无 DISPLAY")
        return QtUI(state, app, ui_cfg.get("window_w", 1024))
    except Exception as e:
        print(f"[ui] PySide6 不可用({e}), 回退到 Tkinter")
    try:
        import tkinter  # noqa
        import PIL  # noqa
        return TkUI(state, app, ui_cfg.get("window_w", 1024))
    except Exception as e:
        print(f"[ui] Tkinter 不可用({e}), 回退到 OpenCV 窗口")
        try:
            import os
            if not os.environ.get("DISPLAY"):
                raise RuntimeError("无 DISPLAY, 使用无界面模式")
            return CvUI(state, app, ui_cfg.get("window_w", 1024))
        except Exception as e2:
            print(f"[ui] OpenCV 窗口不可用({e2}), 使用无界面模式")
            return HeadlessUI(state, app)


# 便于外部标注帧 (避免 UI 与推理耦合)
def annotate_frame(frame, detections, fallen_idx=None):
    for i, d in enumerate(detections):
        draw_pose(frame, d, fallen=(fallen_idx is not None and i == fallen_idx))
    return frame
