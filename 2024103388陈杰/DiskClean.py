"""
DiskCleaner Pro - 高级磁盘清理工具
功能：
1. 扫描并清理系统临时文件
2. 查找并删除重复文件
3. 分析大文件占用
4. 清理日志文件
5. 清理浏览器缓存
6. 清理回收站
7. 生成清理报告
8. 图形用户界面
"""

import os
import sys
import shutil
import hashlib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import datetime
import tempfile
import json
import ctypes
import winreg
import psutil
import glob
import threading
import time

# 常量定义
VERSION = "1.0"
AUTHOR = "Advanced Software Engineering Course"
DEFAULT_SCAN_PATHS = [
    os.path.join(os.environ['USERPROFILE'], 'AppData', 'Local', 'Temp'),
    os.path.join(os.environ['USERPROFILE'], 'Downloads'),
    os.path.join(os.environ['USERPROFILE'], 'Desktop'),
    "C:\\Windows\\Temp"
]

# 浏览器缓存路径
BROWSER_CACHE_PATHS = {
    "Chrome": os.path.join(os.environ['LOCALAPPDATA'], 'Google', 'Chrome', 'User Data', 'Default', 'Cache'),
    "Firefox": os.path.join(os.environ['APPDATA'], 'Mozilla', 'Firefox', 'Profiles'),
    "Edge": os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'Edge', 'User Data', 'Default', 'Cache'),
    "Opera": os.path.join(os.environ['APPDATA'], 'Opera Software', 'Opera Stable', 'Cache')
}

# 日志文件路径
LOG_PATHS = [
    os.path.join(os.environ['WINDIR'], 'Logs'),
    os.path.join(os.environ['WINDIR'], 'System32', 'LogFiles'),
    os.path.join(os.environ['PROGRAMDATA'], 'Microsoft', 'Diagnosis', 'Logs')
]


class DiskScanner:
    """磁盘扫描和分析工具类"""

    def __init__(self):
        self.scan_results = {
            'temp_files': [],
            'duplicate_files': {},
            'large_files': [],
            'log_files': [],
            'browser_cache': [],
            'total_size': 0,
            'reclaimable': 0
        }
        self.scan_progress = 0
        self.scan_active = False
        self.cancel_scan = False
        self.last_scan_time = None

    def format_size(self, size_bytes):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def calculate_folder_size(self, folder_path):
        """计算文件夹大小"""
        total_size = 0
        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    continue
        return total_size

    def scan_temp_files(self, paths=DEFAULT_SCAN_PATHS):
        """扫描临时文件"""
        temp_files = []
        total_size = 0

        for path in paths:
            if not os.path.exists(path):
                continue

            for root, _, files in os.walk(path):
                if self.cancel_scan:
                    return

                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # 跳过正在使用的文件
                        if self.is_file_in_use(file_path):
                            continue

                        file_size = os.path.getsize(file_path)
                        file_age = (datetime.datetime.now() -
                                    datetime.datetime.fromtimestamp(os.path.getmtime(file_path)))

                        # 只考虑大于1KB且超过7天的文件
                        if file_size > 1024 and file_age.days > 7:
                            temp_files.append({
                                'path': file_path,
                                'size': file_size,
                                'modified': os.path.getmtime(file_path)
                            })
                            total_size += file_size
                    except OSError:
                        continue

        self.scan_results['temp_files'] = sorted(temp_files, key=lambda x: x['size'], reverse=True)
        self.scan_results['reclaimable'] += total_size
        self.scan_results['total_size'] += total_size
        return temp_files

    def scan_duplicate_files(self, paths=DEFAULT_SCAN_PATHS):
        """扫描重复文件"""
        file_hashes = {}
        duplicates = {}

        for path in paths:
            if not os.path.exists(path):
                continue

            for root, _, files in os.walk(path):
                if self.cancel_scan:
                    return

                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # 跳过小于1KB的文件
                        if os.path.getsize(file_path) < 1024:
                            continue

                        # 计算文件哈希
                        file_hash = self.calculate_file_hash(file_path)

                        if file_hash in file_hashes:
                            if file_hash not in duplicates:
                                duplicates[file_hash] = [file_hashes[file_hash]]
                            duplicates[file_hash].append({
                                'path': file_path,
                                'size': os.path.getsize(file_path)
                            })
                        else:
                            file_hashes[file_hash] = {
                                'path': file_path,
                                'size': os.path.getsize(file_path)
                            }
                    except (OSError, PermissionError):
                        continue

        self.scan_results['duplicate_files'] = duplicates

        # 计算可回收空间
        dup_size = 0
        for hash_group in duplicates.values():
            # 每组重复文件中保留一个，其余可删除
            dup_size += sum(f['size'] for f in hash_group[1:])

        self.scan_results['reclaimable'] += dup_size
        self.scan_results['total_size'] += dup_size
        return duplicates

    def scan_large_files(self, paths=DEFAULT_SCAN_PATHS, threshold=100 * 1024 * 1024):
        """扫描大文件"""
        large_files = []

        for path in paths:
            if not os.path.exists(path):
                continue

            for root, _, files in os.walk(path):
                if self.cancel_scan:
                    return

                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > threshold:
                            large_files.append({
                                'path': file_path,
                                'size': file_size,
                                'modified': os.path.getmtime(file_path)
                            })
                    except OSError:
                        continue

        self.scan_results['large_files'] = sorted(large_files, key=lambda x: x['size'], reverse=True)
        return large_files

    def scan_log_files(self, paths=LOG_PATHS, max_age=30):
        """扫描日志文件"""
        log_files = []
        total_size = 0
        now = time.time()

        for path in paths:
            if not os.path.exists(path):
                continue

            for root, _, files in os.walk(path):
                if self.cancel_scan:
                    return

                for file in files:
                    if not file.lower().endswith(('.log', '.txt', '.dmp')):
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        file_age = now - os.path.getmtime(file_path)

                        # 只考虑超过指定天数的日志文件
                        if file_age > max_age * 86400:
                            log_files.append({
                                'path': file_path,
                                'size': file_size,
                                'modified': os.path.getmtime(file_path)
                            })
                            total_size += file_size
                    except OSError:
                        continue

        self.scan_results['log_files'] = sorted(log_files, key=lambda x: x['size'], reverse=True)
        self.scan_results['reclaimable'] += total_size
        self.scan_results['total_size'] += total_size
        return log_files

    def scan_browser_cache(self):
        """扫描浏览器缓存"""
        cache_files = []
        total_size = 0

        for browser, path in BROWSER_CACHE_PATHS.items():
            if not os.path.exists(path):
                continue

            # Firefox有多个配置文件
            if browser == "Firefox":
                profiles = glob.glob(os.path.join(path, '*.default*'))
                for profile in profiles:
                    cache_path = os.path.join(profile, 'cache2')
                    if os.path.exists(cache_path):
                        cache_files.extend(self._scan_cache_dir(cache_path, browser))
            else:
                cache_files.extend(self._scan_cache_dir(path, browser))

        # 计算总大小
        for item in cache_files:
            total_size += item['size']

        self.scan_results['browser_cache'] = cache_files
        self.scan_results['reclaimable'] += total_size
        self.scan_results['total_size'] += total_size
        return cache_files

    def _scan_cache_dir(self, path, browser):
        """扫描缓存目录"""
        cache_items = []

        for root, _, files in os.walk(path):
            if self.cancel_scan:
                return cache_items

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # 跳过小文件
                    file_size = os.path.getsize(file_path)
                    if file_size < 1024:
                        continue

                    cache_items.append({
                        'path': file_path,
                        'size': file_size,
                        'browser': browser,
                        'modified': os.path.getmtime(file_path)
                    })
                except OSError:
                    continue

        return cache_items

    def calculate_file_hash(self, file_path, block_size=65536):
        """计算文件哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()

    def is_file_in_use(self, file_path):
        """检查文件是否正在使用"""
        try:
            # 尝试打开文件以检查是否被占用
            with open(file_path, 'a') as f:
                pass
            return False
        except IOError:
            return True

    def empty_recycle_bin(self):
        """清空回收站"""
        try:
            # Windows API 清空回收站
            SHEmptyRecycleBin = ctypes.windll.shell32.SHEmptyRecycleBinW
            SHEmptyRecycleBin(None, None, 1)
            return True
        except Exception:
            return False

    def delete_files(self, file_list):
        """删除文件列表"""
        total_size = 0
        success_count = 0
        fail_count = 0

        for file_path in file_list:
            try:
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    total_size += file_size
                    success_count += 1
                elif os.path.isdir(file_path):
                    dir_size = self.calculate_folder_size(file_path)
                    shutil.rmtree(file_path)
                    total_size += dir_size
                    success_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
                fail_count += 1

        return success_count, fail_count, total_size

    def full_scan(self, callback=None):
        """执行完整扫描"""
        self.scan_active = True
        self.cancel_scan = False
        self.scan_results = {
            'temp_files': [],
            'duplicate_files': {},
            'large_files': [],
            'log_files': [],
            'browser_cache': [],
            'total_size': 0,
            'reclaimable': 0
        }

        start_time = time.time()

        # 创建扫描线程
        scan_thread = threading.Thread(target=self._perform_scan, args=(callback,))
        scan_thread.start()

        return scan_thread

    def _perform_scan(self, callback=None):
        """执行扫描的实际工作"""
        try:
            # 扫描临时文件
            self.scan_temp_files()
            if callback: callback(25)

            # 扫描重复文件
            self.scan_duplicate_files()
            if callback: callback(50)

            # 扫描大文件
            self.scan_large_files()
            if callback: callback(65)

            # 扫描日志文件
            self.scan_log_files()
            if callback: callback(80)

            # 扫描浏览器缓存
            self.scan_browser_cache()
            if callback: callback(95)

            self.last_scan_time = datetime.datetime.now()
            self.scan_active = False

            if callback: callback(100)
        except Exception as e:
            print(f"Scan error: {str(e)}")
            self.scan_active = False
            if callback: callback(0, error=str(e))


class DiskCleanerApp(tk.Tk):
    """磁盘清理工具图形界面"""

    def __init__(self):
        super().__init__()
        self.title(f"DiskCleaner Pro v{VERSION}")
        self.geometry("900x700")
        self.minsize(800, 600)

        # 创建扫描器实例
        self.scanner = DiskScanner()

        # 创建界面元素
        self.create_widgets()

        # 加载上次扫描结果
        self.load_last_scan()

        # 设置样式
        self.set_style()

    def set_style(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 配置颜色
        self.configure(background='#f0f0f0')

        # 配置标签样式
        style.configure('Title.TLabel',
                        font=('Arial', 16, 'bold'),
                        background='#f0f0f0',
                        foreground='#2c3e50')

        style.configure('Subtitle.TLabel',
                        font=('Arial', 11, 'bold'),
                        background='#f0f0f0',
                        foreground='#34495e')

        style.configure('Status.TLabel',
                        font=('Arial', 9),
                        background='#f0f0f0',
                        foreground='#7f8c8d')

        # 配置按钮样式
        style.configure('TButton',
                        font=('Arial', 10),
                        padding=6)

        style.configure('Primary.TButton',
                        background='#3498db',
                        foreground='white',
                        font=('Arial', 10, 'bold'))

        style.map('Primary.TButton',
                  background=[('active', '#2980b9')])

        style.configure('Danger.TButton',
                        background='#e74c3c',
                        foreground='white',
                        font=('Arial', 10, 'bold'))

        style.map('Danger.TButton',
                  background=[('active', '#c0392b')])

        # 配置进度条样式
        style.configure('Horizontal.TProgressbar',
                        thickness=20,
                        background='#3498db',
                        troughcolor='#ecf0f1')

        # 配置树状视图样式
        style.configure('Treeview',
                        font=('Arial', 9),
                        rowheight=25)

        style.configure('Treeview.Heading',
                        font=('Arial', 10, 'bold'))

        style.map('Treeview',
                  background=[('selected', '#3498db')])

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 标题区域
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="DiskCleaner Pro", style='Title.TLabel').pack(side=tk.LEFT)

        status_label = ttk.Label(title_frame, text="就绪", style='Status.TLabel')
        status_label.pack(side=tk.RIGHT)
        self.status_label = status_label

        # 磁盘信息区域
        disk_frame = ttk.LabelFrame(main_frame, text="磁盘信息")
        disk_frame.pack(fill=tk.X, pady=5)

        # 获取磁盘信息
        disk_info = self.get_disk_info()

        for i, (drive, info) in enumerate(disk_info.items()):
            frame = ttk.Frame(disk_frame)
            frame.grid(row=0, column=i, padx=10, pady=5)

            ttk.Label(frame, text=f"{drive} ({info['fstype']})", font=('Arial', 10, 'bold')).pack(anchor=tk.W)

            # 使用画布创建自定义进度条
            canvas = tk.Canvas(frame, width=150, height=20, bg='#ecf0f1', highlightthickness=0)
            canvas.pack(pady=2)

            # 计算已用空间比例
            used_percent = info['used'] / info['total'] if info['total'] > 0 else 0
            used_width = 150 * used_percent

            # 绘制进度条
            canvas.create_rectangle(0, 0, used_width, 20, fill='#3498db', outline='')
            canvas.create_rectangle(0, 0, 150, 20, outline='#bdc3c7')

            # 添加文本标签
            canvas.create_text(75, 10, text=f"{used_percent:.0%} 已用", fill='white' if used_percent > 0.5 else 'black')

            ttk.Label(frame, text=f"已用: {self.scanner.format_size(info['used'])} / "
                                  f"总共: {self.scanner.format_size(info['total'])}").pack(anchor=tk.W)

        # 扫描控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        scan_btn = ttk.Button(control_frame, text="开始扫描",
                              command=self.start_scan, style='Primary.TButton')
        scan_btn.pack(side=tk.LEFT, padx=(0, 10))

        clean_btn = ttk.Button(control_frame, text="清理选中项",
                               command=self.clean_selected, style='Danger.TButton')
        clean_btn.pack(side=tk.LEFT)

        empty_bin_btn = ttk.Button(control_frame, text="清空回收站",
                                   command=self.empty_recycle_bin)
        empty_bin_btn.pack(side=tk.RIGHT)

        # 进度条
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                       maximum=100, style='Horizontal.TProgressbar')
        progress_bar.pack(fill=tk.X)

        # 结果标签
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.X, pady=10)

        ttk.Label(results_frame, text="扫描结果概览", style='Subtitle.TLabel').pack(anchor=tk.W)

        # 创建结果概览的框架
        overview_frame = ttk.Frame(results_frame)
        overview_frame.pack(fill=tk.X, pady=10)

        # 结果概览卡片
        self.result_cards = {}
        categories = [
            ('temp_files', '临时文件', '#e74c3c'),
            ('duplicate_files', '重复文件', '#9b59b6'),
            ('large_files', '大文件', '#3498db'),
            ('log_files', '日志文件', '#f39c12'),
            ('browser_cache', '浏览器缓存', '#1abc9c')
        ]

        for i, (key, name, color) in enumerate(categories):
            card = ttk.Frame(overview_frame, relief=tk.RAISED, borderwidth=1)
            card.grid(row=0, column=i, padx=5, sticky=tk.NSEW)

            # 设置卡片列权重
            overview_frame.columnconfigure(i, weight=1)

            # 卡片标题
            ttk.Label(card, text=name, background=color, foreground='white',
                      font=('Arial', 10, 'bold'), anchor=tk.CENTER).pack(fill=tk.X)

            # 卡片内容
            content = ttk.Frame(card)
            content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            size_label = ttk.Label(content, text="0.00 B", font=('Arial', 11, 'bold'))
            size_label.pack()

            count_label = ttk.Label(content, text="0 项")
            count_label.pack()

            self.result_cards[key] = {'size': size_label, 'count': count_label}

        # 结果详情标签页
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建各个标签页
        self.create_temp_files_tab(notebook)
        self.create_duplicate_files_tab(notebook)
        self.create_large_files_tab(notebook)
        self.create_log_files_tab(notebook)
        self.create_browser_cache_tab(notebook)

        # 状态栏
        status_bar = ttk.Frame(self, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(status_bar, text=f"作者: {AUTHOR}", style='Status.TLabel').pack(side=tk.LEFT, padx=5)

        self.last_scan_label = ttk.Label(status_bar, text="上次扫描: 从未扫描", style='Status.TLabel')
        self.last_scan_label.pack(side=tk.RIGHT, padx=5)

    def create_temp_files_tab(self, notebook):
        """创建临时文件标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="临时文件")

        # 创建树状视图
        columns = ("path", "size", "modified")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")

        # 设置列
        tree.heading("path", text="文件路径")
        tree.heading("size", text="大小")
        tree.heading("modified", text="修改时间")

        tree.column("path", width=400)
        tree.column("size", width=100, anchor=tk.E)
        tree.column("modified", width=150, anchor=tk.E)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.temp_files_tree = tree

    def create_duplicate_files_tab(self, notebook):
        """创建重复文件标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="重复文件")

        # 创建树状视图
        columns = ("path", "size", "group")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")

        # 设置列
        tree.heading("path", text="文件路径")
        tree.heading("size", text="大小")
        tree.heading("group", text="重复组")

        tree.column("path", width=400)
        tree.column("size", width=100, anchor=tk.E)
        tree.column("group", width=100, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.duplicate_files_tree = tree

    def create_large_files_tab(self, notebook):
        """创建大文件标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="大文件")

        # 创建树状视图
        columns = ("path", "size", "modified")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")

        # 设置列
        tree.heading("path", text="文件路径")
        tree.heading("size", text="大小")
        tree.heading("modified", text="修改时间")

        tree.column("path", width=400)
        tree.column("size", width=100, anchor=tk.E)
        tree.column("modified", width=150, anchor=tk.E)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.large_files_tree = tree

    def create_log_files_tab(self, notebook):
        """创建日志文件标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="日志文件")

        # 创建树状视图
        columns = ("path", "size", "modified")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")

        # 设置列
        tree.heading("path", text="文件路径")
        tree.heading("size", text="大小")
        tree.heading("modified", text="修改时间")

        tree.column("path", width=400)
        tree.column("size", width=100, anchor=tk.E)
        tree.column("modified", width=150, anchor=tk.E)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_files_tree = tree

    def create_browser_cache_tab(self, notebook):
        """创建浏览器缓存标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="浏览器缓存")

        # 创建树状视图
        columns = ("path", "size", "browser", "modified")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")

        # 设置列
        tree.heading("path", text="文件路径")
        tree.heading("size", text="大小")
        tree.heading("browser", text="浏览器")
        tree.heading("modified", text="修改时间")

        tree.column("path", width=350)
        tree.column("size", width=100, anchor=tk.E)
        tree.column("browser", width=100, anchor=tk.CENTER)
        tree.column("modified", width=150, anchor=tk.E)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.browser_cache_tree = tree

    def get_disk_info(self):
        """获取磁盘信息"""
        disk_info = {}
        partitions = psutil.disk_partitions()

        for partition in partitions:
            if partition.fstype and 'fixed' in partition.opts:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.device] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent,
                        'fstype': partition.fstype
                    }
                except Exception:
                    continue

        return disk_info

    def start_scan(self):
        """开始扫描"""
        if self.scanner.scan_active:
            messagebox.showinfo("扫描中", "扫描已在运行中，请稍候...")
            return

        # 重置UI状态
        self.status_label.config(text="扫描中...")
        self.progress_var.set(0)

        # 清除之前的扫描结果
        for tree in [self.temp_files_tree, self.duplicate_files_tree,
                     self.large_files_tree, self.log_files_tree, self.browser_cache_tree]:
            for item in tree.get_children():
                tree.delete(item)

        # 开始扫描
        scan_thread = self.scanner.full_scan(self.update_progress)
        self.after(100, self.check_scan_status, scan_thread)

    def update_progress(self, value, error=None):
        """更新进度条"""
        self.progress_var.set(value)

        if error:
            messagebox.showerror("扫描错误", f"扫描过程中发生错误:\n{error}")
            self.status_label.config(text="扫描失败")
            return

        if value == 100:
            # 扫描完成，更新UI
            self.update_results_ui()
            self.status_label.config(text="扫描完成")
            self.save_scan_results()

    def check_scan_status(self, scan_thread):
        """检查扫描状态"""
        if scan_thread.is_alive():
            self.after(100, self.check_scan_status, scan_thread)
        else:
            if not self.scanner.scan_active and self.scanner.last_scan_time:
                self.update_results_ui()
                self.status_label.config(text="扫描完成")

    def update_results_ui(self):
        """更新结果UI"""
        results = self.scanner.scan_results

        # 更新概览卡片
        for key, card in self.result_cards.items():
            if key == 'duplicate_files':
                # 特殊处理重复文件
                count = sum(len(group) for group in results[key].values())
                size = sum(f['size'] for group in results[key].values() for f in group[1:])
                card['count'].config(text=f"{count} 项")
                card['size'].config(text=self.scanner.format_size(size))
            else:
                count = len(results[key])
                size = sum(item['size'] for item in results[key])
                card['count'].config(text=f"{count} 项")
                card['size'].config(text=self.scanner.format_size(size))

        # 更新临时文件列表
        for item in results['temp_files']:
            self.temp_files_tree.insert("", tk.END, values=(
                item['path'],
                self.scanner.format_size(item['size']),
                datetime.datetime.fromtimestamp(item['modified']).strftime('%Y-%m-%d %H:%M:%S')
            ))

        # 更新重复文件列表
        group_id = 1
        for group in results['duplicate_files'].values():
            for item in group:
                self.duplicate_files_tree.insert("", tk.END, values=(
                    item['path'],
                    self.scanner.format_size(item['size']),
                    group_id
                ))
            group_id += 1

        # 更新大文件列表
        for item in results['large_files']:
            self.large_files_tree.insert("", tk.END, values=(
                item['path'],
                self.scanner.format_size(item['size']),
                datetime.datetime.fromtimestamp(item['modified']).strftime('%Y-%m-%d %H:%M:%S')
            ))

        # 更新日志文件列表
        for item in results['log_files']:
            self.log_files_tree.insert("", tk.END, values=(
                item['path'],
                self.scanner.format_size(item['size']),
                datetime.datetime.fromtimestamp(item['modified']).strftime('%Y-%m-%d %H:%M:%S')
            ))

        # 更新浏览器缓存列表
        for item in results['browser_cache']:
            self.browser_cache_tree.insert("", tk.END, values=(
                item['path'],
                self.scanner.format_size(item['size']),
                item['browser'],
                datetime.datetime.fromtimestamp(item['modified']).strftime('%Y-%m-%d %H:%M:%S')
            ))

        # 更新状态栏
        if self.scanner.last_scan_time:
            self.last_scan_label.config(
                text=f"上次扫描: {self.scanner.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                     f"可回收: {self.scanner.format_size(results['reclaimable'])}"
            )

    def clean_selected(self):
        """清理选中的项目"""
        if not messagebox.askyesno("确认清理", "确定要删除选中的文件吗？此操作不可撤销！"):
            return

        # 收集所有选中的文件
        files_to_delete = []

        # 临时文件
        for item in self.temp_files_tree.selection():
            path = self.temp_files_tree.item(item, 'values')[0]
            files_to_delete.append(path)

        # 重复文件
        for item in self.duplicate_files_tree.selection():
            path = self.duplicate_files_tree.item(item, 'values')[0]
            files_to_delete.append(path)

        # 大文件
        for item in self.large_files_tree.selection():
            path = self.large_files_tree.item(item, 'values')[0]
            files_to_delete.append(path)

        # 日志文件
        for item in self.log_files_tree.selection():
            path = self.log_files_tree.item(item, 'values')[0]
            files_to_delete.append(path)

        # 浏览器缓存
        for item in self.browser_cache_tree.selection():
            path = self.browser_cache_tree.item(item, 'values')[0]
            files_to_delete.append(path)

        # 删除文件
        success, fail, total_size = self.scanner.delete_files(files_to_delete)

        # 显示结果
        messagebox.showinfo("清理完成",
                            f"成功删除 {success} 个文件\n"
                            f"失败 {fail} 个文件\n"
                            f"回收空间: {self.scanner.format_size(total_size)}")

        # 重新扫描
        self.start_scan()

    def empty_recycle_bin(self):
        """清空回收站"""
        if not messagebox.askyesno("确认清空", "确定要清空回收站吗？此操作不可撤销！"):
            return

        if self.scanner.empty_recycle_bin():
            messagebox.showinfo("操作成功", "回收站已清空")
        else:
            messagebox.showerror("操作失败", "清空回收站时出错")

    def save_scan_results(self):
        """保存扫描结果"""
        results = {
            'scan_time': self.scanner.last_scan_time.isoformat(),
            'results': self.scanner.scan_results
        }

        # 转换不可序列化的数据
        for key in results['results']:
            if key == 'duplicate_files':
                # 将字典转换为列表
                results['results'][key] = list(results['results'][key].values())

        # 保存到文件
        with open('diskcleaner_scan.json', 'w') as f:
            json.dump(results, f, indent=2)

    def load_last_scan(self):
        """加载上次扫描结果"""
        try:
            if os.path.exists('diskcleaner_scan.json'):
                with open('diskcleaner_scan.json', 'r') as f:
                    data = json.load(f)

                # 转换时间
                self.scanner.last_scan_time = datetime.datetime.fromisoformat(data['scan_time'])

                # 恢复扫描结果
                self.scanner.scan_results = data['results']

                # 将重复文件列表转回字典格式
                dup_dict = {}
                for group in self.scanner.scan_results['duplicate_files']:
                    if group:
                        # 使用第一个文件的哈希作为键
                        file_hash = self.scanner.calculate_file_hash(group[0]['path'])
                        dup_dict[file_hash] = group
                self.scanner.scan_results['duplicate_files'] = dup_dict

                # 更新UI
                self.update_results_ui()
                self.status_label.config(text="已加载上次扫描结果")
                self.last_scan_label.config(
                    text=f"上次扫描: {self.scanner.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                         f"可回收: {self.scanner.format_size(self.scanner.scan_results['reclaimable'])}"
                )
        except Exception as e:
            print(f"加载扫描结果失败: {str(e)}")


if __name__ == "__main__":
    # 创建并运行应用
    app = DiskCleanerApp()
    app.mainloop()