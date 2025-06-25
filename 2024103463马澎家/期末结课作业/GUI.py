import os.path
import sys
import requests
from Bio.PDB import PDBParser
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import io
from MU3DSPstar import Mu3DSP_dssp_aafs_aap
from get_esm import get_esm_fea
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import seq1
import numpy as np
from predict_linear import predict
from functools import partial
import torch
import esm
from Bio import SeqIO
from PyQt5.QtWidgets import QTextEdit

temp = 0
# 20种常用氨基酸字典
amino_acids = {
    'A': 'Alanine (Ala)',
    'R': 'Arginine (Arg)',
    'N': 'Asparagine (Asn)',
    'D': 'Aspartic acid (Asp)',
    'C': 'Cysteine (Cys)',
    'E': 'Glutamic acid (Glu)',
    'Q': 'Glutamine (Gln)',
    'G': 'Glycine (Gly)',
    'H': 'Histidine (His)',
    'I': 'Isoleucine (Ile)',
    'L': 'Leucine (Leu)',
    'K': 'Lysine (Lys)',
    'M': 'Methionine (Met)',
    'F': 'Phenylalanine (Phe)',
    'P': 'Proline (Pro)',
    'S': 'Serine (Ser)',
    'T': 'Threonine (Thr)',
    'W': 'Tryptophan (Trp)',
    'Y': 'Tyrosine (Tyr)',
    'V': 'Valine (Val)'
}


# 在GUI应用中，避免耗时操作阻塞主线程是保持界面响应性的关键

# 我们将耗时较多的计算函数放在一个工作线程中

class WorkerThread(QThread):    # 定义工作线程类    用于模型预测
    calculationDone = pyqtSignal(float)  # 定义一个信号用于计算完成后更新UI
    errorOccurred = pyqtSignal(str)  # 自定义信号来指示错误

    def __init__(self, position_int, wild_res, mutation_res, sequence, model, alphabet, device):
        super(WorkerThread, self).__init__()
        self.position = position_int
        self.wild_res = wild_res
        self.mut_res = mutation_res
        self.wild_seq = sequence
        self.mut_seq = sequence[:position_int] + mutation_res + sequence[position_int + 1:]
        self.model = model
        self.alphabet = alphabet
        self.device = device

    def run(self):                       # 执行耗时操作  run方法是线程的入口点
        try:
            # print(f"突变后的序列为{self.mut_seq}")
            g2s_fea_list = Mu3DSP_dssp_aafs_aap(self.wild_res, self.mut_res, self.position, self.wild_seq)

            g2s_fea = np.array(g2s_fea_list)
            g2s_fea = g2s_fea.reshape(1, -1)
            # print(f"获取了G2S特征，特征维度为{g2s_fea.shape}")

            esm_fea_tensor = get_esm_fea(self.wild_seq, self.mut_seq, self.model, self.alphabet, self.device)
            esm_fea = esm_fea_tensor.detach().cpu().numpy()
            esm_fea = esm_fea.reshape(1, -1)
            print("完成了特征提取")
            # print(f"获取了ESM特征。特征w维度为{esm_fea.shape}")
            print("开始预测")
            result = predict(esm_fea, g2s_fea)
            print("得到预测结果")
            result = result.detach().cpu().numpy()
            result = result[0][0]
            # print(result)
            result = round(result, 4)
            self.calculationDone.emit(result)
        except Exception as e:
            self.errorOccurred.emit(str(e))  # 发出错误信号


class DownloadThread(QThread):
    downloadDone = pyqtSignal(str)

    def __init__(self, db, protein_id):
        super(DownloadThread, self).__init__()
        self.db = db
        self.protein_id = protein_id.lower()

    def run(self):
        if self.db == "PDB":
            # 下载PDB图片
            file_image_path = f'image/{self.protein_id}.png'
            if not os.path.exists(file_image_path):
                image_url = f"https://cdn.rcsb.org/images/structures/{self.protein_id}_assembly-1.jpeg"
                response = requests.get(image_url)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    img.save(file_image_path, "PNG")
                else:
                    self.downloadDone.emit(f"Failed to download image for PDB ID: {self.protein_id}")
                    return

            # 下载PDB文件
            file_pdb_path = f'pdb/{self.protein_id}.pdb'
            if not os.path.exists(file_pdb_path):
                pdb_url = f"https://files.rcsb.org/download/{self.protein_id}.pdb"
                response = requests.get(pdb_url)
                if response.status_code == 200:
                    with open(file_pdb_path, "wb") as file:
                        file.write(response.content)
                else:
                    self.downloadDone.emit(f"Failed to download PDB file for PDB ID: {self.protein_id}")
                    return

            self.downloadDone.emit("Download complete!")

        else:  # UniProt 下载FASTA
            fasta_path = f'fasta/{self.protein_id}.fasta'
            try:
                if not os.path.exists(fasta_path):
                    url = f"https://www.uniprot.org/uniprot/{self.protein_id}.fasta"
                    response = requests.get(url)
                    if response.status_code == 200:
                        os.makedirs('fasta', exist_ok=True)
                        with open(fasta_path, 'w') as f:
                            f.write(response.text)
                    else:
                        self.downloadDone.emit(f"Failed to download FASTA for UniProt ID: {self.protein_id}")
                        return

                # 这里可以加解析FASTA逻辑，比如提取序列长度或简单验证
                with open(fasta_path) as f:
                    lines = f.readlines()
                    seq = ''.join(line.strip() for line in lines if not line.startswith('>'))
                    if len(seq) == 0:
                        self.downloadDone.emit(f"FASTA sequence empty for UniProt ID: {self.protein_id}")
                        return

                self.downloadDone.emit("Download complete!")
            except Exception as e:
                self.downloadDone.emit(f"Exception during UniProt download: {str(e)}")


class ProteinViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 模型加载（占位）
        self.model, self.alphabet, self.device = self.load_model()

        # 数据库选择框
        self.dbSelectComboBox = QComboBox()
        self.dbSelectComboBox.addItems(["PDB", "UniProt", "Manual Input"])
        self.dbSelectComboBox.currentIndexChanged.connect(self.updateUIForDBSelection)

        # 蛋白ID/序列标签（动态）
        self.proteinIDLabel = QLabel('Protein ID:')
        self.proteinIDInput = QLineEdit()

        # Manual 输入的序列框（默认隐藏）
        self.manualSequenceInput = QTextEdit()
        self.manualSequenceInput.setPlaceholderText("Paste your protein sequence here...")
        self.manualSequenceInput.setVisible(False)

        # 下载或处理按钮（文本动态改变）
        self.downloadButton = QPushButton('Download from PDB')
        self.downloadButton.clicked.connect(self.downloadData)

        # 图片显示（占位）
        self.imageLabel = QLabel(self)
        pixmap = QPixmap(400, 400)
        pixmap.fill(QColor('white'))
        self.imageLabel.setPixmap(pixmap)

        # 突变设置相关控件
        self.chainsComboBox = QComboBox()
        self.mutationPositionInput = QLineEdit()
        self.mutationTypeComboBox = QComboBox()

        for short_name, full_name in amino_acids.items():
            self.mutationTypeComboBox.addItem(f"{full_name} ({short_name})", short_name)

        self.confirmButton = QPushButton('Confirm')
        self.confirmButton.clicked.connect(self.confirmMutation)

        # 状态栏
        self.statusLabel = QLabel('System ready')

        # 添加控件到布局
        layout.addWidget(QLabel('Database:'))
        layout.addWidget(self.dbSelectComboBox)

        layout.addWidget(self.proteinIDLabel)
        layout.addWidget(self.proteinIDInput)
        layout.addWidget(self.manualSequenceInput)

        layout.addWidget(self.downloadButton)
        layout.addWidget(QLabel('Protein Image:'))
        layout.addWidget(self.imageLabel)
        layout.addWidget(QLabel('Chains:'))
        layout.addWidget(self.chainsComboBox)
        layout.addWidget(QLabel('Mutation Position:'))
        layout.addWidget(self.mutationPositionInput)
        layout.addWidget(QLabel('Mutation Residue:'))
        layout.addWidget(self.mutationTypeComboBox)
        layout.addWidget(self.confirmButton)
        layout.addWidget(self.statusLabel)

        self.setLayout(layout)
        self.setWindowTitle('SMFFDDG')
        self.show()

    def updateUIForDBSelection(self):
        db = self.dbSelectComboBox.currentText()

        if db == "PDB":
            self.downloadButton.setText("Download from PDB")
            self.proteinIDLabel.setText("Protein ID:")
            self.proteinIDInput.setVisible(True)
            self.manualSequenceInput.setVisible(False)

        elif db == "UniProt":
            self.downloadButton.setText("Download from UniProt")
            self.proteinIDLabel.setText("UniProt ID:")
            self.proteinIDInput.setVisible(True)
            self.manualSequenceInput.setVisible(False)

        elif db == "Manual Input":
            self.downloadButton.setText("Use Sequence Directly")
            self.proteinIDLabel.setText("Protein Sequence:")
            self.proteinIDInput.setVisible(False)
            self.manualSequenceInput.setVisible(True)


    def load_model(self):     # 主UI初始化时便加载好ESM模型
        print("正在加载ESM模型，请稍后.....")
        os.environ['TORCH_HOME'] = 'ESM2_model'
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        print("完成模型加载")
        # print(device)
        return model, alphabet, device

    def downloadData(self):
        db = self.dbSelectComboBox.currentText()

        if db == "Manual Input":
            self.statusLabel.setText("Using manually input sequence.")
            sequence = self.manualSequenceInput.toPlainText().strip()
            if not sequence:
                QMessageBox.warning(self, "Input Error", "Protein sequence is empty.")
                return
            # 修复关键：设置 dummy id/db 以便 onDownloadDone 不报错
            self.db = "Manual Input"
            self.protein_id = "Manual_Protein"

            self.onDownloadDone("Sequence input ready.")  # 复用处理逻辑（伪调用）
            return

        protein_id = self.proteinIDInput.text().strip()
        self.protein_id = protein_id
        self.db = db

        self.downloadThread = DownloadThread(db, protein_id)
        self.downloadThread.downloadDone.connect(self.onDownloadDone)
        self.statusLabel.setText(f"Downloading from {db}... Please wait.")
        self.downloadThread.start()

    def onDownloadDone(self, message):
        if "Failed to download" in message:
            QMessageBox.warning(self, "Download Error", message)
            return

        if self.db == "PDB":
            # PDB图片加载
            image_path = f"image/{self.protein_id}.png"
            image = QImage(image_path)
            if not image.isNull():
                self.imageLabel.setPixmap(QPixmap(image).scaled(400, 400, Qt.KeepAspectRatio))
            else:
                self.imageLabel.setText("Failed to load PNG image.")

            # PDB链解析
            pdb_file = f"pdb/{self.protein_id}.pdb"
            parser = PDBParser()
            try:
                structure = parser.get_structure(self.protein_id, pdb_file)
            except Exception as e:
                self.statusLabel.setText(f"Failed to parse PDB file: {str(e)}")
                self.chainsComboBox.clear()
                return

            self.chainsComboBox.clear()
            seen_chains = set()
            for model in structure:
                for chain in model:
                    if chain.id not in seen_chains:
                        self.chainsComboBox.addItem(chain.id)
                        seen_chains.add(chain.id)

        else:  # UniProt，链框清空即可，imageLabel不动
            self.chainsComboBox.clear()

        self.statusLabel.setText(message)

    def confirmMutation(self):
        position_str = self.mutationPositionInput.text().strip()
        selected_chain_id = self.chainsComboBox.currentText()
        pdb_id = self.proteinIDInput.text().lower().strip()
        pdb_path = f"pdb/{pdb_id}.pdb"
        fasta_path = f"fasta/{pdb_id}.fasta"
        db_source = self.dbSelectComboBox.currentText()

        try:
            sequence = ""

            if db_source == "PDB" and os.path.exists(pdb_path):
                parser = PDBParser()
                structure = parser.get_structure(pdb_id, pdb_path)
                selected_chain = structure[0][selected_chain_id]

                for residue in selected_chain:
                    if residue.id[0] == " ":  # 标准氨基酸
                        res_name = residue.resname
                        sequence += seq1(res_name, undef_code='X')
                print(f"链{selected_chain_id}的序列为(来源PDB): {sequence}")

            elif db_source == "UniProt" and os.path.exists(fasta_path):
                records = list(SeqIO.parse(fasta_path, "fasta"))
                if len(records) == 0:
                    QMessageBox.warning(self, "序列读取错误", "FASTA文件为空或格式错误")
                    return
                fasta_seq = str(records[0].seq)
                sequence = fasta_seq.upper()
                print(f"序列为(来源FASTA): {sequence}")

            elif db_source == "Manual Input":
                sequence = self.manualSequenceInput.toPlainText().strip().upper()
                if not sequence or any(c not in amino_acids.keys() for c in sequence):
                    QMessageBox.warning(self, "输入错误", "请输入有效的蛋白质序列（单字母代码）")
                    return
                print(f"序列为(来源Manual Input): {sequence}")
                selected_chain_id = "Manual"  # 或者直接置空

            else:
                QMessageBox.warning(self, "文件不存在", "找不到对应的PDB文件或FASTA序列文件")
                return

            if not position_str.isdigit():
                QMessageBox.warning(self, "残基位置输入错误", "您输入的突变位置不是数字")
                return

            position_int = int(position_str) - 1
            if position_int < 0 or position_int >= len(sequence):
                QMessageBox.warning(self, "位置错误",
                                    f"突变位置超出蛋白序列范围（长度 {len(sequence)}）")
                return

            mutation_res = self.mutationTypeComboBox.currentData()
            wild_res = sequence[position_int]

            if mutation_res == wild_res:
                QMessageBox.warning(self, "突变残基选择错误",
                                    "选择的突变残基与野生型残基一致，请重新选择")
                return

            print(f"突变信息 - 位置: {position_int + 1}, 原残基: {wild_res}, 突变为: {mutation_res}")

            self.predict_model(position_int, wild_res, mutation_res, sequence)

        except Exception as e:
            print(f"处理错误: {e}")
            QMessageBox.critical(self, "错误", "发生错误，请检查输入和文件状态。")

    def predict_model(self, position_int, wild_res, mutation_res, sequence):
        self.worker = WorkerThread(position_int, wild_res, mutation_res, sequence, self.model, self.alphabet, self.device)
        self.worker.calculationDone.connect(self.onCalculationDone)  # 连接信号到槽，当计算完成后执行onCalculationDone函数
        self.statusLabel.setText('Calculating, please wait.......')  # 更改提示字样
        self.worker.start()    # 开始执行工作线程,会执行WorkerThread类的run方法

    def onCalculationDone(self, result):
        # self.ddgLable.setText(f"DDG Prediction: {result}")
        result = float(result)
        result = round(result, 4)
        if result > 0.5:
            self.statusLabel.setText(f'Calculation complete！\nDDG prediction is {result}\nThis is an stable mutation')
        elif result < 0.5:
            self.statusLabel.setText(f'Calculation complete！\nDDG prediction is {result}\nThis is an unstable mutation')
        else:
            self.statusLabel.setText(f'Calculation complete！\nDDG prediction is {result}\nThe mutation did not affect protein stability')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProteinViewer()
    sys.exit(app.exec_())
