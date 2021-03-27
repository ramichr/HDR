import os
from pathlib import Path

import cv2
import imageio
import io
import numpy as np
from canon_cr3 import Image as Image3
from PIL import Image


from PyQt5 import uic
from PyQt5.QtCore import Qt, QStandardPaths, QRectF, QUrl
from PyQt5.QtGui import QImage, QPainter, QTextDocument
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QInputDialog, QVBoxLayout, QPushButton, QStyle

from Script.SuperHDR_2 import sdr_series_to_hdr_array
from Script.SuperHDR_aux import sdrImage


def get_res(path, filename):
    ''' Get resources relative to the current file'''
    dirpath = os.path.dirname(os.path.realpath(__file__))
    fullpath = Path(os.path.join(dirpath, path + "/" + filename))
    if fullpath.exists():
        return fullpath.as_posix()
    else:
        return None


def ndarrayToQImage(img, fmt=QImage.Format_BGR888):
    height, width, _ = img.shape
    return QImage(img.data, width, height, img.strides[0], fmt)


def PIL2QImage(img, fmt=QImage.Format_RGB888):
    data = img.tobytes("raw", "RGB")
    return QImage(data, img.size[0], img.size[1], fmt)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.sdr_series = []  # sdrImage objects are stored here
        self.hdr_array = None  # output is stored here
        self.previews = []
        self.hdr_preview = None
        # Load UI, it can be also imported from HDRPlusForm.py
        uic.loadUi("HDRPlusForm.ui", self)
        self.setupUi()

    def changeColorIntensity(self, value):
        self.colLabel.setText(f"Color Intensity: {value}")
        # Handle color intensity slider change here

    def changeBrightness(self, value):
        self.brightLabel.setText(f"Brightness: {value}")
        # Handle brightness slider change here

    def changeContrast(self, value):
        self.contLabel.setText(f"Contrast: {value}")
        # Handle contrast slider change here

    def changeWhiteBalance(self, value):
        self.wbLabel.setText(f"White Balance: {value}")
        # Handle white balance slider change here

    def setupUi(self):
        with open(get_res("text", "p1.md")) as p1:
            self.text1.setMarkdown(p1.read())
        with open(get_res("text", "p2.md")) as p2:
            self.text2.setMarkdown(p2.read())
        # self.text1.loadResource(QTextDocument.MarkdownResource, QUrl())
        # self.text2.loadResource(QTextDocument.MarkdownResource, QUrl(get_res("text", "p2.md")))
        self.stackedWidget.setCurrentIndex(0)
        self.addItemImageStackWidget.setCurrentIndex(0)
        self.outputStack.setCurrentIndex(0)

        self.Execute.clicked.connect(self.execute)

        self.brightLabel.setText(f"Brightness: {self.brightSlider.value()}")
        self.brightSlider.valueChanged.connect(self.changeBrightness)

        self.colLabel.setText(f"Color Intensity: {self.colSlider.value()}")
        self.colSlider.valueChanged.connect(self.changeColorIntensity)

        self.contLabel.setText(f"Contrast: {self.contSlider.value()}")
        self.contSlider.valueChanged.connect(self.changeContrast)

        self.wbLabel.setText(f"White Balance: {self.wbSlider.value()}")
        self.wbSlider.valueChanged.connect(self.changeWhiteBalance)

        self.uploadBut.clicked.connect(self.upload)
        self.addItemImageBut.clicked.connect(self.upload)
        self.startBut.clicked.connect(self.start)
        self.downloadBut.clicked.connect(self.download)

    def upload(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            QStandardPaths.writableLocation(QStandardPaths.PicturesLocation),
            "Image files (*.*)",
        )
        imagePath = fname[0]
        if imagePath:
            exposure, ok = QInputDialog.getInt(self,
                                               "Select Exposure",
                                               "Exposure",
                                               0,
                                               flags=Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

            if ok:
                if imagePath.lower().endswith(".cr3"):
                    cimg = Image3(imagePath)
                    pimg = Image.open(io.BytesIO(cimg.jpeg_image)).convert('RGB')
                else:
                    pimg = Image.open(imagePath).convert('RGB')

                img = np.asarray(pimg, np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                preview = ImageViewer(PIL2QImage(pimg))
                i = len(self.previews)
                self.previewsGridLayout.addWidget(preview, i // 3, i % 3)
                self.previews.append(preview)
                self.addItemImageStackWidget.setCurrentIndex(1)
                self.sdr_series.append(
                    sdrImage(img, exposure)
                )

    def download(self):
        if self.hdr_array is None:
            return
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Download HDR Image")
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("HDR Image (*.exr)")
        if dialog.exec_():
            imagePath = dialog.selectedFiles()[0]
            imageio.imwrite(imagePath, self.hdr_array, "exr")

    def start(self):
        if self.hdr_array is None:
            self.hdr_array = sdr_series_to_hdr_array(self.sdr_series)

            img = np.power(self.hdr_array, 1 / 2.2)
            img = np.clip(np.floor(img * 255), 0, 255).astype(np.uint8)

            self.hdr_preview = ndarrayToQImage(img, fmt=QImage.Format_RGB888)
            self.outputLayout.addWidget(ImageViewer(self.hdr_preview))
            self.outputStack.setCurrentIndex(1)

    def execute(self):
        self.verticalLayout_2.setStretch(1, 6)
        self.stackedWidget.setCurrentIndex(1)


class ImageItem(QWidget):
    def __init__(self, qimage=None):
        super().__init__()
        layout = QVBoxLayout(self)
        self.im = ImageViewer(qimage)
        layout.addWidget(self.im)

        self.discardBut = QPushButton()
        self.discardBut.setIcon(self.style().standardIcon(
            QStyle.SP_DialogDiscardButton))
        layout.addWidget(self.discardBut)


class ImageViewer(QWidget):
    def __init__(self, qimage=None):
        super().__init__()
        if qimage:
            self.image = qimage

    def setImage(self, qimage):
        self.image = qimage

    def paintEvent(self, e):
        qp = QPainter(self)
        qp.setRenderHints(QPainter.SmoothPixmapTransform |
                          QPainter.Antialiasing)
        image = self.image
        w = qp.device().width()
        h = qp.device().height()

        if not image.isNull():
            iw = image.width()
            ih = image.height()
            if ih > 0 and iw > 0:
                f = min(w / iw, h / ih)

                nw = iw * f
                nh = ih * f

                qp.drawImage(QRectF((w - nw) / 2, (h - nh) / 2, nw, nh), image)


if __name__ == "__main__":
    import sys

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([])

    # Make taskbar icon work on Windows
    import platform

    if platform.system() == "Windows":
        import ctypes

        appid = "hdrplus.hdrplus"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
