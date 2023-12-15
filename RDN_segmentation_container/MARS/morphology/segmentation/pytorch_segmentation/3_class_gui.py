'''
Gui wrapper for the 3 class segmentation

Author: Sun, Yung-Chen yzs5463@psu.edu
Author: Yazdani, Amirsaeed auy200@psu.edu
Author: Nick Stephens nbs49@psu.edu
'''

import os
import sys
import wx
import torch
import pathlib
import numpy as np
from PIL import Image
from net.unet_light_rdn import UNet_Light_RDN

#Path where the MARS icon lives. the sys.argv[0] returns the folder where this script is located.
setup_path = pathlib.Path(os.path.abspath(os.path.dirname(sys.argv[0]))).parent.parent.parent.parent.joinpath("Setup")
icon_path = setup_path.joinpath('mars_icon.bmp')

#The default model path is contained within MARS.
default_model = pathlib.Path(os.path.abspath(os.path.dirname(sys.argv[0]))).joinpath("model").joinpath("unet_light_rdn.pth")
print(default_model)

color_dict = [[0.0], [128.0], [255.0]]
model_path = None
data_folder = None
save_folder = None
file_types = ["tif", "png", "jpg", "bmp"]
type_dict = {"tif": "TIFF", "png": "PNG", "jpg": "JPEG", "bmp": "BMP"}


class segment_widget(wx.Frame):
    def __init__(self, parent, title):
        super(segment_widget, self).__init__(parent, title=title, size=(1500, 400))

        #Set up the asthectics for the panel
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour("Dark Grey")
        self.panel.SetForegroundColour("White")

        label_font = wx.Font(pointSize=13, family=wx.SCRIPT, style=wx.NORMAL, weight=wx.NORMAL)

        box = wx.BoxSizer(wx.VERTICAL)
        self.label = wx.StaticText(self.panel, label="Please select segmentation options...", pos=(25, 20))
        self.label.SetFont(label_font)
        box.Add(self.label, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)

        box = wx.BoxSizer(wx.VERTICAL)
        self.file_t = wx.StaticText(self.panel, label="Output file type", pos=(73, 65))
        self.file_t.SetFont(label_font)
        box.Add(self.file_t)
        label_font = wx.Font(pointSize=12, family=wx.SCRIPT, style=wx.NORMAL, weight=wx.NORMAL)
        my_btn = wx.Button(self.panel, label='Choose model ', size=(150, 40), pos=(50, 100))
        my_btn.SetFont(label_font)
        my_btn.SetForegroundColour("White")
        my_btn.SetBackgroundColour("Dark Slate Grey")
        my_btn.Bind(wx.EVT_BUTTON, self.chooses_model_path)

        my_btn = wx.Button(self.panel, label='Data folder', size=(150, 40), pos=(50, 150))
        my_btn.SetFont(label_font)
        my_btn.SetForegroundColour("White")
        my_btn.SetBackgroundColour("Dark Grey")
        my_btn.Bind(wx.EVT_BUTTON, self.chooses_data_folder)

        my_btn = wx.Button(self.panel, label='Output folder', size=(150, 40), pos=(50, 200))
        my_btn.SetFont(label_font)
        my_btn.SetForegroundColour("White")
        my_btn.SetBackgroundColour("Dark Grey")
        my_btn.Bind(wx.EVT_BUTTON, self.chooses_save_folder)

        my_btn = wx.Button(self.panel, label='Start!', size=(150, 40), pos=(50, 250))
        my_btn.SetFont(label_font)
        my_btn.SetForegroundColour("White")
        my_btn.SetBackgroundColour("Dark Grey")
        my_btn.Bind(wx.EVT_BUTTON, self.start)

        self.Show()

        font = wx.Font(13, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)

        self.combo = wx.ComboBox(self.panel, choices=file_types, size=(80, 40), pos=(250, 60))
        self.combo.SetFont(font)
        self.combo.SetForegroundColour("White")
        self.combo.SetBackgroundColour("Dark Grey")
        self.combo.Bind(wx.EVT_COMBOBOX, self.chooses_file_type)

        self.path1 = wx.StaticText(self.panel, -1, 'Use default', (250, 110))
        self.path1.SetFont(font)
        self.path2 = wx.StaticText(self.panel, -1, 'None', (250, 160))
        self.path2.SetFont(font)
        self.path3 = wx.StaticText(self.panel, -1, 'None', (250, 210))
        self.path3.SetFont(font)
        self.progress = wx.StaticText(self.panel, -1, 'None', (250, 260))
        self.progress.SetFont(font)

        wx.StaticBitmap(self.panel, -1, bitmap=wx.Bitmap(str(icon_path)), pos=(1425, 17))

    def chooses_file_type(self, event):
        self.file_type = self.combo.GetValue()
        self.label.SetLabel(f"Segmented images will be written out as {self.file_type} files...")
        print(self.file_type)

    def chooses_model_path(self, event):
        global model_path
        dialog = wx.FileDialog(None, "Choose path", style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
        try:
            self.path1.SetLabel(path)
            model_path = path
        except UnboundLocalError:
            pass

    def chooses_data_folder(self, event):
        global data_folder
        dialog = wx.DirDialog(None, "Choose path", style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
        try:
            self.path2.SetLabel(path)
            data_folder = path
        except UnboundLocalError:
            pass

    def chooses_save_folder(self, event):
        global save_folder
        dialog = wx.DirDialog(None, "Choose path", style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
        try:
            self.path3.SetLabel(path)
            save_folder = path
        except UnboundLocalError:
            pass

    def start(self, event):
        net = UNet_Light_RDN(n_channels=1, n_classes=3)
        net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        net.cuda()
        net.eval()

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        image_names = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

        self.progress.SetLabel('Start processing')
        seg_count = len(image_names)
        seg_now = seg_count
        for i in range(len(image_names)):

            image_name = image_names[i]
            image = Image.open(os.path.join(data_folder, image_name)).convert('L')
            image = np.array(image)

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            image = image.unsqueeze(0).float() / 255.0
            image = image.cuda()
            with torch.no_grad():
                pred = net(image)
            pred = pred.argmax(1)
            pred = pred.cpu().squeeze().data.numpy()

            pred_img = np.zeros(pred.shape)
            for i in range(len(color_dict)):
                for j in range(len(color_dict[i])):
                    pred_img[pred == i] = color_dict[i][0]

            pred_img = pred_img.astype(np.uint8)

            f_type = type_dict[str(self.file_type)]
            pred_img = Image.fromarray(pred_img, 'L')
            image_name = f"{image_name[:-3]}.{str(self.file_type)}"
            image_name = image_name.replace("..", ".")
            pred_img.save(os.path.join(save_folder, image_name), str(f_type))

            seg_now -= 1
            per_complete = abs((1 - (seg_now / seg_count)) * 100)
            self.progress.SetLabel(f'{seg_now} of {seg_count} remaining, {per_complete:3.2f}% complete...')

        self.progress.SetLabel('Segmentations are done!')

if __name__ == '__main__':
    app = wx.App()
    segment_widget(None, title='3 class trabecular segmentation')
    app.MainLoop()
