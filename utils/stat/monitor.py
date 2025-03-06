"""
a monitor to record all info
including metric, loss, result
draw the loss and monitor metric curves
"""
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from collections import defaultdict
from PIL import Image

import torch
from utils.stat import html


class Monitor(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.record_dir = osp.join(cfgs["stat"]["record_dir"], cfgs["misc"]["running_name"], "record")
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

        self.log_dir = osp.join(self.record_dir, 'info.log')
        if cfgs["stat"]["resume"] != None:
            file_mode = "a"
        else:
            file_mode = "w"
        with open(self.log_dir, file_mode) as f:
            record_time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
            temp = "#" * 50
            f.write(f"{temp}\nRecording Time {record_time}\n{temp}\n")

        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)

        if cfgs["stat"]["monitor"]["vis"]:
            self.display_id = 1
            self.use_html = True
            self.win_size = 160
            self.name = cfgs['misc']['running_name']
            self.cfgs = cfgs
            self.saved = False
            if self.display_id > 0:
                import visdom
                self.vis = visdom.Visdom(port=cfgs["stat"]["monitor"]['display_port'])

            if self.use_html:
                self.web_dir = osp.join(self.record_dir, 'web')
                self.img_dir = os.path.join(self.web_dir, 'images')
                self.csv_dir = os.path.join(self.web_dir, 'csv')
                if osp.exists(self.web_dir) is False:
                    os.mkdir(self.web_dir)
                    os.mkdir(self.img_dir)
                    os.mkdir(self.csv_dir)
                print('create web directory %s for visualization...' % self.web_dir)

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1
    
    def reset_kv(self, key):
        self.name2val[key] = .0
        self.name2cnt[key] = 0

    def dumpkv2log(self, epoch):
        d = self.name2val
        d["epoch"] = epoch
        out = d.copy()  # Return the dict for unit testing purposes
        arr = dict2str(out)
        with open(self.log_dir, "a") as f:
            f.write(arr)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def logkv(self, key, val):
        self.name2val[key] = val

    def log_info(self, arr, end=''):
        with open(self.log_dir, "a") as f:
            f.write(arr)
        print(arr, end=end)

    def reset_visdom(self):
        self.saved = False
    

    # TODO load old data and merge new data, then vis metric and images and save it.
    # |visuals|: dictionary of images to display or save\
    def display_current_images(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = 0  # self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show report in the browser
            idx = 1
            for label, data in visuals.items():
                if isinstance(data, str):
                    self.vis.text(data, opts=dict(title=label), win=self.display_id + idx)
                elif isinstance(data, torch.Tensor):
                    data = ((data - data.min()) / (data.max() - data.min())) * 255
                    if len(data.size()) == 3:
                        data = data.permute([1, 2, 0])
                        data = data.detach().cpu().numpy()
                        self.display_current_images({label: data}, epoch, save_result)
                    else:
                        data = data.permute([0, 2, 3, 1])
                        data = data.detach().cpu().numpy()
                        data = {f'{label}_{i}': data[i] for i in range(data.shape[0])}
                        self.display_current_images(data, epoch, save_result)
                else:
                    raise ValueError
                idx += 1

        """
        if self.use_html:  # save report to a html file
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, report in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()
        """

    # errors: dictionary of error labels and values
    def plot_current_metrics(self, epoch, metrics, counter_ratio=None):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        if counter_ratio is None:
            self.plot_data['X'].append(epoch)
        else:
            self.plot_data['X'].append(epoch+counter_ratio)
        self.plot_data['Y'].append([metrics[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' metrics over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_metrics(self, epoch, metrics, t, mode):
        message = '(%s - epoch: %d | time: %.3f) ' % (mode, epoch, t)
        for k, v in metrics.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        # save image to the disk

    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        # short_path = ntpath.basename(image_path[0])
        # name = os.path.splitext(short_path)[0]

        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            save_image(image_numpy, save_path)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_data_plt(self, webpage, visuals, pred_gt, pred, image_path):
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            img = image_numpy[0].cpu().float().numpy()
            fig = plt.imshow(img[0, ...])
            fig.set_cmap('gray')
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        image_name = '%s_%s.png' % (name, 'pred_gt')
        save_path = os.path.join(image_dir, image_name)
        img = pred_gt.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        ims.append(image_name)
        txts.append('pred_gt')
        links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_result_fig(self, img, imgName, webpage, image_path):
        image_dir = webpage.get_image_dir()
        short_path = image_path.split('/')
        name = short_path[-1]
        image_name = '%s_%s.png' % (name, imgName)
        save_path = os.path.join(image_dir, image_name)
        img = img.astype(float)
        fig = plt.imshow(img)
        fig.set_cmap('gray')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def save_image(image_numpy, image_path):
    image_numpy = np.uint8(image_numpy)
    image_pil = Image.fromarray(image_numpy)
    # image_pil = scipy.misc.toimage(image_numpy)
    image_pil.save(image_path)
