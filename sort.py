"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Fork from https://github.com/abewley/sort
    editor : weida wang (wade.wang96@outlook.com)
    date : 2020/10/18
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use(
    'Qt5Agg')  # use PyQt as backend to make the windows look like : https://s1.ax1x.com/2020/10/18/0XRKw8.png,
# official description : https://matplotlib.org/api/matplotlib_configuration_api.html#matplotlib.use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

try:
    from numba import jit  # 是python的一个JIT库，通过装饰器来实现运行时的加速
except:
    def jit(func):
        return func

np.random.seed(0)


#  指派问题（匈牙利算法）：https://www.cnblogs.com/ylHe/p/9287384.html，算法原理：https://blog.csdn.net/z464387937/article/details/51227347
def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


@jit
def iou(bb_test, bb_gt):
    """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    # IOU =（bb_test和bb_gt框相交部分面积）/ (bb_test框面积+bb_gt框面积 - 两者相交面积)
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):  # 如果检测框不带置信度
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))  # 返回[x1,y1,x2,y2]
    else:  # 如果检测框带置信度
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape(
            (1, 5))  # 返回[x1,y1,x2,y2,score]


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model #定义匀速模型
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 状态变量是7维，观测值是4维的，按照需要的维度构建目标
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities 对未观测到的初始速度给出高的不确定性
        self.kf.P *= 10.  # 默认定义的协方差矩阵是np.eye(dim_x)，将P中的数值与10， 1000相乘，赋值不确定性
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 将bbox转为 [x,y,s,r]^T形式，赋给状态变量X的前4位
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)  # 将 [ center x, center y, s, r ] 格式转化为 box [x1, y1, x2, y2]


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)  # 检测框与跟踪框IOU矩阵

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)  # 计算检测器与跟踪器的IOU并赋值给IOU矩阵对应位置

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(
                -iou_matrix)  # 加上负号是因为linear_assignment求的是最小代价组合，而我们需要的是IOU最大的组合方式，所以取负号
    else:
        matched_indices = np.empty(shape=(0, 2))  # 得到匹配项

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)  # 获得不匹配检测框
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)  # 获得不匹配跟踪器

    # filter out matched with low IOU, 以下代码用于过滤掉IOU低的匹配对
    matches = []  # 存放过滤后的匹配结果
    for m in matched_indices:  # 遍历粗匹配结果
        if iou_matrix[m[0], m[1]] < iou_threshold:  # m[0]是检测器ID， m[1]是跟踪器ID，如它们的IOU小于阈值则将它们视为未匹配成功
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))  # 将过滤后的匹配对维度变形成1x2形式
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)  # 如果过滤后匹配结果为空，那么返回空的匹配结果
    else:
        matches = np.concatenate(matches, axis=0)  # 如果过滤后匹配结果非空，则按0轴方向继续添加匹配对

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)  # 其中跟踪器数组是5列的（最后一列是ID）


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))  # 根据当前所有卡尔曼跟踪器的个数创建二维零矩阵，维度为：卡尔曼跟踪器ID个数x 5 (这5列内容为bbox与ID)
        to_del = []  # 存放待删除
        ret = []  # 存放最后返回的结果
        for t, trk in enumerate(trks):  # 循环遍历卡尔曼跟踪器列表
            pos = self.trackers[t].predict()[0]  # 用卡尔曼跟踪器 t 预测对应物体在当前帧中的bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)  # 如果预测的bbox为空，那么将第t个卡尔曼跟踪器删除
        trks = np.ma.compress_rows(
            np.ma.masked_invalid(trks))  # 将预测为空的卡尔曼跟踪器所在行删除，最后trks中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空bbox
        for t in reversed(to_del):  # 对to_del数组进行倒序遍历
            self.trackers.pop(t)  # 从跟踪器中删除 to_del中的上一帧跟踪器ID
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections，对于新增的未匹配的检测结果，创建并初始化跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])  # 将新增的未匹配的检测结果dets[i,:]传入KalmanBoxTracker
            self.trackers.append(trk)  # 将新创建和初始化的跟踪器trk 传入trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):  # 对新的卡尔曼跟踪器集进行倒序遍历
            d = trk.get_state()[0]  # 获取trk跟踪器的状态 [x1,y1,x2,y2]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to mot_benchmark.", type=str)  # the path of mot_benchmark
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str,
                        default='test')  # choose train phase or test phase
    parser.add_argument("--dataset", help="choose one of datasets of mot_benchmark", type=str,
                        default='KITTI-16')  # choose train phase or test phase
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    seq_path = args.seq_path
    phase = args.phase
    dataset = args.dataset
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display，Bbox颜色种类
    if (display):  # 如果要显示追结果到屏幕上,2D绘图初始化
        if not args.seq_path:
            print(
                '\n\tERROR: mot_benchmark path not found!\n\n    Please input the --seq_path argument \n    If you did not download MOT benchmark dataset, Please go to download it first : \n    https://motchallenge.net/data/2D_MOT_2015/#download.')
            exit()
        plt.ion()  # 用于动态绘制显示图像
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, dataset, 'det',
                           'det.txt')  # 如果dataset换成*，则会遍历所有数据集：ADL-Rundle，ETH，KITTI, PETS09, TUD, Venice

    # 因为本repo只为了演示跟踪效果，目标检测结果直接用现成的，即det.txt里的被检测物体的BBox信息。
    # 第0列代表帧数，第2-6列代表物体的BBox
    for seq_dets_fn in glob.glob(pattern):  # 循环依次处理多个数据集,这里我已经改成只处理一个数据集
        mot_tracker = Sort()  # create instance of the SORT tracker,创建SORT跟踪器实例
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  # #加载det.txt中的目标检测结果到矩阵seq_dets中
        seq = seq_dets_fn[pattern.find('*'):].split('/')[0]
        seq2 = seq_dets_fn[pattern.find('*'):].split('/')
        print('seq_dets_fn = {}'.format(seq_dets_fn))

        with open('output/%s.txt' % (dataset), 'w') as out_file:
            print("Processing %s." % (dataset))
            for frame in range(int(seq_dets[:, 0].max())):  # 确定视频序列总帧数，并进行for循环
                frame += 1  # detection and frame numbers begin at 1，由于视频序列帧数是从1开始的，因此加1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]  # 提取检测结果中的[x1,y1,w,h,score]到det
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]，将dets中的第2,3列的数加上第0,1列的数后赋值给2,3列；
                total_frames += 1

                if display:
                    fn = '%s/%s/%s/img1/%06d.jpg' % (seq_path, phase, dataset, frame)  # 原图像路径名
                    im = io.imread(fn)  # 加载图像
                    ax1.imshow(im)  # 显示图像
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()  # 获取当前的时间
                trackers = mot_tracker.update(dets)  # 将当前帧中所有检测物体的BBox送入SORT算法,返回对所有物体的跟踪结果BBox
                cycle_time = time.time() - start_time
                total_time += cycle_time  # SORT跟踪器总共耗费时间

                # 将SORT更新的所有跟踪结果逐个画到当前帧,井display到屏幕上
                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    # if display:
    #     print("Note: to get real runtime results run without the option: --display")
