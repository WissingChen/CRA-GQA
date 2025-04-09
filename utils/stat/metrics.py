
from typing import Any
import numpy as np


class Metric(object):
    def __init__(self, cfgs) -> None:
        pass

    def __call__(self, pred_ground, target_ground, pred_qa=None, subset=None, gs=False):
        mIoU, mIoP = 0, 0
        cnt, cqt, cqtplus = 0, 0, 0
        crt3, crt5 = 0, 0
        crtp3, crtp5 = 0, 0
        cqt_5, cqt_3 = 0, 0
        ufcqt_5, ufcqt_3 = 0, 0
        for vid, anno in target_ground.items():
            for qid, locs in anno['location'].items():
                if not (f'{vid}_{qid}' in pred_ground):
                    # print(vid, qid)
                    continue
                if subset != None:
                    # Non-Blind and Non-Sig QA subset
                    if not (f'{vid}_{qid}' in subset):
                        continue
                max_tIoU, max_tIoP = 0, 0
                kid = f'{vid}_{qid}'
                for loc in locs:
                    span = pred_ground[f'{vid}_{qid}']
                    # we need to multiply video duration if Gaussian
                    if gs: span = np.round(np.asarray(span)*anno['duration'], 1)
                    tIoU, tIoP = self.get_tIoU(loc, span)
                    if tIoU > max_tIoU:
                        max_tIoU = tIoU
                    if tIoP > max_tIoP:
                        max_tIoP = tIoP
                if max_tIoP >= 0.3:
                    crtp3 += 1
                    if  max_tIoP >= 0.5:
                        crtp5 += 1
                        
                        if pred_qa:
                            if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                                cqt+= 1
                                # print(kid)

                if max_tIoU >= 0.3:
                    crt3 += 1
                    if max_tIoU >= 0.5:
                        crt5 += 1

                if max_tIoP < 0.5:
                    if pred_qa:
                        if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                            ufcqt_5+=1
                        else:
                            cqt_5+= 1
                            # print(kid)
                    if max_tIoP < 0.3:
                        if pred_qa:
                            if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                                ufcqt_3+=1
                            else:
                                cqt_3+= 1
                if pred_qa:
                    kid = f'{vid}_{qid}'
                    if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                        cqtplus+= 1
                        # print(kid)

                cnt += 1
                mIoU += max_tIoU
                mIoP += max_tIoP
        
        mIoU = mIoU /cnt * 100
        mIoP = mIoP/cnt * 100
        score = {
            "Acc&GQA": cqt*1.0/cnt*100,
            "Acc&VQA": cqtplus*1./cnt*100,
            # "FR": (cqt*1.0/cnt*100)/(cqtplus*1./cnt*100)*100,
            # "Err&0.3": cqt_3*1.0/cnt*100,
            # "Err&0.5": cqt_5*1.0/cnt*100,
            # "UF&0.3": ufcqt_3*1.0/cnt*100,
            # "UF&0.5": ufcqt_5*1.0/cnt*100,
            "mIoP": mIoP,
            "TIoP@0.3": crtp3*1.0/cnt*100,
            "TIoP@0.5": crtp5*1.0/cnt*100,
            "mIoU": mIoU,
            "TIoU@0.3": crt3*1.0/cnt*100,
            "TIoU@0.5": crt5*1.0/cnt*100
        }
        return score
    
    @staticmethod
    def get_tIoU(loc, span):

        if span[0] == span[-1]:
            if loc[0] <= span[0] and span[0] <= loc[1]:
                return 0, 1
            else:
                return 0, 0
        
        span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
        span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
        dis_i = (span_i[1] - span_i[0])
        if span_u[1] > span_u[0]:
            IoU = dis_i / (span_u[1] - span_u[0]) 
        else: 
            IoU = 0.0
        if span[-1] > span[0]:
            IoP = dis_i / (span[-1] - span[0]) 
        else:
            IoP = 0.0

        return IoU, IoP

    @staticmethod
    def combine(pred1, pred2, gt, way="itsc"):
        """
        pred1: ground segment by gaussian mask
        pred2: ground segment by post-hoc attention
        gt: to get NExT-GQA subset
        """
        def _cb_seg(seg1, seg2, way='itsc'):
            # print(seg1, seg2)
            if way == 'uni':
                # 选择并集
                ts = [seg1[0], seg1[1], seg2[0], seg2[1]]
                ts = sorted(ts)
                new_seg = [ts[0], ts[-1]]
            elif way == 'itsc':
                # 选择交集
                start = seg1[0] if seg1[0] > seg2[0] else seg2[0]
                end = seg1[1] if seg1[1] < seg2[1] else seg2[1]
                if not (start <= end):
                    new_seg = seg2.tolist() #trust more on attention
                else:
                    new_seg = [start, end]
            elif way == 'itsc_refine': # 差
                # 选择交集
                start = seg1[0] if seg1[0] > seg2[0] else seg2[0]
                end = seg1[1] if seg1[1] < seg2[1] else seg2[1]
                if not (start <= end):
                    new_seg = seg1.tolist() #trust more on attention
                else:
                    new_seg = [start, end]
            elif way == 'itsc_swap': # 更差
                # 选择交集
                start = seg1[0] if seg1[0] > seg2[0] else seg2[0]
                end = seg1[1] if seg1[1] < seg2[1] else seg2[1]
                if not (start <= end):
                    new_seg = [end, start] #swap the grounding
                else:
                    new_seg = [start, end]
            elif way == 'itsc_mean': # 差
                # 选择交集
                start = seg1[0] if seg1[0] > seg2[0] else seg2[0]
                end = seg1[1] if seg1[1] < seg2[1] else seg2[1]
                if not (start <= end):
                    new_seg = [(seg1[0]+seg2[0])//2, (seg1[1]+seg2[1])//2]
                else:
                    new_seg = [start, end]
            return new_seg
        
        cb_ground = {}
        for vqid, seg in pred1.items():
            if len(vqid.split('_'))>2:
                vid = vqid.split('_')[0]
                qid = '_'.join(vqid.split('_')[1:])
            else:
                vid, qid = vqid.split('_')
            if not (vid in gt and qid in gt[vid]['location']):
                continue 
            duration = gt[vid]['duration']
            seg = np.round(np.asarray(seg)*duration, 1)
            seg_att = np.asarray(pred2[vqid])
            new_seg  = _cb_seg(seg, seg_att, way=way)
            cb_ground[vqid] = new_seg
        
        # save_to()
        return cb_ground

    def generate_ground(self, preds, segs, qas, thd_weight=0.3, dis=10):
        # print(len(preds))
        res_ground = {}
        for idx, row in qas.iterrows():
            vid, qid = str(row['video_id']), str(row['qid'])
            vid_qid = '_'.join([vid, qid])
            atts = preds[vid_qid]
            # fids = np.linspace(0, 31, 24, dtype='int')
            cur = np.asarray(segs[vid])#[fids] #[[16]]

            # for VGT hierarchical attention
            # vatts = []
            # for id, fatt in enumerate(atts['fatt']):
            #     catt_v = atts['catt'][id]
            #     for v in fatt:
            #         hint = np.round(v+catt_v, 2)
            #         vatts.append(hint)
            

            seg = self.find_seg_ada(atts, cur, thd_weight, dis)
            # seg = self.find_seg_win(atts)
            # print(vid_qid, seg)
            
            res_ground[vid_qid] = seg

        return res_ground
    
    @staticmethod
    def find_seg_win(vatt):
        # 定义窗口大小
        window_sizes = [3, 5, 7]
        preds = np.array(vatt.copy()).reshape([1, -1])
        _, length = preds.shape
        # 初始化结果变量
        max_probs = np.zeros(1)
        max_indices = np.zeros([1, 2])
        overall_max = preds.max(axis=1)
        # 遍历所有窗口大小
        for window_size in window_sizes:
            for i in range(length - window_size + 1):
                window_probs = preds[:, i:i + window_size].sum(axis=1)
                
                # 计算每个窗口内最大值的位置
                window_max = preds[:, i:i + window_size].max(axis=1)
                
                # 获取整体向量的最大值
                overall_max = preds.max(axis=1)
                
                # 确保窗口包含最大值
                valid_mask = (window_max == overall_max.squeeze())
                max_mask = (window_probs > max_probs) & valid_mask
                
                max_probs[max_mask] = window_probs[max_mask]
                max_indices[max_mask, 0] = i
                max_indices[max_mask, 1] = i + window_size - 1

        start = (max_indices[:, 0] / 31.)[0]
        end = (max_indices[:, 1] / 31.)[0]
        return [start, end]
    
    @staticmethod
    def find_seg_ada(vatt, cur, thd_weight=0.3, dis=10):
        #jointly consider the attentioin score and its distance with the maximal point
        if not isinstance(vatt, list): 
            vatt = [vatt]
        n = len(vatt)
        vatt = np.asarray(vatt)
        vatt = (np.asarray(vatt)-np.min(vatt))/(np.max(vatt)-np.min(vatt)+1e-12)
        max_id = np.argmax(vatt)
        mean_s = np.mean(vatt)
    
        path = [max_id]
        # cid, fid = max_id//4, max_id%4
        # stamp_s = cur[cid][fid]
        stamp_s = cur[max_id]
        i = 1
        thd = mean_s + thd_weight*mean_s #(0.1~0.5)
        # print(mean_s)
        # dis = 10
        while max_id - i > 0:
            # cid, fid = (max_id-i)//4, (max_id-i)%4
            # stamp = cur[cid][fid]
            stamp = cur[max_id-i]
            elapse = abs(stamp_s - stamp)
            score = vatt[max_id-i] / (1+elapse/dis)
            # print(elapse, score, vatt[max_id-i])
            if score >= thd:
                path.append(max_id-i)
            elif elapse > dis:
                break
            i += 1
        i = 1
        while max_id + i < n:
            # cid, fid = (max_id+i)//4, (max_id+i)%4
            # stamp = cur[cid][fid]
            stamp = cur[max_id + i]
            elapse = abs(stamp_s - stamp)
            score = vatt[max_id+i] / (1+elapse/dis)
            # print(elapse, score, vatt[max_id+i])
            if score >= thd:
                path.append(max_id+i)
            elif elapse > dis:
                break
            i += 1
            
        sid, eid = np.min(path), np.max(path)
        start, end = cur[sid], cur[eid]
        # start = cur[sid//4][sid%4]
        # end = cur[eid//4][eid%4]
        # min distance is 1.5
        # NOTE 提升IoU性能的小技巧
        # if end - start < 1.8:
            # start -= 0.9
            # end += 0.9
        return [start, end]


