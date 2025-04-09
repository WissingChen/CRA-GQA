import collections
import os
import numpy as np
from tabulate import tabulate
import torch
from torch import nn
from .base import BasePipeline
from utils.misc.misc import get_mask, load_file
import matplotlib.pyplot as plt
from random import shuffle
from scipy.special import softmax as np_softmax
import time
from utils.misc.misc import get_remain_time


class CRAPipeline(BasePipeline):
    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)
    
    def get_loss(self, batch):
        answer_id, answer, video_frames, question = (
                batch["answer_id"],
                batch["answer"],
                batch["video_frames"].cuda(),
                batch["question"].cuda(),
            )
        
        qsns_id, qsns_token_ids, qsns_seq_len = (
            batch['qsns_id'],
            batch['qsns_token_ids'],
            batch['qsns_seq_len']
        )
        
        video_len = batch["video_len"]

        pad_id = self.tokenizer.pad_token_id
        question_mask = (question!=pad_id).float().cuda() 
        answer_mask = (answer!=pad_id).float().cuda()

        video_mask = (
            get_mask(video_len, self.cfgs["dataset"]["max_feats"]).cuda() if self.cfgs["dataset"]["max_feats"] > 0 else None
        )

        BS = answer_id.size(0)
        pred = {}
        target = {}
        # find the video moments that are relevant to the question. 
        # qsns_mask = (qsns_token_ids != pad_id).float().cuda()
        if self.cfgs["model"]["vg_loss"]:
            video_len = batch["video_len"]

            video_mask = (
                get_mask(video_len, self.cfgs["dataset"]["max_feats"]).cuda() if self.cfgs["dataset"]["max_feats"] > 0 else None
            )

            vq_output = self.model(video_frames, video_mask, qsns_token_ids, qsns_id, mode="VQ")
            video_proj = vq_output[0].unsqueeze(2)
            vq_predicts = torch.bmm(vq_output[1], video_proj).squeeze()
            vq_predicts = vq_predicts.view(answer.size(0), self.cfgs["model"]["prop_num"], -1).mean(dim=1)
            # base
            pred["vq_pred"] = vq_predicts
            target["vq_target"] = qsns_id.cuda()
            # align
            pred["vq_align_pred"] = vq_output[2]["time_param"]["grounding_feature"]
            target["vq_align_gt"] = vq_output[2]["align_gt"]

        #refine the moments according to the answer
        gqa_output = self.model(video_frames, video_mask, answer, answer_id, mode="VQA")
        video_proj = gqa_output[0].unsqueeze(2)
        vqa_predicts = torch.bmm(gqa_output[1], video_proj).squeeze()
        predicts = vqa_predicts.view(answer.size(0), self.cfgs["model"]["prop_num"], -1).mean(dim=1)

        # vqa_loss = self.criterion(predicts, answer_id.cuda())
        pred["vqa_pred"] = predicts
        target["vqa_target"] = answer_id.cuda()

        pred["vqa_align_pred"] = gqa_output[2]["time_param"]["grounding_feature"]
        target["vqa_align_gt"] = gqa_output[2]["align_gt"]

        loss = self.criterion(pred, target)

        predicted = torch.max(predicts, dim=1).indices.cpu()
        running_acc = (predicted == answer_id).sum().item() / BS
        
        return loss, running_acc

    def _train_epoch(self, epoch):
        
        self.monitor.log_info(f"\n{'#'*20} Epoch {epoch}/{self.cfgs['optim']['epochs']} ... {'#'*20}\n")
        #########################################
        # Train Stage
        #########################################
        print("Training Stage")
        self.model.train()
        # reset monitor
        self.monitor.reset_kv(f"{self.cfgs['stat']['monitor']['metric']}")
        self.monitor.reset_kv(f"total_loss")
        for key in self.cfgs["optim"]["loss"].keys():
            if key == "ce" or key == "align":
                continue
            self.monitor.reset_kv(f"{key}_loss")
        # train iter

        for idx, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, monitor_metric = self.get_loss(batch)
            
            self.monitor.logkv_mean("acc", monitor_metric)
            self.optimizer.zero_grad()
            loss["total_loss"].backward()
            if self.cfgs["optim"]["clip"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfgs["optim"]["clip"])
            self.optimizer.step()
            self.lr_scheduler.step()

            progress = float(idx + 1) / len(self.train_dataloader)
            end_time = time.time()
            ct, rbt, ret = get_remain_time(start_time, end_time, idx, len(self.train_dataloader), epoch, self.cfgs["optim"]['epochs'])
            print_info = f"Progress: {progress:3.2%} (rbt-ret: {rbt}-{ret}) | Train {self.cfgs['stat']['monitor']['metric']}: {monitor_metric:3.2%}"

            for _, (loss_fns, _loss) in enumerate(loss.items()):
                running_loss = _loss.detach().cpu().numpy()
                self.monitor.logkv_mean(f"{loss_fns}", running_loss)
                print_info += f" {loss_fns}: {running_loss:.4f}"

            # print info
            print(f'\r{print_info}', end='', flush=True)
        
        print_info = [[], []]
        for _, (name, val) in enumerate(self.monitor.name2val.items()):
            print_info[0].append(f"mean {name}")
            print_info[1].append(f"{val:.4f}")

        self.monitor.log_info(f'\n{tabulate(print_info, headers="firstrow", tablefmt="grid")}', end="\n")

        #########################################
        # Eval Stage
        #########################################
        print("Eval Stage")
        # val set
        val_score, val_result = self.eval(self.val_dataloader)
        # test set
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
            test_result = val_result
        else:
            test_score, test_result = self.eval(self.test_dataloader)
        score = {}
        score.update({f"val_{k}": val_score[k] for k in val_score.keys()})
        score.update({f"test_{k}": test_score[k] for k in test_score.keys()})
        self.print_score(score)
        return score, test_result
    
    def print_score(self, score):
        def _print_score(_score):
            self.monitor.log_info(f"\n")
            for k in _score.keys():
                self.monitor.log_info(f"{k}  ")
            self.monitor.log_info(f"\n")
            for k in _score.keys():
                self.monitor.log_info(f"{_score[k]:.1f} \t ")

        self.monitor.log_info(f"\nVal Set\t mean acc: {score['val_acc']:.2%}")
        if 'val_pre-hoc' in score.keys():
            self.monitor.log_info("\n================ val_attn ================")
            _print_score(score['val_pre-hoc'])
        self.monitor.log_info("\n================ val_post-hoc ==================")
        _print_score(score['val_post-hoc'])
        if 'val_merge' in score.keys():
            self.monitor.log_info("\n================== val_merge ===================")
            _print_score(score['val_merge'])

        self.monitor.log_info(f"\nTest Set\t mean acc: {score['test_acc']:.2%}")
        if 'test_pre-hoc' in score.keys():
            self.monitor.log_info("\n============== test_attn ================")
            _print_score(score['test_pre-hoc'])
        self.monitor.log_info("\n================ test_post-hoc ================")
        _print_score(score['test_post-hoc'])
        if 'test_merge' in score.keys():
            self.monitor.log_info("\n================= test_merge ==================")
            _print_score(score['test_merge'])
        print("\n")
    
    def draw_grounding(self, gt, posthoc, attn, merge, epoch, probs=None):
        img_path = os.path.join(self.save_dir, "record", "grounding")
        if os.path.exists(img_path):
            # os.remove(img_path)
            pass
        else:
            os.mkdir(img_path)

        vid_qid = list(posthoc.keys())
        # if probs is not None:
        #    probs = {vid_qid[i]: probs[i] for i in range(len(vid_qid))}

        nums = 32  # len(vid_qid)
        shuffle(vid_qid)
        for i in range(nums):
            _vid_qid = vid_qid[i]
            _vid, _qid = _vid_qid.split('_')
            duration = gt[_vid]['duration']
            gt_g = gt[_vid]["location"][_qid]
            posthoc_g = posthoc[f'{_vid}_{_qid}']
            attn_g = np.round(np.asarray(attn[f'{_vid}_{_qid}'])*duration, 1)
            merge_g = merge[f'{_vid}_{_qid}']

            # 创建一个图和轴
            fig, ax = plt.subplots()
            
            # 对于每个时间区间，使用对应的颜色绘制一个条形图
            ax.plot(attn_g, [3, 3], color="blue", linewidth=6)
            ax.plot(posthoc_g, [2, 2], color="yellow", linewidth=6)
            ax.plot(merge_g, [1, 1], color="green", linewidth=6)
            ax.plot(gt_g[0], [0, 0], color="red", linewidth=6)  # 调整 linewidth 来改变时间条的粗细
            if probs is not None:
                _probs = probs[_vid_qid]  # [length]
                _probs = np.array(_probs)*3
                ax.plot(np.linspace(0, duration, 32), _probs, color="gray", linewidth=3, label="ATT-RLSTM")

            # 优化图表布局
            # plt.xticks(rotation=45)  # 旋转时间标签，使其更易读
            plt.tight_layout()       # 自动调整子图参数，使之填充整个图像区域

            fig.savefig(os.path.join(img_path, f'{epoch}_{_vid}_{_qid}.png'), dpi=300)
            del fig, ax
        
    def eval(self, data_loader, epoch=None, save=False):
        self.model.eval()

        count = 0

        score = {"acc": .0}
        results = {}
        ground_res = {}
        ground_pre = {}
        merge_probs = {}
        keyframe = {}
        ori_keyframe = {}
        with torch.no_grad():
            if not self.cfgs["dataset"]["mc"]:
                self.model.module._compute_answer_embedding(self.a2v)
            
            for idx, batch in enumerate(data_loader):
                answer_id, answer, video_frames, question, question_id = (
                    batch["answer_id"],
                    batch["answer"].to("cuda"),
                    batch["video_frames"].to("cuda"),
                    batch["question"].to("cuda"),
                    batch['question_id'],    
                )

                
                gsub = batch['gsub']

                video_len = batch["video_len"]
            
                if self.cfgs["model"]["lan"] in ['DistilBERET','BERT', 'DeBERTa']:
                    pad_id = 0
                elif self.cfgs["model"]["lan"] == 'RoBERTa':
                    pad_id = 1
                question_mask = (question!=pad_id).float() 
                answer_mask = (answer!=pad_id).float() 
                video_mask = get_mask(video_len, video_frames.size(1)).to("cuda")
                count += answer_id.size(0)
                bsize = answer_id.shape[0]
                
                #############Model FLOPs##########
                # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
                # flops = FlopCountAnalysis(model, inputs)
                # print('Model FLOPs:', flops.total()/1000000) #use batch_size 1
                # break
                ###################################
                """
                if isinstance(self.model, torch.nn.DataParallel):
                    fusion_proj, answer_proj, kargs = self.model.module(
                        video_frames,
                        video_mask,
                        answer,
                    )
                else:
                """
                # out = self.model.get_loss(batch, self.criterion)
                fusion_proj, answer_proj, kargs = self.model(
                    video_frames,
                    video_mask,
                    answer,
                    # gsub = gsub
                )
                # predicts = fusion_proj.squeeze()
                # fatt = fatt.squeeze().cpu().numpy()
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicted = torch.max(predicts, dim=1).indices.cpu()
                score["acc"] += (predicted == answer_id).sum().item()
                
                video_attn = kargs['fatt']

                video_attn = video_attn.squeeze().cpu().numpy()

                if 'time_param' in kargs.keys():
                    time_param = kargs['time_param']
                    start_time = time_param["start"].detach().cpu().numpy()
                    end_time = time_param["end"].detach().cpu().numpy()
                    _ground_pre = np.column_stack((start_time, end_time))
                    # _merge_probs = (time_param["key"].detach().cpu().numpy() + video_attn) / 2.
                    _keyframe = time_param["key"].detach().cpu().numpy()
                    _ori_keyframe = time_param["ori_key"].detach().cpu().numpy()
                    for bs, qid in enumerate(question_id):
                        results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
                        ground_res[qid] = video_attn[bs].tolist()
                        ground_pre[qid] = _ground_pre[bs].tolist()
                        # merge_probs[qid] = _merge_probs[bs].tolist()
                        keyframe[qid] = _keyframe[bs].tolist()
                        ori_keyframe[qid] = _ori_keyframe[bs].tolist()
                    merge_probs = None
                else:
                    merge_probs = None   
                    for bs, qid in enumerate(question_id):
                        results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
                        ground_res[qid] = video_attn[bs].tolist()
                

                # print info
                progress = float(idx + 1) / len(data_loader)
                print(f"\rProgress: {progress:.2%}", end='', flush=True)


            # ---------------------------------------
            # Eval and Save the result
            # ---------------------------------------
            score["acc"] /= count
            gt_file = os.path.join(self.cfgs["dataset"]["csv_path"], f'gsub_{data_loader.dataset.split}.json')
            gt_ground = load_file(gt_file)
            seg_file = os.path.join(self.cfgs["dataset"]["csv_path"], f'frame2time_{data_loader.dataset.split}.json')
            segs = load_file(seg_file)
            qa_file = os.path.join(self.cfgs["dataset"]["csv_path"], f'{data_loader.dataset.split}.csv')
            qas = load_file(qa_file)
            
            post_hoc_ground = self.metric.generate_ground(ground_res, segs, qas)
            
            # post-hoc
            score["post-hoc"] = self.metric(post_hoc_ground, gt_ground, results, gs=False)
            result = {"ph": post_hoc_ground, "result": results, "gt_ground": gt_ground, "ph_attn": ground_res}

            if 'time_param' in kargs.keys():
                # pre-hoc
                score["pre-hoc"] = self.metric(ground_pre, gt_ground, results, gs=True if 'time_param' in kargs.keys() else False)
            
                # merge post-hoc and pre-hoc
                if merge_probs is not None:
                    merge_ground = self.metric.generate_ground(merge_probs, segs, qas)
                else:
                    merge_ground = self.metric.combine(ground_pre, post_hoc_ground, gt_ground)
                score["merge"] = self.metric(merge_ground, gt_ground, results, gs=False)
                
                result["tg"] = ground_pre
                result["key"] = keyframe
                result["ori_key"] = ori_keyframe
                result["merge"] = merge_ground
            
            #if data_loader.dataset.split == "val":
            #    plot_thread = threading.Thread(target=self.draw_grounding, args=(gt_ground, post_hoc_ground, ground_pre, merge_ground, epoch, key_probs))
            #    plot_thread.start()
                # self.draw_grounding(gt_ground, post_hoc_ground, ground_pre, merge_ground, epoch)
        return score, result
    
    def infer(self, epoch=None):
        count_param = sum(p.numel() for p in self.model.parameters())
        print(f"Model Params: {count_param}")
        print("\nEval Stage")
        # val set
        val_score, _ = self.eval(self.val_dataloader, epoch, True if self.cfgs["dataset"]["name"] == "star" else False)
        # test set
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
        else:
            test_score, _ = self.eval(self.test_dataloader, epoch, True)
        score = {}
        score.update({f"val_{k}": val_score[k] for k in val_score.keys()})
        score.update({f"test_{k}": test_score[k] for k in test_score.keys()})
        self.print_score(score)
        return score