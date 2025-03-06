import collections
import os
import numpy as np
import torch
from torch import nn
from .base import BasePipeline
from utils.misc.misc import get_mask, load_file
from tabulate import tabulate
import time
from utils.misc import get_remain_time

class TempCLIPPipeline(BasePipeline):
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
        assert self.cfgs["model"]["baseline"] in ['NG+', 'posthoc', 'NG']
        # find the video moments that are relevant to the question. 
        # qsns_mask = (qsns_token_ids != pad_id).float().cuda()
        if self.cfgs["model"]["vg_loss"]:
            vt_proj, txt_proj, args_vg = self.model(
                video_frames,
                video_mask,
                # question,
                # question_mask,
                answer=qsns_token_ids,
                answer_id = qsns_id,
                stage='GD'
            )
            vt_proj = vt_proj.unsqueeze(2)
            vq_predicts = torch.bmm(txt_proj, vt_proj).squeeze() # [B, 5, 1]
            vq_predicts = vq_predicts.view(BS, self.cfgs["model"]["prop_num"], -1).mean(dim=1)
            # vg_loss = self.criterion(vq_predicts, qsns_id.cuda())
            pred["vq_pred"] = vq_predicts
            target["vq_target"] = qsns_id.cuda()

        #refine the moments according to the answer
        fusion_proj, answer_proj, args_vg = self.model(
            video_frames,
            video_mask,
            # question,
            # question_mask,
            answer=answer.cuda(),
            gauss_weight = None, #args_vg['gweight'] if args.vg_loss else None,
            stage = 'GQA'
        )
        
        fusion_proj = fusion_proj.unsqueeze(2)
        predicts = torch.bmm(answer_proj, fusion_proj).squeeze() # [B, 5, 1]
        predicts = predicts.view(BS, self.cfgs["model"]["prop_num"], -1).mean(dim=1)

        # vqa_loss = self.criterion(predicts, answer_id.cuda())
        pred["vqa_pred"] = predicts
        target["vqa_target"] = answer_id.cuda()

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
            if key == "ce":
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
            print_info = f"Progress: {progress:3.2%} (ret: {ret}) | Train {self.cfgs['stat']['monitor']['metric']}: {monitor_metric:3.2%}"

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
    
    def save_result(self):
        pass
    
    def print_score(self, score):
        def _print_score(_score):
            self.monitor.log_info(f"\n")
            for k in _score.keys():
                self.monitor.log_info(f"{k}  ")
            self.monitor.log_info(f"\n")
            for k in _score.keys():
                self.monitor.log_info(f"{_score[k]:.1f} \t ")

        self.monitor.log_info(f"\nVal Set\t mean acc: {score['val_acc']:.2%}")
        if self.cfgs["model"]["baseline"] in ['NG+', 'NG']:
            self.monitor.log_info("\n================ val_gauss_mask ================")
            _print_score(score['val_gauss_mask'])
        self.monitor.log_info("\n================ val_post-hoc ==================")
        _print_score(score['val_post-hoc'])
        if self.cfgs["model"]["baseline"] in ['NG+', 'NG']:
            self.monitor.log_info("\n================== val_merge ===================")
            _print_score(score['val_merge'])

        self.monitor.log_info(f"\nTest Set\t mean acc: {score['test_acc']:.2%}")
        if self.cfgs["model"]["baseline"] in ['NG+', 'NG']:
            self.monitor.log_info("\n============== test_gauss_mask ================")
            _print_score(score['test_gauss_mask'])
        self.monitor.log_info("\n================ test_post-hoc ================")
        _print_score(score['test_post-hoc'])
        if self.cfgs["model"]["baseline"] in ['NG+', 'NG']:
            self.monitor.log_info("\n================= test_merge ==================")
            _print_score(score['test_merge'])
        print("\n")
        
    def eval(self, data_loader):
        self.model.eval()

        count = 0

        score = {"acc": .0}
        results = {}
        ground_res = {}
        ground_gs = {}
        ground_mask = {}
        gs_att = {}
        with torch.no_grad():
            if not self.cfgs["dataset"]["mc"]:
                self.model.module._compute_answer_embedding(self.a2v)
            
            for idx, batch in enumerate(data_loader):
                answer_id, answer, video_frames, question, question_id = (
                    batch["answer_id"],
                    batch["answer"].cuda(),
                    batch["video_frames"].cuda(),
                    batch["question"].cuda(),
                    batch['question_id'],    
                )
            
                video_len = batch["video_len"]
            
                if self.cfgs["model"]["lan"] in ['DistilBERET','BERT', 'DeBERTa']:
                    pad_id = 0
                elif self.cfgs["model"]["lan"] == 'RoBERTa':
                    pad_id = 1
                question_mask = (question!=pad_id).float() 
                answer_mask = (answer!=pad_id).float() 
                video_mask = get_mask(video_len, video_frames.size(1)).cuda()
                count += answer_id.size(0)
                bsize = answer_id.shape[0]
                
                #############Model FLOPs##########
                # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
                # flops = FlopCountAnalysis(model, inputs)
                # print('Model FLOPs:', flops.total()/1000000) #use batch_size 1
                # break
                ###################################
                fusion_proj, answer_proj, kargs = self.model(
                    video_frames,
                    video_mask,
                    question,
                    question_mask,
                    answer,
                    stage='GQA'
                )
                # predicts = fusion_proj.squeeze()
                # fatt = fatt.squeeze().cpu().numpy()
                
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

                # predicts = predicts.view(bsize, 8, -1).max(dim=1)[0] #slightly wrose than mean
                if self.cfgs["model"]["baseline"] in ['NG+', 'NG']:
                    prediction = predicts.view(bsize, self.cfgs["model"]["prop_num"], self.cfgs["dataset"]["mc"])
                    
                    # prop_scores = torch.from_numpy(np.zeros((bsize, args.prop_num))).cuda()
                    # for bs, aid in enumerate(answer_id):
                    #     prop_scores[bs] = prediction[bs, :, aid] #.cpu().numpy()

                    index = torch.argmax(predicts, dim=-1)
                    prop_scores = torch.from_numpy(np.zeros(index.shape)).cuda()
                    for bp_id, max_op in enumerate(index):
                        prop_scores[bp_id] = predicts[bp_id,max_op] # predicts[torch.argmax(predicts, dim=-1)].view(bsize, -1)
                    
                    predicts = prediction.mean(dim=1)
                    prop_scores = prop_scores.view(bsize, self.cfgs["model"]["prop_num"])
                    
                    prop_idx = (-prop_scores).argsort(dim=-1)

                    # print(idx.shape)
                    att = kargs['fatt'].view(bsize, self.cfgs["model"]["prop_num"], -1)
                    att = att.gather(1, prop_idx.unsqueeze(-1).expand(-1, -1, att.size(-1)))
                    gc = kargs['gcenter'].view(bsize, self.cfgs["model"]["prop_num"]).gather(index=prop_idx, dim=-1)
                    gw = kargs['gwidth'].view(bsize, self.cfgs["model"]["prop_num"]).gather(index=prop_idx, dim=-1)
                    gmask = kargs['gweight'].view(bsize, self.cfgs["model"]["prop_num"], self.cfgs["dataset"]["max_feats"]).cpu().numpy() #.gather(index=idx, dim=1)
                    gamma = self.cfgs["model"]["gamma"]
                    props = torch.stack([torch.clamp(gc-gamma*gw/self.cfgs["model"]["sigma"], min=0), torch.clamp(gc+gamma*gw/self.cfgs["model"]["sigma"], max=1)], dim=-1)
                    
                    props = props.cpu().numpy()
                    
                    if self.cfgs["model"]["vote"]:
                        if self.cfgs["model"]["vote"] == 1:
                            # vote for the proposal with maximal overlap with others
                            c = np.ones((bsize, self.cfgs["model"]["prop_num"]))
                            votes = np.zeros((bsize, self.cfgs["model"]["prop_num"]))
                            for i in range(self.cfgs["model"]["prop_num"]):
                                for j in range(self.cfgs["model"]["prop_num"]):
                                    iou = calculate_IoU_batch((props[:, i, 0], props[:, i, 1]), (props[:, j, 0], props[:, j, 1]))
                                    iou = iou * c[:, j]
                                    votes[:, i] = votes[:, i] + iou
                            prop_idx = np.argmax(votes, axis=1)
                            prop = props[np.arange(bsize), prop_idx]
                            att = att[torch.arange(bsize), prop_idx, :]
                        elif self.cfgs["model"]["vote"] == 2:
                            assert self.cfgs["mdoel"]["vote"] == 2, 'not implemented yet'
                            #vote for the intersection of multiple proposals
                    else:
                        #directly choose the temporal proposal with highest confidence
                        prop = props[:, 0].squeeze()
                        att = att[:,0,:]

                predicted = torch.max(predicts, dim=1).indices.cpu()
                score["acc"] += (predicted == answer_id).sum().item()
                
                if self.cfgs["model"]["baseline"] == 'posthoc':
                    att = kargs['fatt']

                att = att.squeeze().cpu().numpy()
                
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
                    if self.cfgs["model"]["baseline"] == 'posthoc':
                        ground_res[qid] = att[bs].tolist()
                    else:
                        ground_gs[qid] = prop[bs].tolist()
                        ground_mask[qid] = gmask[bs]
                        ground_res[qid] = att[bs].tolist()
            
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
            # gauss mask
            if self.cfgs["model"]["baseline"] in ['NG+', 'NG']:
                score["gauss_mask"] = self.metric(ground_gs, gt_ground, results, subset=None, gs=True)
                # merge post-hoc and gauss mask
                merge_ground = self.metric.combine(ground_gs, post_hoc_ground, gt_ground)
                score["merge"] = self.metric(merge_ground, gt_ground, results, gs=False)

                result["gs"] = ground_gs
                result["merge"] = merge_ground
        return score, result
    
    def inference(self, epoch=None):
        count_param = sum(p.numel() for p in self.model.parameters())
        print(f"Model Params: {count_param}")
        print("\nEval Stage")
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