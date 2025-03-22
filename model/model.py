import torch
from Utils.TimeLogger import log
from Params import args
from Model import Model, vgae_encoder, vgae_decoder, vgae
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict, l2_reg_loss, Metric
from copy import deepcopy


class Coach:
    def __init__(self, handler):
        self.item_emb_trained = None
        self.user_emb_trained = None
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save): 
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')

        recall10Max = 0
        ndcg10Max = 0
        recall20Max = 0
        ndcg20Max = 0
        recall40Max = 0
        ndcg40Max = 0
        bestEpoch10 = 0
        bestEpoch20 = 0
        bestEpoch40 = 0
        stloc = 0
        patience = 0
        log('Model Initialized')

        for ep in range(stloc, args.epoch):
            # temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            temperature = 0
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(temperature)
            log(self.makePrint('Train', ep, reses, tstFlag))
            if patience > 30:
                break 
            if tstFlag:
                reses = self.testEpoch()
                if (recall10Max > reses['Recall10']):
                    patience = patience +1
                if reses['Recall10'] > recall10Max:
                    recall10Max = reses['Recall10']
                    ndcg10Max = reses['NDCG10']
                    bestEpoch10 = ep
                # log(self.makePrint('Test', ep, reses, tstFlag))
                if reses['Recall20'] > recall20Max:
                    recall20Max = reses['Recall20']
                    ndcg20Max = reses['NDCG20']
                    bestEpoch20 = ep
                # log(self.makePrint('Test', ep, reses, tstFlag))
                if reses['Recall40'] > recall40Max:
                    recall40Max = reses['Recall40']
                    ndcg40Max = reses['NDCG40']
                    bestEpoch40 = ep
                log(self.makePrint('Test', ep, reses, tstFlag))

            print('recall', recall10Max, recall20Max, recall40Max)
            print('ndcg', ndcg10Max, ndcg20Max, ndcg40Max)
            print()
        print('Best epoch : ', bestEpoch10, ' , Recall : ', recall10Max, ' , NDCG : ', ndcg10Max)
        print('Best epoch : ', bestEpoch20, ' , Recall : ', recall20Max, ' , NDCG : ', ndcg20Max)
        print('Best epoch : ', bestEpoch40, ' , Recall : ', recall40Max, ' , NDCG : ', ndcg40Max)

    def prepareModel(self):
        self.model = Model().cuda()

        encoder = vgae_encoder().cuda()
        decoder = vgae_decoder().cuda()
        self.generator_1 = vgae(encoder, decoder).cuda()

        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def trainEpoch(self, temperature):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        generate_loss_1, bpr_loss, im_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0
        steps = trnLoader.dataset.__len__() // args.batch

        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()  
            socialadj = deepcopy(self.handler.torchSocialAdj)

            data2 = self.generator_generate(self.generator_1)
            

            self.opt.zero_grad()
            self.opt_gen_1.zero_grad()

            ancs, poss, negs = tem 
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()


            out2, out2_ = self.model.forward_fusion2(data2, socialadj, 1, 0.1)    
            out1, out1_ = self.model.forward_fusion2(data, socialadj, 2, 0.1)

            # loss = self.model.loss_graphcl_1(out1, out2,social_out, ancs, poss,extracted_user_neighbors_dict).mean() * args.ssl_reg
            loss_user, loss_item = self.model.loss_graphcl_2(out1, out2, ancs, poss)
            # loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
            loss = loss_user.mean() * args.ssl_reg + loss_item.mean() * args.ssl_reg
            im_loss += float(loss)  
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # info bottleneck   
            

            _out2 = self.model.forward_graphcl(data2)
            _out1 = self.model.forward_graphcl(data)

            loss_ib = self.model.loss_graphcl(_out1, out1_.detach(), ancs, poss) + self.model.loss_graphcl(_out2,
                                                                                                          out2_.detach(),
                                                                                                          ancs, poss)
            loss = loss_ib.mean() * args.ib_reg
            ib_loss += float(ib_loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # BPR
            usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            pos_score = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
            neg_score = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
            loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
            bprLoss = torch.mean(loss)

            regLoss = calcRegLoss(self.model) * args.reg
            loss = bprLoss + regLoss
            bpr_loss += float(bprLoss)
            reg_loss += float(regLoss) 
            loss.backward()

            loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)

            loss = loss_1
            generate_loss_1 += float(loss_1)
            loss.backward() 

            self.opt.step()
            self.opt_gen_1.step()

            with torch.no_grad():
                self.user_emb_trained, self.item_emb_trained = self.model.forward_gcn(self.handler.torchBiAdj)


            log('Step %d/%d: gen 1 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f  ' % (
                i,
                steps,
                generate_loss_1,
                bpr_loss,
                im_loss,
                ib_loss,
                reg_loss,
            ), save=False, oneline=True)

        ret = dict()
        ret['Gen_1 Loss'] = generate_loss_1 / steps
        ret['BPR Loss'] = bpr_loss / steps
        ret['IM Loss'] = im_loss / steps
        ret['IB Loss'] = ib_loss / steps
        ret['Reg Loss'] = reg_loss / steps
        ret['loss'] = bpr_loss / steps + im_loss / steps + reg_loss / steps + ib_loss / steps

        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall10, epNdcg10 = [0] * 2
        epRecall20, epNdcg20 = [0] * 2
        epRecall40, epNdcg40 = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            # usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
            usrEmbeds = self.user_emb_trained
            itmEmbeds = self.item_emb_trained
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 10e8
            _, topLocs10 = torch.topk(allPreds, args.topk10)
            recall10, ndcg10 = self.calcRes(topLocs10.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,
                                            args.topk10)
            _, topLocs20 = torch.topk(allPreds, args.topk20)
            recall20, ndcg20 = self.calcRes(topLocs20.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,
                                            args.topk20)
            _, topLocs40 = torch.topk(allPreds, args.topk40)
            recall40, ndcg40 = self.calcRes(topLocs40.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,
                                            args.topk40)
            epRecall10 += recall10
            epNdcg10 += ndcg10
            epRecall20 += recall20
            epNdcg20 += ndcg20
            epRecall40 += recall40
            epNdcg40 += ndcg40
            log('Steps %d/%d: recall10 = %.2f, ndcg10 = %.2f ,recall20 = %.2f, ndcg20 = %.2f ,recall40 = %.2f, ndcg40 = %.2f         '
                % (i, steps, recall10, ndcg10, recall20, ndcg20, recall40, ndcg40, ), save=False,
                oneline=True)
        ret = dict()
        ret['Recall10'] = epRecall10 / num
        ret['NDCG10'] = epNdcg10 / num
        ret['Recall20'] = epRecall20 / num
        ret['NDCG20'] = epNdcg20 / num
        ret['Recall40'] = epRecall40 / num
        ret['NDCG40'] = epNdcg40 / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds, topk):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def generator_generate(self, generator):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.handler.torchBiAdj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view