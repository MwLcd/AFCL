import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = './Datasets/yelp/'
		elif args.data == 'lastfm':
			predir = './Datasets/lastFM/'
		elif args.data == 'beer':
			predir = './Datasets/beerAdvocate/'
		elif args.data == 'ciao':
			predir = './Datasets/ciao/'
		elif args.data == 'doubanbook':
			predir = './Datasets/doubanbook/'
		elif args.data == 'epinions':
			predir = './Datasets/epinions/'
		elif args.data == 'douban':
			predir = './Datasets/douban/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.socialfile = predir + 'ufMat_1.pkl'
		self.rsocialfile = predir + 'ufsMat_dict.pkl'

	def loadOneFile(self, filename):     #加载数据集，并将数据集的矩阵格式转化为coo格式
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):  #归一化邻接矩阵
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt) #将处理过的度数组转换为对角矩阵
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		#创建用户-物品邻接矩阵，并将此矩阵归一化，并转化为pytorch适合处理的稀疏张量的形式
		#这个邻接矩阵左上角是用户-用户关系矩阵，右上角是用户-物品关系矩阵，左下角是物品-用户关系矩阵，右下角是物品-物品关系矩阵
		#在这个矩阵中，a关系矩阵和b关系矩阵是空的
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)  #归一化过程，A=DAD


		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def makesocialAdj(self, mat):
		mat = self.normalizeAdj(mat)   #归一化操作
		# print(mat)
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):  #加载测试集和训练集,加载用户社交网络
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		ufMat = self.loadOneFile(self.socialfile)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		self.usmat=ufMat
		args.user_social, args.user_social = ufMat.shape
		self.torchSocialAdj = self.makesocialAdj(ufMat)

		trnData = TrnData(trnMat, ufMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat, uucoomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.uurows = uucoomat.row
		self.uuDokmat = uucoomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
		self.uuNegs = np.zeros(len(self.uurows)).astype(np.int32)
	def negSampling(self):    #此处生成了负样本用于用户的训练
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg
		for i in range(len(self.uurows)):
			u = self.uurows[i]
			while True:
				uNeg = np.random.randint(args.user)
				if (u, uNeg) not in self.uuDokmat:
					break
			self.uuNegs[i] = uNeg


	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])