import sys
import pandas as pd
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

class EvalCap:
    def __init__(self, refer_file='/home/xiaoy/workspace/caption/data/Flickr8k_Dataset/annotations.json', resul_file='', mode="debug", language="english"):
	self.eval = {}
        self.mode = mode
	self.langu = language
	# read reference file in json format	
	if language == "chinese":
            annotation_chinese = '/home/xiaoy/workspace/caption/data/flickr8k-cn/annotations.json'
	    self.refer = json.load(open(annotation_chinese, 'r'))
	else:
	    self.refer = json.load(open(refer_file, 'r'))
	# read result file in json format
	if self.mode != "debug":
	    self.resul = json.load(open(resul_file, 'r'))
	
    def evaluate_eng(self):
	if self.langu == "english":
	    print "evaluate in wrong language"	
	    return 0
	# build reference dictionary
	gts = {}
	gts = {ann['image_id']: [] for ann in self.refer}
	for ann in self.refer:
	    gts[ann['image_id']] += [ann['caption']]
			
	# build result dictionary
	res = {}
	if self.mode == "debug":   # debug
	    # for debug mode: fetch one sentence for annotation as caption
	    res = {ann['image_id']: [] for ann in self.refer}
	    for ann in self.refer:
		if len(res[ann['image_id']]) == 0:
		    res[ann['image_id']] += [ann['caption']]
		else:
		    continue
	else:
	    res = {ann['image_id']: [] for ann in self.resul}
	    for ann in self.resul:
		res[ann['image_id']] += [ann['caption']]
	
        # build scorers
	scorers = [
		(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
		(Meteor(),"METEOR"),
		(Rouge(), "ROUGE_L"),
		(Cider(), "CIDEr")
		]
	# compute and print score
	for scorer, method in scorers:
	    print 'computing %s score...'%(scorer.method())
	    score, scores = scorer.compute_score(gts, res)
	    if type(method) == list:
		for sc, scs, m in zip(score, scores, method):
		    # self.setEval(sc, m)
		    self.eval[m] = sc
		    # self.setImgToEvalImgs(scs, gts.keys(), m)
		    print "%s: %0.3f"%(m, sc)
	    else:
		# self.setEval(score, method)
		self.eval[method] = score
		# self.setImgToEvalImgs(scores, gts.keys(), method)
		print "%s: %0.3f"%(method, score)
    def evaluate_chn(self):
	if self.langu == "english":
	    print "evaluate in wrong language"	
	    return 0
        # change codec to 'utf-8'
	stdout = sys.stdout
	reload(sys)
	sys.stdout = stdout
	sys.setdefaultencoding('utf-8')
	# build reference dictionary
	gts = {}
	gts = {ann['image_name']: [] for ann in self.refer}
	for ann in self.refer:
	    gts[ann['image_name']] += [ann['caption']]
			
	# build result dictionary
	res = {}
	if self.mode == "debug":   # debug
	    # for debug mode: fetch one sentence for annotation as caption
	    res = {ann['image_name']: [] for ann in self.refer}
	    for ann in self.refer:
		if len(res[ann['image_name']]) == 0:
		    res[ann['image_name']] += [ann['caption']]
		else:
		    continue
	else:
	    res = {ann['image_name']: [] for ann in self.resul}
	    for ann in self.resul:
		res[ann['image_name']] += [ann['caption']]
	
        # build scorers
	scorers = [

		(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
		(Meteor(),"METEOR"),
		(Rouge(), "ROUGE_L"),
		(Cider(), "CIDEr")
		]
	# compute and print score
	for scorer, method in scorers:
	    print 'computing %s score...'%(scorer.method())
	    score, scores = scorer.compute_score(gts, res)
	    if type(method) == list:
		for sc, scs, m in zip(score, scores, method):
		    # self.setEval(sc, m)
		    self.eval[m] = sc
		    # self.setImgToEvalImgs(scs, gts.keys(), m)
		    print "%s: %0.3f"%(m, sc)
	    else:
		# self.setEval(score, method)
		self.eval[method] = score
		# self.setImgToEvalImgs(scores, gts.keys(), method)
		print "%s: %0.3f"%(method, score)
if __name__ == '__main__':
    test_eval = EvalCap(language="chinese")
    test_eval.evaluate_chn()
