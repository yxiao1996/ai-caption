import sys
import pandas as pd
import json
import jieba as jb
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

"""wrapper for evaluating functions wrote for ms coco, use to evaluate Chinese caption"""
"""
Class: EvalCap()

Parameters:
	refer_file: path to annotation file (json format)
        resul_fule: path to caption result file (json format)
        mode: keyword "debug" for evaluaete with fake data (fake data is fetched from annotation, which result in the ceiling performance)
	language: "english" for evaluate english caption, "chinese" for evaluate chinese caption

functions:
	parse_annotation_chn(): parse chinese annotations (usually chinese sentences have no space between words)
	evaluate_eng(): evaluate English caption
	evaluate_chn(): evaluate Chinese caption	
"""
"""Aramisbuildtoys 9/15/2017"""

class EvalCap:
    def __init__(self, refer_file='/home/xiaoy/workspace/caption/data/Flickr8k_Dataset/annotations.json', resul_file='.', mode, language="english"):
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
	"""evaluate English caption, reserve for debug"""

	if self.langu != "english":
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
	"""evaluate Chinese caption"""

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
	# debug
	# print len(res[self.refer[0]['image_name']])
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
    def parse_annotation_chn(self):
	"""parse chinese annotations"""

	if self.langu != "chinese":
	    print "parse the wrong annotation file"
	    return 0
	# space used to parse
	space = " "
	# parse caption sentence in annotation file
	for ann in self.refer:
	    caption_cut = space.join(jb.cut(ann['caption'], cut_all=False))
	    ann['caption'] = caption_cut

if __name__ == '__main__':
    test_eval_chn = EvalCap(language="chinese")
    test_eval_chn.parse_annotation_chn()
    test_eval_chn.evaluate_chn()
    #test_eval_eng = EvalCap(language="english")
    #test_eval_eng.evaluate_eng()
