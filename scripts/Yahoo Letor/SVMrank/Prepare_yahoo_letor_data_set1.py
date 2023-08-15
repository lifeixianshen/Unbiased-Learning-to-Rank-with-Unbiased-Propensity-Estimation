import os,sys
import random
import math
import json

list_lengths = []

def read_one_instance(feature_fin, rank_score_fin):
	feature_line = feature_fin.readline()
	score_line = rank_score_fin.readline()
	if feature_line == '' or score_line == '':
		return None, None, None, None

	arr = feature_line.strip().split(' ')
	if len(arr) < 3:
		print('thing wrong')
	label = float(arr[0])
	qid = arr[1].split(':')[1]
	features = list(arr[2:])
	score = float(score_line)
	return qid, features, label, score

def prepare_one_set(rank_cutoff, feature_path, rank_score_path, output_path ,set_name):
	with open(feature_path + set_name + '.txt') as feature_fin:
		rank_score_fin = open(rank_score_path + set_name + '.predict')

		qid_list = []
		qid_did_map, qid_feature_map, qid_label_map, qid_score_map = {}, {}, {}, {}
		qid, feature, label, score = read_one_instance(feature_fin, rank_score_fin)
		line_num = 0
		while qid != None:
			if qid not in qid_did_map:
				qid_list.append(qid)
				qid_did_map[qid], qid_feature_map[qid], qid_label_map[qid], qid_score_map[qid] = [], [], [], []
			did = f'{set_name}_{qid}_{str(line_num)}'
			qid_did_map[qid].append(did)
			qid_feature_map[qid].append(feature)
			qid_label_map[qid].append(label)
			qid_score_map[qid].append(score)
			qid, feature, label, score = read_one_instance(feature_fin, rank_score_fin)
			line_num += 1
	rank_score_fin.close()

	#generate rank lists with rank cutoff
	qid_initial_rank_map, qid_gold_rank_map = {}, {}
	for qid in qid_list:
		scores = qid_score_map[qid]
		rank_length = min(rank_cutoff, len(scores))
		list_lengths.append(rank_length)
		#qid_initial_rank_map[qid] store the indexes to raw data
		qid_initial_rank_map[qid] = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:rank_length]
		labels = [qid_label_map[qid][idx] for idx in qid_initial_rank_map[qid]]
		#qid_gold_rank_map[qid] store the rerank indexes to qid_initial_rank_map[qid]
		qid_gold_rank_map[qid] = sorted(range(len(labels)), key=lambda k: labels[k], reverse=True)

	with open(output_path + set_name + '.qrels','w') as qrel_fout:
		initial_trec_fout = open(output_path + set_name + '.trec.init_list','w')
		gold_trec_fout = open(output_path + set_name + '.trec.gold_list','w')
		for qid in qid_list:
			for i in xrange(len(qid_initial_rank_map[qid])):
				idx = qid_initial_rank_map[qid][i]
							#qrel_fout.write(qid + ' 0 ' + qid_did_map[qid][idx] + ' ' 
							#				+ str(int(qid_label_map[qid][idx])) + '\n')
				initial_trec_fout.write(
					(
						f'{qid} Q0 {qid_did_map[qid][idx]} {str(i + 1)} {str(qid_score_map[qid][idx])}'
						+ ' RankSVM\n'
					)
				)
				gold_idx = qid_initial_rank_map[qid][qid_gold_rank_map[qid][i]]
				gold_trec_fout.write(
					(
						f'{qid} Q0 {qid_did_map[qid][gold_idx]} {str(i + 1)} {str(qid_label_map[qid][gold_idx])}'
						+ ' Gold\n'
					)
				)
					#output qrels
			for i in xrange(len(qid_did_map[qid])):
				qrel_fout.write(
					(f'{qid} 0 {qid_did_map[qid][i]} {int(qid_label_map[qid][i])}' + '\n')
				)
	initial_trec_fout.close()
	gold_trec_fout.close()

	with open(output_path + set_name + '.feature','w') as feature_fout:
		with open(output_path + set_name + '.init_list','w') as initial_rank_fout:
			gold_rank_fout = open(output_path + set_name + '.gold_list','w')
			weight_fout = open(output_path + set_name + '.weights','w')
			initial_score_fout = open(output_path + set_name + '.initial_scores','w')
			line_num = 0
			for qid in qid_list:
				initial_rank_fout.write(qid)
				gold_rank_fout.write(qid)
				weight_fout.write(qid)
				initial_score_fout.write(qid)
				line_num_scope = list(
					xrange(line_num, line_num + len(qid_initial_rank_map[qid]))
				)
				for i in xrange(len(qid_initial_rank_map[qid])):
								#gold_rank_fout.write(' ' + str(line_num_scope[qid_gold_rank_map[qid][i]]))
					gold_rank_fout.write(f' {str(qid_gold_rank_map[qid][i])}')
					initial_rank_fout.write(f' {str(line_num)}')
					idx = qid_initial_rank_map[qid][i]
					weight_fout.write(f' {str(qid_label_map[qid][idx])}')
					initial_score_fout.write(f' {str(qid_score_map[qid][idx])}')
					feature_fout.write(qid_did_map[qid][idx])
					for x in qid_feature_map[qid][idx]:
						#svmlight format feature index starts from 1, but we need it to start from 0
						arr = x.split(':')
						feature_fout.write(f' {str(int(arr[0]) - 1)}:{arr[1]}')
					feature_fout.write('\n')
					line_num += 1

				initial_rank_fout.write('\n')
				gold_rank_fout.write('\n')
				weight_fout.write('\n')
				initial_score_fout.write('\n')
		gold_rank_fout.close()
	weight_fout.close()
	initial_score_fout.close()


def main():
	YAHOO_DATA_PATH = f'{sys.argv[1]}/set1.'
	MODEL_NAME = 'SVMrank'
	DATASET_NAME = 'set1'
	INITIAL_RANK_PATH = sys.argv[2]
	OUTPUT_PATH = sys.argv[3]
	RANK_CUT = sys.argv[4]
	SET_NAME = ['train','test','valid']
	FEATURE_DIM = 700
	#SET_NAME = ['valid']


	for set_name in SET_NAME:
		if not os.path.exists(OUTPUT_PATH + set_name + '/'):
			os.makedirs(OUTPUT_PATH + set_name + '/')
		prepare_one_set(RANK_CUT, YAHOO_DATA_PATH, INITIAL_RANK_PATH, OUTPUT_PATH + set_name + '/', set_name)

	settings = {'embed_size': FEATURE_DIM, 'rank_cutoff': RANK_CUT}
	with open(f'{OUTPUT_PATH}settings.json', 'w') as set_fout:
		json.dump(settings, set_fout)
	print('Longest list length %d' % (max(list_lengths)))
	print('Average list length %d' % (sum(list_lengths) / float(len(list_lengths))))

if __name__ == "__main__":
	main()


